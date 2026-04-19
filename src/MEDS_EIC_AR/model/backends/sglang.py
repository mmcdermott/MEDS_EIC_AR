"""SGLang implementation of :class:`GenerationBackend` (issue #88).

SGLang exposes an offline ``Engine`` (in-process, not an HTTP server) whose ``generate`` takes a
batch of token-id prompts and returns, per request, a dict of newly generated tokens. The field
name has moved across SGLang versions — current ``v0.5.x`` uses ``"output_ids"``, earlier
releases used ``"token_ids"``. We probe both (``_SGLANG_OUTPUT_KEYS`` below) and raise a loud
``KeyError`` if neither is present. The adapter wraps all of this to match our protocol's
tensor-in / tensor-out contract and to enforce the per-row post-EOS padding invariant the
rolling loop relies on.

Why this file is non-trivial despite the thin public surface:

1. **SGLang ships its own scheduler subprocess.** Every ``Engine(...)`` constructor forks a
   child process that owns GPU memory and the actual model weights. If the CLI process exits
   without calling ``engine.shutdown()``, the child can linger and hold GPU — a real failure
   mode on shared machines. We register an ``atexit`` handler per instance so shutdown happens
   even on unhandled exceptions in the parent.
2. **SGLang's engine accepts a narrower kwarg set than HF's generate.** The
   :class:`~MEDS_EIC_AR.model.backends.base.GenerationBackend` protocol requires backends to
   only forward options their engine accepts. HF-specific keys like ``logits_processor`` /
   ``stopping_criteria`` are silently stripped here because SGLang's
   ``SamplingParams`` can't consume them; forwarding them would raise ``TypeError`` inside the
   engine subprocess, which surfaces as a broken-pipe in the parent and is very hard to debug.
3. **Left-padded prompt tensors must be compressed to ragged Python lists.** SGLang's Python
   API takes ``list[list[int]]`` for ``input_ids`` (one token-id list per prompt, no padding);
   we use the caller's ``attention_mask`` to drop pad positions *before* handing off to SGLang.
   Forwarding padded tensors would make SGLang treat pad ids as real input tokens.
4. **Variable-length outputs must be right-padded back to a dense ``[B, new_len]`` tensor.**
   SGLang returns a ragged list-of-lists keyed by per-row stopping time. We pad on the right
   with ``pad_token_id`` and emit a dense tensor; this also satisfies the per-row post-EOS
   invariant (anything after EOS is pad). See ``_pad_right_to_tensor``.

Gotchas accounted for:

- **Return format might include the prompt.** Historically SGLang has changed whether
  ``token_ids`` means "prompt + new" or "new only" between releases. The version validated
  at the time this was written returns new-only, which is what our contract wants. A smoke assertion in
  the unit test catches a regression loudly.
- **``skip_tokenizer_init=True``** is essential — MEDS codes are already token ids, not text,
  and SGLang's tokenizer path would otherwise try to load a tokenizer from the HF directory
  and either fail or do pointless work.
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import GenerationConfig

logger = logging.getLogger(__name__)


#: Kwargs that flow through ``Model.generate(**kwargs)`` and are meaningful only to HF's
#: ``generate`` — SGLang's ``Engine.generate`` / ``SamplingParams`` don't accept them. Stripping
#: them here honors the :class:`GenerationBackend` protocol contract ("only forward options
#: supported by the active engine") rather than relying on the engine to reject them (which
#: would surface as a broken-pipe from the scheduler subprocess).
_HF_ONLY_KWARGS: frozenset[str] = frozenset(
    {
        "logits_processor",
        "stopping_criteria",
        "prefix_allowed_tokens_fn",
        "streamer",
        "assistant_model",
        "negative_prompt_ids",
        "negative_prompt_attention_mask",
    }
)

#: The field names SGLang has used for the newly-generated token ids in its output dicts.
#: ``output_ids`` is the current (v0.5.x) name; ``token_ids`` was used in older releases.
#: We probe for both so a version bump in either direction doesn't silently produce empty rows;
#: if neither key is found we raise loudly (see ``generate_chunk``).
_SGLANG_OUTPUT_KEYS: tuple[str, ...] = ("output_ids", "token_ids")


def _strip_padding_to_lists(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> list[list[int]]:
    """Convert a padded ``[B, L_in]`` prompt tensor into a ragged list-of-lists of real tokens.

    SGLang's ``Engine.generate`` takes one token-id list per prompt and doesn't understand an
    external attention mask; pad positions must be dropped *before* the call. The padding
    direction is caller-defined — this repo uses left-padding per
    ``configs/datamodule/generate_trajectories.yaml``, but rolling-chunk prompts can also
    contain right-side padding for samples that already finished in a prior chunk, so we
    don't assume left-only.

    If ``attention_mask`` is ``None`` we treat every position as real (mirrors HF's behavior
    when the caller omits the mask). In practice ``Model._generate_chunk`` always passes a
    mask.

    Examples:
        >>> import torch
        >>> input_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 0]], dtype=torch.long)
        >>> mask = torch.tensor([[False, False, True, True, True],
        ...                       [False, True,  True, True, False]])
        >>> _strip_padding_to_lists(input_ids, mask)
        [[1, 2, 3], [4, 5, 6]]

        With no mask, nothing is stripped:

        >>> _strip_padding_to_lists(input_ids, None)
        [[0, 0, 1, 2, 3], [0, 4, 5, 6, 0]]
    """
    if attention_mask is None:
        return [row.tolist() for row in input_ids]

    cpu_ids = input_ids.detach().cpu()
    cpu_mask = attention_mask.detach().cpu().to(torch.bool)
    return [row[m].tolist() for row, m in zip(cpu_ids, cpu_mask, strict=True)]


def _pad_right_to_tensor(
    new_tokens_per_row: list[list[int]],
    *,
    pad_value: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Right-pad a ragged list-of-lists into a ``[B, new_len]`` tensor.

    ``new_len`` is the max length across rows. Shorter rows are padded with ``pad_value`` on
    the right, which directly satisfies the :class:`GenerationBackend` per-row post-EOS
    invariant: SGLang stops each row at the first EOS (or at ``max_new_tokens``), so anything
    past that point is naturally pad.

    Examples:
        >>> import torch
        >>> out = _pad_right_to_tensor(
        ...     [[1, 2, 3], [4, 5]],
        ...     pad_value=0,
        ...     device="cpu",
        ... )
        >>> out
        tensor([[1, 2, 3],
                [4, 5, 0]])

        Empty rows are handled (``[B, 0]`` output when every row is empty):

        >>> _pad_right_to_tensor([[], []], pad_value=0, device="cpu").shape
        torch.Size([2, 0])
    """
    batch_size = len(new_tokens_per_row)
    new_len = max((len(row) for row in new_tokens_per_row), default=0)
    out = torch.full((batch_size, new_len), pad_value, dtype=dtype, device=device)
    for i, row in enumerate(new_tokens_per_row):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=dtype, device=device)
    return out


class SGLangBackend:
    """SGLang-engine implementation of :class:`GenerationBackend`.

    Accepts an HF on-disk model directory (Llama-format since #108); the companion helper
    :func:`MEDS_EIC_AR.model.backends.export.export_lightning_to_hf_dir` materializes a
    Lightning checkpoint into such a directory.

    Args:
        hf_model_dir: Path to an HF-format model directory (``config.json`` + weight shards).
        engine_kwargs: Forwarded to ``sglang.Engine``. Typical keys:
            ``mem_fraction_static``, ``max_running_requests``, ``tp_size``,
            ``disable_cuda_graph``. ``skip_tokenizer_init=True`` is always set internally and
            cannot be overridden here — we generate from token ids, never text.
        sgl_module: Test-only injection point for a fake ``sglang`` module. ``None`` means
            lazy-import the real package. This is the only way we can unit-test the backend
            without the optional ``sglang`` dep installed; production code should never pass
            this argument.

    Notes on process lifecycle:
        ``sgl.Engine(...)`` forks a scheduler subprocess. We register an ``atexit`` hook so
        ``shutdown()`` runs even on unhandled parent-process exceptions. Callers can also call
        :meth:`shutdown` / use the backend as a context manager for deterministic teardown.
    """

    def __init__(
        self,
        hf_model_dir: Path | str,
        *,
        engine_kwargs: dict[str, Any] | None = None,
        sgl_module: Any | None = None,
    ):
        if sgl_module is None:
            try:
                import sglang as sgl_module
            except ImportError as e:  # pragma: no cover — exercised only when dep absent
                raise ImportError(
                    "SGLangBackend requires the optional ``sglang`` dependency. "
                    "Install with ``pip install MEDS_EIC_AR[sglang]`` or ``uv sync --extra sglang``."
                ) from e

        self._sgl = sgl_module
        self._engine_kwargs = dict(engine_kwargs or {})
        # ``skip_tokenizer_init`` is load-bearing: MEDS code ids are already tokens, and
        # leaving tokenizer init on would make SGLang try to load a tokenizer from the HF dir
        # (which :func:`export_lightning_to_hf_dir` deliberately stubs rather than populates).
        # Overwrite unconditionally rather than ``setdefault`` — the class docstring promises
        # this cannot be overridden, and a caller passing ``engine_kwargs={"skip_tokenizer_init":
        # False}`` would otherwise silently break the pipeline.
        self._engine_kwargs["skip_tokenizer_init"] = True
        self._engine = sgl_module.Engine(model_path=str(hf_model_dir), **self._engine_kwargs)
        self._is_shutdown = False
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Terminate the SGLang scheduler subprocess.

        Idempotent — safe to call more than once. Called automatically via ``atexit`` on
        parent-process exit; callers wanting deterministic teardown (e.g. in tests) can call
        this directly.
        """
        if self._is_shutdown:
            return
        try:
            self._engine.shutdown()
        except Exception as e:  # pragma: no cover — best-effort cleanup on exit
            logger.warning(f"SGLangBackend.shutdown() raised {type(e).__name__}: {e}")
        finally:
            self._is_shutdown = True
            # Unregister the atexit handler so it doesn't accumulate in long-running processes or
            # test suites that create many backends. ``atexit.unregister`` is idempotent (safe if
            # the handler was already removed) and doesn't raise.
            atexit.unregister(self.shutdown)

    def __enter__(self) -> SGLangBackend:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    def generate_chunk(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        generation_config: GenerationConfig,
        **kwargs,
    ) -> torch.Tensor:
        """Run one SGLang engine pass and return only the newly generated tokens.

        Matches the :class:`GenerationBackend.generate_chunk` contract: returns
        ``[B, new_len]`` with ``new_len <= generation_config.max_new_tokens``. Rows that hit
        EOS before ``max_new_tokens`` have pad on the right thanks to the right-pad helper.

        HF-only kwargs (``logits_processor``, ``stopping_criteria``, …) are stripped before
        forwarding so a caller passing a cross-backend kwargs dict doesn't blow up the engine
        subprocess. The stripped kwargs are logged at debug level.
        """
        stripped = {k: v for k, v in kwargs.items() if k in _HF_ONLY_KWARGS}
        if stripped:
            logger.debug(
                f"SGLangBackend stripped {sorted(stripped)} from generate_chunk kwargs — "
                "these are HF-only and not accepted by the SGLang engine."
            )
        forwarded = {k: v for k, v in kwargs.items() if k not in _HF_ONLY_KWARGS}

        prompts = _strip_padding_to_lists(input_ids, attention_mask)

        # Map HF ``GenerationConfig`` → SGLang ``SamplingParams``. Intentional translations:
        #   - ``do_sample=False`` → ``temperature=0.0`` regardless of the caller's configured
        #     temperature (SGLang uses ``temperature=0`` as its greedy signal; no separate
        #     boolean). When ``do_sample=True`` the caller's ``generation_config.temperature``
        #     is honored. This matches HF's behavior: ``temperature`` is a no-op when
        #     ``do_sample=False``.
        #   - ``eos_token_id`` → ``stop_token_ids=[eos]``. SGLang supports a list; we pass a
        #     single-element list to mirror HF's single-eos semantics here.
        # ``top_p``/``top_k`` are deliberately not translated yet — none of today's callers
        # set them (see ``Model._generate_chunk`` — the only call site), and translating them
        # is properly part of #82's logits-processor work.
        if generation_config.do_sample:
            configured_temp = getattr(generation_config, "temperature", None)
            temperature = float(configured_temp) if configured_temp is not None else 1.0
        else:
            temperature = 0.0
        sampling_params = self._sgl.SamplingParams(
            max_new_tokens=generation_config.max_new_tokens,
            temperature=temperature,
            stop_token_ids=[generation_config.eos_token_id]
            if generation_config.eos_token_id is not None
            else None,
        )

        outputs = self._engine.generate(input_ids=prompts, sampling_params=sampling_params, **forwarded)
        # SGLang returns a list of dicts, one per prompt. The new-tokens field has historically
        # lived under the ``output_ids`` key (v0.5.x) or ``token_ids`` (older releases). Probe
        # both and prefer ``output_ids`` when present. The mock tests assert both variants are
        # accepted so a version bump that flips the field name doesn't silently regress.
        # Raise explicitly rather than falling back to ``[]`` so a future SGLang version that
        # uses yet another field name fails loudly rather than producing silent all-pad outputs.
        new_tokens_per_row = []
        for i, out in enumerate(outputs):
            tokens = next((out[k] for k in _SGLANG_OUTPUT_KEYS if k in out), None)
            if tokens is None:
                raise KeyError(
                    f"SGLang output[{i}] has none of the expected token-id keys {_SGLANG_OUTPUT_KEYS}. "
                    f"Got keys: {sorted(out)}. This may indicate a SGLang version mismatch — "
                    "check whether the installed version returns tokens under a different field name."
                )
            row_tokens = list(tokens)
            # Defensive: the ``GenerationBackend`` contract is that ``generate_chunk`` returns
            # *new-only* tokens, and SGLang's current ``output_ids`` key holds new tokens only.
            # A future SGLang version that flips back to "prompt + new" semantics (as older
            # releases did) would silently corrupt the rolling loop — the extra prompt tokens
            # would be fed back as "newly generated" on the next chunk, duplicating the prompt
            # in the accumulated sequence. Fail loudly instead: the only way a row's length can
            # exceed ``max_new_tokens`` under new-only semantics is if the engine included the
            # prompt.
            if len(row_tokens) > generation_config.max_new_tokens:
                raise RuntimeError(
                    f"SGLang output[{i}] returned {len(row_tokens)} tokens but "
                    f"``max_new_tokens={generation_config.max_new_tokens}`` — the engine appears "
                    "to be returning the prompt prefix plus new tokens rather than new-only. "
                    "This breaks the GenerationBackend contract and would silently corrupt the "
                    "rolling loop. Check the installed SGLang version's ``Engine.generate`` "
                    "return-format semantics and, if needed, strip the prompt here before "
                    "emitting."
                )
            new_tokens_per_row.append(row_tokens)
        return _pad_right_to_tensor(
            new_tokens_per_row,
            pad_value=generation_config.pad_token_id,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
