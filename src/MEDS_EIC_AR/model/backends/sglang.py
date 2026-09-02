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
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
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

#: Positions SGLang holds back from the model's context window, on top of the prompt.
#:
#: Two layers are involved and only the second costs tokens. The tokenizer manager checks
#: ``len(input) + max_new_tokens >= context_len`` and either raises or, under
#: ``allow_auto_truncate``, clamps to ``context_len - len(input)`` — a no-op at exactly the
#: boundary. The *scheduler* is what actually trims, in ``init_req_max_new_tokens``::
#:
#:     max_req_len    = min(context_len - 1, max_token_pool_size - 1)   # tp_worker
#:     max_new_tokens = min(requested, max_req_len - len(input) - 1)    # scheduler
#:
#: which is ``context_len - len(input) - 2`` whenever the context length is the binding term.
#: That ``min`` is unconditional: no warning, and not gated on ``allow_auto_truncate``. HF's
#: ``generate``, by contrast, will fill the window exactly. Matches the measurement it was
#: derived from — on a 512-position model with a 128-token prompt, 382 new tokens is the largest
#: request honored in full, while 383 and 384 both come back as 382.
#:
#: **Two is a floor, not a guarantee.** When the KV pool is the binding term instead — a large
#: model, or a small ``mem_fraction_static`` — ``max_req_len`` is smaller than ``context_len - 1``
#: and the shortfall grows. Capping by this constant keeps the common case honest; the residual is
#: what ``allow_auto_truncate`` is left on to absorb.
#:
#: The subtraction also goes negative: at ``len(input) == context_len - 1`` the scheduler's cap is
#: ``-1``, and ``Req.check_finished`` stops on ``len(output_ids) >= max_new_tokens``, which holds
#: at zero output tokens. The request completes having generated nothing — which is why
#: :meth:`SGLangBackend._cap_max_new_tokens` raises rather than letting the rolling loop receive an
#: empty chunk it would spin on.
_SGLANG_CONTEXT_RESERVE = 2


#: Positions SGLang holds back from the *input* side of the context window.
#:
#: Distinct from :data:`_SGLANG_CONTEXT_RESERVE`, which bounds how many tokens SGLang will
#: *emit*. This one bounds how many prompt tokens it will actually *read*, and exceeding it is
#: far more damaging: SGLang does not refuse the request, it silently drops the tail of the
#: prompt and generates from the truncated prefix, so the model is conditioned on something the
#: caller never asked for. Derivation, from ``sglang==0.5.9``::
#:
#:     max_req_len       = min(context_len - 1, max_token_pool_size - 1)   # tp_worker.py
#:     max_req_input_len = max_req_len - 5                                 # tp_worker.py
#:     if len(input_ids) >= max_req_input_len: truncate or reject          # managers/utils.py
#:
#: The comparison is ``>=``, so the longest prompt SGLang reads in full is
#: ``max_req_input_len - 1``, i.e. ``context_len - 7`` whenever the context length is the
#: binding term. Measured against a 16-position model: a 9-token prompt is honored, a 10-token
#: prompt is already truncated.
#:
#: Why this is not merely SGLang's business: ``allow_auto_truncate=True`` (set unconditionally
#: below) is what turns the rejection into a silent truncation, and SGLang's own truncation
#: warning is emitted through a logger its engine init has already set to ERROR, so nothing
#: surfaces. Left unchecked, ``Model._rolling_generate``'s default ``rolling_context_size`` of
#: ``max_seq_len - 1`` sits far above this ceiling and every rolling chunk loses its prompt
#: tail — which is exactly how SGLang came to emit grammar-invalid trajectories where the HF
#: backend emitted valid ones (issue #171).
_SGLANG_INPUT_RESERVE = 7


#: Smallest attention head dimension SGLang's default FlashInfer attention backend can dispatch.
#: Below this it aborts the scheduler subprocess with ``FlashInfer Internal Error: Invalid
#: configuration``, which reaches the parent as SIGQUIT / exit ``-9`` — indistinguishable at a
#: glance from an out-of-memory kill, and so a genuinely expensive thing to debug. Measured against
#: ``sglang==0.5.9``: ``head_dim=32`` aborts, ``head_dim=64`` works. The ``triton`` attention
#: backend has no such floor.
_FLASHINFER_MIN_HEAD_DIM = 64


def _read_hf_config(hf_model_dir: Path | str) -> dict:
    """Parse an HF model directory's ``config.json``, or return ``{}`` if it can't be read.

    Degrading to an empty dict rather than raising is deliberate: every use of this config is a
    pre-flight check that improves diagnostics, and none of them is worth refusing to construct
    the backend over. A missing config costs the checks, not the run.

    Examples:
        >>> import json, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "config.json").write_text(json.dumps({"head_dim": 64}))
        ...     _read_hf_config(d)
        {'head_dim': 64}

        Missing and malformed configs both degrade to ``{}``:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     _read_hf_config(d)
        {}
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "config.json").write_text("{not json")
        ...     _read_hf_config(d)
        {}
    """
    config_fp = Path(hf_model_dir) / "config.json"
    try:
        config = json.loads(config_fp.read_text())
    except (OSError, ValueError) as e:
        logger.warning(
            f"Could not read {config_fp} ({type(e).__name__}: {e}). The SGLang backend's "
            "pre-flight checks on context length and attention head dimension will be skipped."
        )
        return {}
    return config if isinstance(config, dict) else {}


def _config_int(config: dict, key: str) -> int | None:
    """Pull an integer field out of a parsed HF config, or ``None`` if it isn't there.

    Examples:
        >>> _config_int({"head_dim": 64}, "head_dim")
        64
        >>> print(_config_int({}, "head_dim"))
        None
        >>> print(_config_int({"head_dim": None}, "head_dim"))
        None
        >>> print(_config_int({"head_dim": "64"}, "head_dim"))
        None

        ``bool`` is a subclass of ``int`` but never a valid answer here:

        >>> print(_config_int({"head_dim": True}, "head_dim"))
        None
    """
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _check_head_dim_against_attention_backend(head_dim: int | None, engine_kwargs: dict) -> None:
    """Refuse to start SGLang on a default attention backend that cannot dispatch this head dim.

    SGLang defaults to FlashInfer, which aborts on ``head_dim`` below
    :data:`_FLASHINFER_MIN_HEAD_DIM`. The abort kills the scheduler subprocess and surfaces in the
    parent as SIGQUIT / exit ``-9``, so what is really a shape constraint reads as an
    out-of-memory kill — the single most misleading failure this backend can produce. Catching it
    here costs nothing and names both remedies.

    The check only fires when ``attention_backend`` is left unset, i.e. when *we* are the ones
    letting SGLang pick FlashInfer. A caller who names a backend explicitly has made a choice and
    is trusted with it, which also means this cannot wrongly block a future FlashInfer that lifts
    the floor.

    Args:
        head_dim: The model's attention head dimension, or ``None`` if it could not be read (in
            which case there is nothing to check).
        engine_kwargs: The kwargs about to be passed to ``sglang.Engine``.

    Raises:
        ValueError: If the head dimension is below the FlashInfer floor and no attention backend
            was named.

    Examples:
        Fine: head dim at or above the floor, or unknown, or a backend named explicitly.

        >>> _check_head_dim_against_attention_backend(64, {})
        >>> _check_head_dim_against_attention_backend(None, {})
        >>> _check_head_dim_against_attention_backend(32, {"attention_backend": "triton"})

        Refused: below the floor with the backend left to SGLang's default.

        >>> _check_head_dim_against_attention_backend(32, {})
        Traceback (most recent call last):
            ...
        ValueError: SGLang's default FlashInfer attention backend cannot dispatch
        attention_head_dim=32 ...
    """
    if head_dim is None or head_dim >= _FLASHINFER_MIN_HEAD_DIM:
        return
    if engine_kwargs.get("attention_backend") is not None:
        return
    raise ValueError(
        f"SGLang's default FlashInfer attention backend cannot dispatch attention_head_dim="
        f"{head_dim} (its floor is {_FLASHINFER_MIN_HEAD_DIM}). It would abort the scheduler "
        "subprocess with 'FlashInfer Internal Error: Invalid configuration', which reaches this "
        "process as SIGQUIT / exit -9 and looks like an out-of-memory kill rather than a shape "
        "error. Either train with lightning_module.model.gpt_kwargs.attention_head_dim>="
        f"{_FLASHINFER_MIN_HEAD_DIM}, or select an attention backend that has no such floor with "
        "backend.engine_kwargs.attention_backend=triton."
    )


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
        # ``allow_auto_truncate`` keeps window-saturating requests from being *rejected*; it is
        # not what costs tokens. The rolling loop in
        # :meth:`MEDS_EIC_AR.model.Model._rolling_generate` sets
        # ``chunk_budget = max_seq_len - prompt_len`` per call — total positions requested =
        # ``max_context_length`` exactly. HF's ``generate`` accepts that boundary (positions
        # ``0..max_pos-1`` inclusive); SGLang's tokenizer manager raises on it unless this flag is
        # set. The token loss happens further in, in the scheduler, and happens either way — see
        # ``_SGLANG_CONTEXT_RESERVE``. So this flag is what keeps boundary requests alive, and
        # :meth:`generate_chunk` is what makes the resulting shortfall visible instead of leaving
        # it to be discovered in the output shape. Non-overridable for the same reason as
        # ``skip_tokenizer_init``: turning it off converts every window-saturating chunk from a
        # shortfall into a crash mid-run.
        self._engine_kwargs["allow_auto_truncate"] = True
        hf_config = _read_hf_config(hf_model_dir)
        self._context_len = _config_int(hf_config, "max_position_embeddings")
        if self._context_len is None:
            logger.warning(
                f"No integer ``max_position_embeddings`` in {Path(hf_model_dir) / 'config.json'}; "
                "SGLang's own auto-truncation will apply instead, which clamps silently."
            )
        _check_head_dim_against_attention_backend(_config_int(hf_config, "head_dim"), self._engine_kwargs)
        # Log the ceiling once at construction rather than per call — the rolling loop calls
        # ``generate_chunk`` many times with the same shape, and a per-call warning would bury
        # the run's real output.
        self._logged_context_clamp = False
        self._logged_prompt_ceiling = False
        self._engine = sgl_module.Engine(model_path=str(hf_model_dir), **self._engine_kwargs)
        self._is_shutdown = False
        atexit.register(self.shutdown)

    @property
    def max_prompt_len(self) -> int | None:
        """Longest prompt SGLang will read in full, or ``None`` if the context length is unknown.

        Prompts longer than this are silently truncated by the engine rather than rejected (see
        :data:`_SGLANG_INPUT_RESERVE`), so callers that build their own prompt windows -- notably
        :meth:`MEDS_EIC_AR.model.model.Model._rolling_generate` -- must clamp to this rather than
        to the model's raw context length. ``None`` means "unknown, don't clamp", which restores
        the pre-fix behavior rather than guessing a ceiling that might be wrong.
        """
        if self._context_len is None:
            return None
        return max(self._context_len - _SGLANG_INPUT_RESERVE, 0)

    def _check_prompt_len(self, prompt_len: int) -> None:
        """Refuse a prompt SGLang would silently truncate.

        Raising beats the alternative: SGLang would accept the request, drop the tail of the
        prompt, and return tokens conditioned on a prefix the caller never asked for. That is
        invisible at the API surface -- the output has the right shape and the right length --
        and it is what produced grammar-invalid trajectories in issue #171.
        """
        ceiling = self.max_prompt_len
        if ceiling is None or prompt_len <= ceiling:
            return
        raise ValueError(
            f"SGLang would silently truncate this {prompt_len}-token prompt: on a "
            f"{self._context_len}-position model it reads at most {ceiling} prompt tokens "
            f"(max_req_input_len = min(context_len - 1, kv_pool - 1) - 5, compared with >=). "
            "It does not reject the request -- it drops the tail and generates from the "
            "remaining prefix, so the model would be conditioned on something you did not ask "
            "for. The HF backend has no such limit. Lower the prompt window to at most "
            f"{ceiling} tokens (for rolling generation, set "
            f"rolling_generation.rolling_context_size<={ceiling})."
        )

    def _cap_max_new_tokens(self, requested: int, prompt_len: int) -> int:
        """Clamp ``requested`` to what SGLang will actually honor for a ``prompt_len``-token prompt.

        Returns ``requested`` unchanged when the model's context length could not be read (in
        which case we fall back to the engine's own auto-truncate) or when the request already
        fits. Otherwise returns the ceiling and logs the shortfall once.

        Raises:
            ValueError: If the prompt is so long that no new tokens fit under the ceiling. This is
                reachable from the rolling loop, whose default ``rolling_context_size`` of
                ``max_seq_len - 1`` leaves only one position for new tokens — one fewer than the
                reserve needs. Raising beats returning an empty chunk, which the rolling loop
                would read as "no progress" and spin on forever.
        """
        if self._context_len is None:
            return requested

        allowed = self._context_len - prompt_len - _SGLANG_CONTEXT_RESERVE
        if allowed >= requested:
            return requested

        if allowed <= 0:
            raise ValueError(
                f"SGLang cannot generate any new tokens for a {prompt_len}-token prompt against a "
                f"{self._context_len}-position context window: its scheduler caps max_new_tokens "
                f"at max_req_len - len(input) - 1, which holds back {_SGLANG_CONTEXT_RESERVE} "
                f"positions and leaves {allowed} here. At or past this point SGLang returns an "
                "empty completion rather than an error, which the rolling loop would read as no "
                "progress and spin on. HF's generate accepts this prompt, so the ceiling is "
                "SGLang-specific. Set rolling_generation.rolling_context_size to at most "
                f"{self._context_len - _SGLANG_CONTEXT_RESERVE - 1} so every chunk has room for at "
                "least one new token."
            )

        if not self._logged_context_clamp:
            self._logged_context_clamp = True
            logger.warning(
                f"SGLang chunk budget clamped from {requested} to {allowed} new tokens for a "
                f"{prompt_len}-token prompt: its scheduler holds back "
                f"{_SGLANG_CONTEXT_RESERVE} of the model's {self._context_len} positions, where "
                "HF's generate would use the full window. Generated trajectories will be "
                "correspondingly shorter than the HF backend's on window-saturating chunks. "
                "Logged once per backend instance."
            )
        return allowed

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

        ``generation_config.max_new_tokens`` is a request, not a guarantee: SGLang reserves
        ``_SGLANG_CONTEXT_RESERVE`` positions of the model's context window that HF's ``generate``
        would happily use, so a window-saturating chunk comes back that many tokens shorter. The
        shortfall is applied here and logged rather than left to the engine to swallow; see
        :meth:`_cap_max_new_tokens`.
        """
        stripped = {k: v for k, v in kwargs.items() if k in _HF_ONLY_KWARGS}
        if stripped:
            logger.debug(
                f"SGLangBackend stripped {sorted(stripped)} from generate_chunk kwargs — "
                "these are HF-only and not accepted by the SGLang engine."
            )
        forwarded = {k: v for k, v in kwargs.items() if k not in _HF_ONLY_KWARGS}

        prompts = _strip_padding_to_lists(input_ids, attention_mask)

        # Apply SGLang's context ceiling ourselves, against the longest prompt in the batch (the
        # binding one, since ``max_new_tokens`` is shared across the whole call). Doing it here
        # rather than leaving it to ``allow_auto_truncate`` is what makes the shortfall visible;
        # see ``_SGLANG_CONTEXT_RESERVE``.
        longest_prompt = max((len(p) for p in prompts), default=0)
        # Input ceiling first: an over-long prompt is silently truncated by the engine, which is
        # strictly worse than the output-side shortfall handled just below.
        self._check_prompt_len(longest_prompt)
        max_new_tokens = self._cap_max_new_tokens(generation_config.max_new_tokens, longest_prompt)

        # Map HF ``GenerationConfig`` → SGLang sampling-params dict. Intentional translations:
        #   - ``do_sample=False`` → ``temperature=0.0`` regardless of the caller's configured
        #     temperature (SGLang uses ``temperature=0`` as its greedy signal; no separate
        #     boolean). When ``do_sample=True`` the caller's ``generation_config.temperature``
        #     is honored. This matches HF's behavior: ``temperature`` is a no-op when
        #     ``do_sample=False``.
        #   - ``eos_token_id`` → ``stop_token_ids=[eos]``. SGLang supports a list; we pass a
        #     single-element list to mirror HF's single-eos semantics here.
        #   - ``top_k`` / ``top_p`` are not translated, and that is now a *deliberate no-op
        #     rather than a gap*. ``Model._generate_chunk`` (the only call site) pins them to
        #     ``top_k=0`` / ``top_p=1.0``, i.e. no truncation, because untruncated ancestral
        #     sampling is the only sampling mode this repo supports. SGLang's own defaults
        #     (``top_k=1073741824``, ``top_p=1.0``) are likewise non-truncating, so both
        #     backends already sample from the same law and forwarding the fields would change
        #     nothing. If a truncated mode is ever introduced, it must be plumbed through here
        #     at the same time — otherwise the two backends would silently disagree.
        #
        # Pass a plain ``dict`` (rather than ``sglang.SamplingParams``) because the stable
        # public shape of ``Engine.generate(sampling_params=...)`` is ``Dict | List[Dict]``
        # (see ``sglang/srt/entrypoints/engine.py``). The ``SamplingParams`` class lives under
        # ``sglang.srt.sampling.sampling_params`` and is not exported at the top level in
        # v0.5.x — older revisions of this file referenced ``self._sgl.SamplingParams`` and
        # crashed with ``AttributeError`` on every first call against a real engine.
        if generation_config.do_sample:
            configured_temp = getattr(generation_config, "temperature", None)
            temperature = float(configured_temp) if configured_temp is not None else 1.0
        else:
            temperature = 0.0
        sampling_params: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stop_token_ids": (
                [generation_config.eos_token_id] if generation_config.eos_token_id is not None else None
            ),
        }

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
            if len(row_tokens) > max_new_tokens:
                raise RuntimeError(
                    f"SGLang output[{i}] returned {len(row_tokens)} tokens but "
                    f"``max_new_tokens={max_new_tokens}`` — the engine appears "
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
