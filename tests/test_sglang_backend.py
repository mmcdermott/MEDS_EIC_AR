"""Mock-based unit tests for ``SGLangBackend`` (issue #88).

These tests exercise the adapter's shape/padding/kwarg-handling logic without needing the
optional ``sglang`` dependency installed or a GPU. Every test injects a fake ``sglang``
module via the backend's ``sgl_module`` constructor hook; a companion gated integration
test (under ``tests/grammar/``, in a later PR) will run the real engine end-to-end against
the grammar suite on a GPU runner.

What's deliberately NOT tested here:

- Real SGLang correctness (that's the GPU-gated integration test's job).
- Cross-backend byte-parity vs HF (gotcha §7 of #88 — floating-point drift makes exact
  parity flaky; the gated grammar test compares *properties*, not tokens).
- The ``atexit`` shutdown hook (hard to assert cleanly in-process; covered by ``shutdown()``
  idempotence test below).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import GenerationConfig

from MEDS_EIC_AR.model.backends import GenerationBackend, SGLangBackend
from MEDS_EIC_AR.model.backends.sglang import _pad_right_to_tensor, _strip_padding_to_lists


class _FakeSamplingParams:
    """Records everything passed to ``sgl.SamplingParams(...)`` so tests can assert on it."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs


class _FakeEngine:
    """A fake ``sgl.Engine`` that returns pre-programmed per-prompt token lists.

    Records every ``generate`` call so tests can assert on the exact kwargs forwarded (in
    particular, that HF-only kwargs have been stripped by the backend before reaching here).
    """

    def __init__(self, model_path: str, **engine_kwargs: Any):
        self.model_path = model_path
        self.engine_kwargs = engine_kwargs
        self.generate_calls: list[dict] = []
        # Tests inject what the next ``generate`` call should return.
        self._next_outputs: list[dict] | None = None
        self.shutdown_calls = 0

    def set_next_outputs(self, outputs: list[dict]) -> None:
        self._next_outputs = outputs

    def generate(self, *, input_ids: list[list[int]], sampling_params: _FakeSamplingParams, **kw):
        self.generate_calls.append(
            {
                "input_ids": [list(row) for row in input_ids],
                "sampling_params": sampling_params.kwargs,
                "extra_kwargs": dict(kw),
            }
        )
        if self._next_outputs is None:
            raise AssertionError("Test forgot to set_next_outputs before calling generate.")
        return self._next_outputs

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeSGLModule:
    """Matches the tiny surface of the real ``sglang`` module that the backend touches."""

    def __init__(self):
        self.last_engine: _FakeEngine | None = None

    def Engine(self, *, model_path: str, **engine_kwargs: Any) -> _FakeEngine:  # noqa: N802 — mirrors the real sglang.Engine class name
        eng = _FakeEngine(model_path=model_path, **engine_kwargs)
        self.last_engine = eng
        return eng

    SamplingParams = _FakeSamplingParams


def _make_backend() -> tuple[SGLangBackend, _FakeSGLModule]:
    """Helper: build a backend wired to a fresh fake sglang module."""
    fake = _FakeSGLModule()
    backend = SGLangBackend("/tmp/ignored_model_path", sgl_module=fake)
    return backend, fake


# ---------------------------------------------------------------------------
# Protocol / structural contract
# ---------------------------------------------------------------------------


def test_sglang_backend_satisfies_protocol():
    """``SGLangBackend`` must satisfy the ``GenerationBackend`` runtime_checkable protocol.

    Key correctness: PR 2 of issue #88 explicitly says the SGLang adapter must not be forced
    to inherit from a repo-owned base class. The ``@runtime_checkable`` check verifies the
    structural match — not just presence of ``generate_chunk``, but at a shape compatible
    with what ``Model._generate_chunk`` calls it with.
    """
    backend, _ = _make_backend()
    assert isinstance(backend, GenerationBackend)


def test_engine_receives_skip_tokenizer_init_by_default():
    """``skip_tokenizer_init=True`` must be set on every Engine construction.

    MEDS codes are already token ids, not text, so SGLang's tokenizer path would be pointless work at best and
    a crash at worst (our HF export writes a stub tokenizer config, not a real tokenizer). This is the single
    most important engine kwarg; losing it would be a silent performance regression plus a real crash risk.
    """
    _, fake = _make_backend()
    assert fake.last_engine is not None
    assert fake.last_engine.engine_kwargs["skip_tokenizer_init"] is True


def test_engine_caller_can_override_other_engine_kwargs():
    """Caller-provided ``engine_kwargs`` should be forwarded to ``sgl.Engine(...)`` unchanged."""
    fake = _FakeSGLModule()
    backend = SGLangBackend(
        "/tmp/x",
        engine_kwargs={"mem_fraction_static": 0.7, "tp_size": 2},
        sgl_module=fake,
    )
    del backend  # quiet
    assert fake.last_engine.engine_kwargs["mem_fraction_static"] == 0.7
    assert fake.last_engine.engine_kwargs["tp_size"] == 2
    # Default still applied:
    assert fake.last_engine.engine_kwargs["skip_tokenizer_init"] is True


def test_skip_tokenizer_init_cannot_be_overridden_by_caller():
    """A caller passing ``skip_tokenizer_init=False`` in ``engine_kwargs`` must be ignored.

    MEDS codes are already token ids. Turning tokenizer init back on would make SGLang try to
    load a tokenizer from the exported HF directory (which ``export_lightning_to_hf_dir``
    deliberately stubs, not populates), breaking the pipeline. The class docstring promises
    this invariant is non-overridable; the test locks in that the implementation enforces it.
    """
    fake = _FakeSGLModule()
    backend = SGLangBackend(
        "/tmp/x",
        engine_kwargs={"skip_tokenizer_init": False},
        sgl_module=fake,
    )
    del backend
    assert fake.last_engine.engine_kwargs["skip_tokenizer_init"] is True


# ---------------------------------------------------------------------------
# Padding / shape conversion
# ---------------------------------------------------------------------------


def test_strip_padding_to_lists_handles_left_and_right_pad():
    """Mixed-direction padding (rolling chunks can have both) must be fully stripped.

    The rolling loop doesn't promise purely-left or purely-right padding — a sample that
    finished in an earlier chunk has right-side pad from the finished-mask, on top of any
    left-side prompt pad. ``attention_mask`` is the ground truth for "which positions are
    real", so the strip operation must trust it.
    """
    input_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 0]], dtype=torch.long)
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]], dtype=torch.bool)
    assert _strip_padding_to_lists(input_ids, mask) == [[1, 2, 3], [4, 5, 6]]


def test_pad_right_to_tensor_satisfies_post_eos_invariant():
    """Right-pad naturally satisfies the per-row post-EOS invariant.

    The :class:`GenerationBackend` contract says rows that hit EOS before the chunk end must
    have ``pad_token_id`` in every position after EOS. SGLang stops each row at ``eos``, so
    the ragged output already has EOS as the last token per row; right-padding with pad_id
    ensures anything after that is pad. If row 1 emits 5 tokens and row 2 emits 3 tokens
    (possibly including EOS), the padded tensor has pad at [1, 3], [1, 4].
    """
    out = _pad_right_to_tensor(
        [[10, 11, 12, 13, 14], [20, 21, 22]],
        pad_value=0,
        device="cpu",
    )
    assert out.shape == (2, 5)
    assert out[1, 3].item() == 0
    assert out[1, 4].item() == 0


def test_generate_chunk_end_to_end_shape_and_padding():
    """Drive one ``generate_chunk`` through the fake engine and verify the output tensor.

    Asserts three things at once:
      1. Shape is ``[B, new_len]`` with ``new_len == max(lens)`` across the rows returned by
         the engine.
      2. Rows are right-padded with ``generation_config.pad_token_id``, not with zeros or
         whatever the engine happened to emit.
      3. The returned tensor's dtype matches ``input_ids.dtype`` — downstream rolling-loop
         code does ``sequence_so_far[:, start:end] = new_tokens`` and a dtype mismatch there
         would silently upcast or crash.
    """
    backend, fake = _make_backend()
    fake.last_engine.set_next_outputs(
        [
            {"output_ids": [5, 6, 7, 8]},  # 4 new tokens (hit max_new_tokens)
            {"output_ids": [9, 37]},  # 2 new tokens (hit eos at 37)
        ]
    )
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]], dtype=torch.long)
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    cfg = GenerationConfig(max_new_tokens=4, do_sample=False, pad_token_id=0, eos_token_id=37)

    out = backend.generate_chunk(input_ids, attention_mask=mask, generation_config=cfg)

    assert out.shape == (2, 4)
    assert out.dtype == input_ids.dtype
    # First row: full length, no padding needed.
    assert out[0].tolist() == [5, 6, 7, 8]
    # Second row: emitted EOS at position 1; positions 2..3 must be pad (0).
    assert out[1].tolist() == [9, 37, 0, 0]


def test_generate_chunk_accepts_legacy_token_ids_key():
    """Older SGLang versions returned ``token_ids`` (vs. modern ``output_ids``).

    The backend probes for ``output_ids`` first and falls back to ``token_ids`` so a version
    bump that flips field names doesn't silently break us with empty outputs. This locks
    that probe in.
    """
    backend, fake = _make_backend()
    fake.last_engine.set_next_outputs([{"token_ids": [1, 2, 3]}])
    input_ids = torch.tensor([[0, 4, 5]], dtype=torch.long)
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    cfg = GenerationConfig(max_new_tokens=3, do_sample=False, pad_token_id=0, eos_token_id=99)

    out = backend.generate_chunk(input_ids, attention_mask=mask, generation_config=cfg)

    assert out.tolist() == [[1, 2, 3]]


def test_generate_chunk_rejects_prompt_prefixed_engine_output():
    """A future SGLang version flipping to ``prompt + new`` semantics must fail loudly.

    The :class:`GenerationBackend` contract is new-only. SGLang's current release returns
    new-only tokens under ``output_ids``, but historical releases used ``token_ids`` with
    variable (prompt-inclusive vs. new-only) semantics. Silently accepting prompt-prefixed
    output would corrupt the rolling loop — the prompt tokens would get fed back as "newly
    generated" on the next chunk, duplicating the prompt in the accumulated sequence.

    The backend's defensive check: any row whose returned length exceeds ``max_new_tokens``
    can only happen if the engine included the prompt, so we raise a ``RuntimeError`` with a
    version-mismatch pointer. This test drives a fake engine that emits ``prompt + new`` and
    asserts the error fires.
    """
    backend, fake = _make_backend()
    # Simulate "prompt + new" semantics: prompt length was 3 (mask sums), max_new_tokens is 2,
    # so new-only output would be ≤2 tokens. Emit 5 tokens (3 prompt + 2 new) to trigger the
    # defensive check.
    fake.last_engine.set_next_outputs([{"output_ids": [4, 5, 6, 7, 8]}])
    input_ids = torch.tensor([[0, 4, 5, 6]], dtype=torch.long)
    mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.bool)
    cfg = GenerationConfig(max_new_tokens=2, do_sample=False, pad_token_id=0, eos_token_id=99)

    with pytest.raises(RuntimeError, match=r"prompt prefix plus new tokens"):
        backend.generate_chunk(input_ids, attention_mask=mask, generation_config=cfg)


def test_generate_chunk_raises_on_unknown_output_key():
    """Unknown output key must raise ``KeyError`` rather than silently producing empty rows.

    A future SGLang version using a different field name (e.g. ``new_token_ids``) should fail
    loudly so the issue is immediately obvious rather than manifesting as all-pad output tensors.
    """
    backend, fake = _make_backend()
    fake.last_engine.set_next_outputs([{"new_token_ids": [1, 2, 3]}])
    input_ids = torch.tensor([[4, 5]], dtype=torch.long)
    mask = torch.tensor([[1, 1]], dtype=torch.bool)
    cfg = GenerationConfig(max_new_tokens=3, do_sample=False, pad_token_id=0, eos_token_id=99)

    with pytest.raises(KeyError, match="new_token_ids"):
        backend.generate_chunk(input_ids, attention_mask=mask, generation_config=cfg)


# ---------------------------------------------------------------------------
# Kwarg stripping
# ---------------------------------------------------------------------------


def test_hf_only_kwargs_stripped_before_engine_call():
    """HF-only kwargs must not reach ``Engine.generate`` — forwarding them would TypeError inside the SGLang
    scheduler subprocess, surfacing as an opaque broken-pipe in the parent.

    This is the concrete honor-the-protocol ("implementations must only forward options supported by the
    active engine") check for SGLangBackend.
    """
    backend, fake = _make_backend()
    fake.last_engine.set_next_outputs([{"output_ids": [1]}])
    input_ids = torch.tensor([[0, 4]], dtype=torch.long)
    mask = torch.tensor([[0, 1]], dtype=torch.bool)
    cfg = GenerationConfig(max_new_tokens=1, do_sample=False, pad_token_id=0, eos_token_id=37)

    backend.generate_chunk(
        input_ids,
        attention_mask=mask,
        generation_config=cfg,
        logits_processor=["something"],
        stopping_criteria=["also something"],
        some_random_pass_through=True,
    )

    call = fake.last_engine.generate_calls[0]
    assert "logits_processor" not in call["extra_kwargs"]
    assert "stopping_criteria" not in call["extra_kwargs"]
    # Non-HF-specific kwargs still pass through — the engine can reject them itself if it
    # doesn't understand them, but the backend shouldn't decide for it.
    assert call["extra_kwargs"].get("some_random_pass_through") is True


# ---------------------------------------------------------------------------
# Sampling config translation
# ---------------------------------------------------------------------------


def test_do_sample_false_maps_to_temperature_zero():
    """HF's ``do_sample=False`` → SGLang's ``temperature=0.0`` (no separate greedy flag).

    If this mapping regressed to ``temperature=1.0``, greedy grammar tests would silently
    become stochastic and threshold-check flakily.
    """
    backend, fake = _make_backend()
    fake.last_engine.set_next_outputs([{"output_ids": [1]}])
    input_ids = torch.tensor([[4]], dtype=torch.long)
    mask = torch.tensor([[1]], dtype=torch.bool)
    cfg = GenerationConfig(max_new_tokens=1, do_sample=False, pad_token_id=0, eos_token_id=37)

    backend.generate_chunk(input_ids, attention_mask=mask, generation_config=cfg)

    sp = fake.last_engine.generate_calls[0]["sampling_params"]
    assert sp["temperature"] == 0.0
    assert sp["stop_token_ids"] == [37]
    assert sp["max_new_tokens"] == 1


def test_do_sample_true_uses_nonzero_temperature():
    """``do_sample=True`` must map to a non-zero temperature so SGLang actually samples."""
    backend, fake = _make_backend()
    fake.last_engine.set_next_outputs([{"output_ids": [1]}])
    input_ids = torch.tensor([[4]], dtype=torch.long)
    mask = torch.tensor([[1]], dtype=torch.bool)
    cfg = GenerationConfig(max_new_tokens=1, do_sample=True, pad_token_id=0, eos_token_id=37)

    backend.generate_chunk(input_ids, attention_mask=mask, generation_config=cfg)

    sp = fake.last_engine.generate_calls[0]["sampling_params"]
    assert sp["temperature"] > 0.0


def test_do_sample_true_honors_caller_temperature():
    """Caller-provided ``temperature`` on the GenerationConfig must reach SGLang under sampling.

    Regression guard against a prior revision where SGLangBackend hard-coded ``temperature=1.0``
    for the sampling branch, silently overriding the caller's choice. HF's behavior is that
    ``temperature`` is honored when ``do_sample=True`` and ignored when ``do_sample=False``;
    this test locks in the same semantics for SGLang.
    """
    backend, fake = _make_backend()
    fake.last_engine.set_next_outputs([{"output_ids": [1]}])
    input_ids = torch.tensor([[4]], dtype=torch.long)
    mask = torch.tensor([[1]], dtype=torch.bool)
    cfg = GenerationConfig(
        max_new_tokens=1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=0,
        eos_token_id=37,
    )

    backend.generate_chunk(input_ids, attention_mask=mask, generation_config=cfg)

    sp = fake.last_engine.generate_calls[0]["sampling_params"]
    assert sp["temperature"] == 0.7


def test_do_sample_false_ignores_caller_temperature():
    """``do_sample=False`` must always map to ``temperature=0.0`` regardless of caller config.

    Matches HF's behavior (temperature is meaningless under greedy). Prevents a subtle bug
    where a caller passing ``do_sample=False, temperature=2.0`` could inadvertently enable
    sampling inside SGLang by leaking the non-zero temperature through.
    """
    backend, fake = _make_backend()
    fake.last_engine.set_next_outputs([{"output_ids": [1]}])
    input_ids = torch.tensor([[4]], dtype=torch.long)
    mask = torch.tensor([[1]], dtype=torch.bool)
    cfg = GenerationConfig(
        max_new_tokens=1,
        do_sample=False,
        temperature=2.0,
        pad_token_id=0,
        eos_token_id=37,
    )

    backend.generate_chunk(input_ids, attention_mask=mask, generation_config=cfg)

    sp = fake.last_engine.generate_calls[0]["sampling_params"]
    assert sp["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_shutdown_is_idempotent():
    """``shutdown()`` called twice must not raise or double-invoke the engine's shutdown.

    The ``atexit`` hook can fire after an explicit shutdown, so idempotence is a real
    requirement, not a cosmetic one.
    """
    backend, fake = _make_backend()
    backend.shutdown()
    backend.shutdown()
    assert fake.last_engine.shutdown_calls == 1


def test_context_manager_protocol():
    """``with SGLangBackend(...) as b:`` should clean up on exit."""
    fake = _FakeSGLModule()
    with SGLangBackend("/tmp/x", sgl_module=fake) as backend:
        assert backend is not None
    assert fake.last_engine.shutdown_calls == 1


# ---------------------------------------------------------------------------
# Error surfaces
# ---------------------------------------------------------------------------


def test_missing_sglang_raises_clear_error(monkeypatch):
    """When the ``sglang`` extra is not installed, the import path must raise an actionable message.

    Without this, a user doing ``backend=sglang`` with the default install would see an
    uncaught ``ImportError: No module named 'sglang'`` and have to guess the fix. The
    wrapped message points them at the extra.
    """
    # Force the lazy import inside ``SGLangBackend.__init__`` to fail by making ``import
    # sglang`` raise. We can't just monkeypatch sys.modules because the import happens
    # inside the function body; instead we shadow the name.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sglang":
            raise ImportError("No module named 'sglang'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install MEDS_EIC_AR\\[sglang\\]"):
        SGLangBackend("/tmp/ignored")
