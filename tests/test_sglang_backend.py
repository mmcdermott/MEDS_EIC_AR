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

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from transformers import GenerationConfig

if TYPE_CHECKING:
    from pathlib import Path

from MEDS_EIC_AR.model.backends import GenerationBackend, SGLangBackend
from MEDS_EIC_AR.model.backends.sglang import (
    _FLASHINFER_MIN_HEAD_DIM,
    _SGLANG_CONTEXT_RESERVE,
    _pad_right_to_tensor,
    _strip_padding_to_lists,
)


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

    def generate(self, *, input_ids: list[list[int]], sampling_params: dict, **kw):
        # SGLang's real ``Engine.generate`` takes ``sampling_params: Union[Dict, List[Dict]]``;
        # the backend passes a dict so the fake mirrors that stable public contract rather than
        # instantiating a ``SamplingParams`` class (which is not exported at the top level of
        # the ``sglang`` package in v0.5.x — see the backend docstring for the full rationale).
        self.generate_calls.append(
            {
                "input_ids": [list(row) for row in input_ids],
                "sampling_params": dict(sampling_params),
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


def _make_backend() -> tuple[SGLangBackend, _FakeSGLModule]:
    """Helper: build a backend wired to a fresh fake sglang module."""
    fake = _FakeSGLModule()
    backend = SGLangBackend("/tmp/ignored_model_path", sgl_module=fake)
    return backend, fake


def _make_backend_with_context(tmp_path: Path, context_len: int) -> tuple[SGLangBackend, _FakeSGLModule]:
    """Build a backend over an HF directory whose ``config.json`` declares ``context_len`` positions.

    The plain :func:`_make_backend` points at a path with no ``config.json``, so the backend cannot
    determine a ceiling and leaves budgets alone — which is what keeps the older tests here
    unaffected by the context-window clamp. Tests that exercise the clamp need a readable config.
    """
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": context_len}))
    fake = _FakeSGLModule()
    backend = SGLangBackend(tmp_path, sgl_module=fake)
    return backend, fake


def _one_row_call(backend: SGLangBackend, fake: _FakeSGLModule, prompt_len: int, max_new_tokens: int):
    """Run one ``generate_chunk`` with a ``prompt_len``-token prompt; return the sampling params sent."""
    input_ids = torch.arange(1, prompt_len + 1, dtype=torch.long).unsqueeze(0)
    fake.last_engine.set_next_outputs([{"output_ids": [7]}])
    cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=0, eos_token_id=99)
    backend.generate_chunk(
        input_ids, attention_mask=torch.ones_like(input_ids, dtype=torch.bool), generation_config=cfg
    )
    return fake.last_engine.generate_calls[-1]["sampling_params"]


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


def test_engine_receives_allow_auto_truncate_by_default():
    """``allow_auto_truncate=True`` must be set on every Engine construction.

    The rolling loop in ``Model._rolling_generate`` sets ``chunk_budget = max_seq_len -
    prompt_len`` on each chunk, so ``input_len + max_new_tokens == max_context_length`` on
    boundary prompts. HF accepts this (positions are inclusive), SGLang rejects it by default
    (``input + max_new >= max_context`` is a hard failure in the tokenizer-manager validator).

    The backend now applies that ceiling itself, visibly, so this flag is a safety net rather
    than the mechanism: it catches a residual mismatch (a SGLang whose reserve differs from the
    one we subtract) as a shortfall instead of a crash mid-run. It is *not* parity with HF —
    the clamp costs real tokens, which is why the clamp is logged where it happens.
    """
    _, fake = _make_backend()
    assert fake.last_engine is not None
    assert fake.last_engine.engine_kwargs["allow_auto_truncate"] is True


def test_allow_auto_truncate_cannot_be_overridden_by_caller():
    """A caller passing ``allow_auto_truncate=False`` in ``engine_kwargs`` must be ignored.

    Same reasoning as ``skip_tokenizer_init``: with it off, any residual mismatch between the
    reserve this backend subtracts and the one the installed SGLang enforces turns from a
    shortfall into a hard failure partway through a run. The class docstring promises this
    invariant is non-overridable.
    """
    fake = _FakeSGLModule()
    backend = SGLangBackend(
        "/tmp/x",
        engine_kwargs={"allow_auto_truncate": False},
        sgl_module=fake,
    )
    del backend
    assert fake.last_engine.engine_kwargs["allow_auto_truncate"] is True


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


# ---------------------------------------------------------------------------
# Context-window ceiling
# ---------------------------------------------------------------------------


def test_budget_below_the_ceiling_is_passed_through_unchanged(tmp_path: Path):
    """A request that comfortably fits must reach the engine exactly as asked.

    Guards against a clamp that fires unconditionally and silently shortens every chunk, which would be a
    worse bug than the one being fixed.
    """
    backend, fake = _make_backend_with_context(tmp_path, context_len=512)
    sp = _one_row_call(backend, fake, prompt_len=128, max_new_tokens=256)
    assert sp["max_new_tokens"] == 256


def test_budget_at_the_ceiling_is_passed_through_unchanged(tmp_path: Path):
    """The largest request SGLang honors in full must not be clamped.

    ``context_len - prompt_len - reserve`` is the boundary; one token below it and at it are the
    two cases an off-by-one in the cap would get wrong.
    """
    backend, fake = _make_backend_with_context(tmp_path, context_len=512)
    ceiling = 512 - 128 - _SGLANG_CONTEXT_RESERVE
    assert _one_row_call(backend, fake, prompt_len=128, max_new_tokens=ceiling - 1)["max_new_tokens"] == (
        ceiling - 1
    )
    assert _one_row_call(backend, fake, prompt_len=128, max_new_tokens=ceiling)["max_new_tokens"] == ceiling


def test_budget_over_the_ceiling_is_clamped_and_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A window-saturating request must be cut to the ceiling *by us*, visibly.

    This is the reported defect: the shared rolling loop asks for ``max_seq_len - prompt_len``,
    i.e. the full window, and SGLang's ``allow_auto_truncate`` used to swallow the excess without
    a word. The clamp is now applied here and announced.
    """
    backend, fake = _make_backend_with_context(tmp_path, context_len=512)
    ceiling = 512 - 128 - _SGLANG_CONTEXT_RESERVE

    with caplog.at_level("WARNING"):
        sp = _one_row_call(backend, fake, prompt_len=128, max_new_tokens=384)

    assert sp["max_new_tokens"] == ceiling
    assert any("clamped from 384" in r.getMessage() for r in caplog.records), (
        f"Expected a warning naming the clamp; got {[r.getMessage() for r in caplog.records]}."
    )


def test_clamp_warning_is_logged_only_once_per_backend(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """The rolling loop calls ``generate_chunk`` once per chunk, all with the same saturating shape.

    Warning every time would bury the run's real output under hundreds of identical lines, so the message is
    emitted once per backend instance.
    """
    backend, fake = _make_backend_with_context(tmp_path, context_len=512)
    with caplog.at_level("WARNING"):
        for _ in range(5):
            _one_row_call(backend, fake, prompt_len=128, max_new_tokens=384)

    clamp_warnings = [r for r in caplog.records if "clamped from" in r.getMessage()]
    assert len(clamp_warnings) == 1, f"Expected exactly one clamp warning, got {len(clamp_warnings)}."


def test_prompt_leaving_no_room_raises_rather_than_returning_an_empty_chunk(tmp_path: Path):
    """A prompt at ``context_len - 1`` must raise, not return zero tokens.

    This is exactly what the rolling loop's default ``rolling_context_size = max_seq_len - 1``
    produces once the window saturates: a prompt one position short of the window, leaving room for
    one new token where SGLang needs the reserve to fit too. ``_rolling_generate`` advances by the
    number of tokens a chunk returns and loops until its budget is met, so an empty chunk would
    make it spin forever. The error names the knob to change.
    """
    backend, fake = _make_backend_with_context(tmp_path, context_len=512)
    with pytest.raises(ValueError, match="rolling_context_size"):
        _one_row_call(backend, fake, prompt_len=511, max_new_tokens=1)


def test_ceiling_uses_the_longest_prompt_in_the_batch(tmp_path: Path):
    """``max_new_tokens`` is shared across the batch, so the longest prompt is the binding one.

    Rolling chunks are ragged after post-EOS padding is stripped: rows that finished earlier come
    through shorter. Sizing the budget off a short row would let the longest row exceed the window.
    """
    backend, fake = _make_backend_with_context(tmp_path, context_len=64)
    # Row 0 is 8 real tokens, row 1 is 40 — left-padded to a common width, as the rolling loop emits.
    input_ids = torch.arange(1, 41, dtype=torch.long).repeat(2, 1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[0, :32] = False

    fake.last_engine.set_next_outputs([{"output_ids": [7]}, {"output_ids": [7]}])
    cfg = GenerationConfig(max_new_tokens=40, do_sample=False, pad_token_id=0, eos_token_id=99)
    backend.generate_chunk(input_ids, attention_mask=attention_mask, generation_config=cfg)

    sp = fake.last_engine.generate_calls[-1]["sampling_params"]
    assert sp["max_new_tokens"] == 64 - 40 - _SGLANG_CONTEXT_RESERVE


def test_unreadable_config_leaves_the_budget_alone(tmp_path: Path):
    """With no readable ``config.json`` the ceiling is unknown, so the request must pass through.

    Falling back to the engine's own ``allow_auto_truncate`` is worse than capping proactively, but
    much better than refusing to construct the backend or guessing a context length.
    """
    fake = _FakeSGLModule()
    backend = SGLangBackend(tmp_path, sgl_module=fake)  # empty dir: no config.json
    assert backend._context_len is None
    assert _one_row_call(backend, fake, prompt_len=128, max_new_tokens=100_000)["max_new_tokens"] == 100_000


# ---------------------------------------------------------------------------
# Attention-backend / head-dim pre-flight
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, **fields: Any) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(fields))
    return tmp_path


def test_small_head_dim_on_the_default_attention_backend_is_refused(tmp_path: Path):
    """Constructing over a narrow model without naming an attention backend must raise.

    SGLang would otherwise pick FlashInfer, which aborts the scheduler subprocess on head dims below its
    floor. That abort reaches the parent as SIGQUIT / exit -9, so a shape constraint presents as an out-of-
    memory kill — the error here exists to keep anyone from chasing memory.
    """
    _write_config(tmp_path, max_position_embeddings=512, head_dim=32)
    with pytest.raises(ValueError, match="FlashInfer"):
        SGLangBackend(tmp_path, sgl_module=_FakeSGLModule())


def test_small_head_dim_is_allowed_when_an_attention_backend_is_named(tmp_path: Path):
    """Naming ``attention_backend`` explicitly is a deliberate choice and must be honored.

    This is both the documented remedy (``sglang_demo.yaml`` sets ``triton`` for exactly this
    reason) and the reason the check can't wrongly block a future FlashInfer that lifts the floor.
    """
    _write_config(tmp_path, max_position_embeddings=512, head_dim=32)
    fake = _FakeSGLModule()
    SGLangBackend(tmp_path, engine_kwargs={"attention_backend": "triton"}, sgl_module=fake)
    assert fake.last_engine.engine_kwargs["attention_backend"] == "triton"


def test_head_dim_at_the_floor_is_allowed_on_the_default_attention_backend(tmp_path: Path):
    """The floor is inclusive — a model exactly at it must construct without complaint."""
    _write_config(tmp_path, max_position_embeddings=512, head_dim=_FLASHINFER_MIN_HEAD_DIM)
    SGLangBackend(tmp_path, sgl_module=_FakeSGLModule())


def test_unknown_head_dim_does_not_block_construction(tmp_path: Path):
    """A config without ``head_dim`` leaves nothing to check, and must not be treated as a failure.

    Same principle as the unreadable-config case for the context ceiling: these are pre-flight
    diagnostics, and none of them is worth refusing to construct the backend over.
    """
    _write_config(tmp_path, max_position_embeddings=512)
    SGLangBackend(tmp_path, sgl_module=_FakeSGLModule())
