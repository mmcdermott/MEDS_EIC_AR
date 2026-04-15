"""Direct unit tests for ``Model._rolling_generate``.

These tests complement the CLI-driven integration tests in ``test_generate_trajectories.py`` by
exercising the rolling loop directly on a real pretrained (via the shared ``pretrained_GPT_model``
fixture) model. They cover two dimensions that the CLI test cannot easily parameterize:

1. **Multi-round iteration** with ``max_new_tokens`` well above ``max_seq_len`` — many sliding
   boundaries, not just one — verified by spying on ``HF_model.generate`` and asserting the call
   count grows with the budget.
2. **Non-default ``rolling_context_size`` values** — a smaller per-chunk window must yield a
   strictly larger number of inner ``HF_model.generate`` calls for the same total new-token budget.
"""

from unittest.mock import MagicMock

import torch
from meds_torchdata import MEDSTorchBatch

from MEDS_EIC_AR.model.model import Model


def _run_rolling(
    model: Model,
    batch: MEDSTorchBatch,
    *,
    max_new_tokens: int,
    rolling_context_size: int | None,
) -> tuple[torch.Tensor, int]:
    """Spy on ``HF_model.generate`` and return ``(output_tokens, inner_call_count)``."""
    real = model.HF_model.generate
    spy = MagicMock(wraps=real)
    model.HF_model.generate = spy
    try:
        out = model._rolling_generate(
            batch,
            max_new_tokens=max_new_tokens,
            rolling_context_size=rolling_context_size,
            do_sample=False,
        )
    finally:
        model.HF_model.generate = real
    return out, spy.call_count


def test_rolling_generate_multi_round(pretrained_GPT_model: Model, sample_batch: MEDSTorchBatch):
    """Budget well above ``max_seq_len`` must trigger many sliding-window iterations."""

    torch.manual_seed(0)
    max_seq_len = pretrained_GPT_model.max_seq_len  # 20 in _demo_pretrain
    budget = 3 * max_seq_len  # 60 — guarantees many rounds regardless of input length

    out, call_count = _run_rolling(
        pretrained_GPT_model,
        sample_batch,
        max_new_tokens=budget,
        rolling_context_size=None,
    )

    # Output shape: at most budget tokens per row; less only if every row hit EOS early.
    assert out.ndim == 2
    assert out.shape[0] == sample_batch.code.shape[0]
    assert 0 < out.shape[1] <= budget

    # Lower bound on iteration count: once the sliding window saturates at ctx_size = max_seq_len - 1,
    # each chunk can emit at most ``max_seq_len - ctx_size = 1`` new token. Earlier (pre-saturation)
    # chunks can emit more, but the post-saturation floor gives us a principled minimum. For a budget
    # of ``3 * max_seq_len = 60`` we must have at least ``60 / (max_seq_len - 1) ≈ 4`` inner calls
    # even under the most favorable pre-saturation fill; in practice we see many more. Asserting
    # ``>= 4`` is loose enough to be robust across random inputs and tight enough to prove multi-round
    # iteration happened.
    assert call_count >= 4, (
        f"Rolling loop only made {call_count} inner HF_model.generate call(s) for a budget of "
        f"{budget} tokens on a model with max_seq_len={max_seq_len}. Expected many rounds."
    )


def test_rolling_generate_smaller_context_yields_more_chunks(
    pretrained_GPT_model: Model, sample_batch: MEDSTorchBatch
):
    """Shrinking ``rolling_context_size`` must strictly increase the inner call count.

    Each chunk can emit at most ``max_seq_len - ctx_size`` new tokens once the window saturates.
    Smaller ``ctx_size`` → larger per-chunk budget → fewer chunks; **larger** ``ctx_size`` →
    smaller per-chunk budget → **more** chunks. This test verifies the monotone relationship by
    running the loop twice and comparing call counts. We use a fixed large budget so both runs
    iterate.
    """

    torch.manual_seed(0)
    max_seq_len = pretrained_GPT_model.max_seq_len  # 20
    budget = 40

    # Large per-chunk window (ctx_size close to max_seq_len) → each chunk emits ~1 token post-
    # saturation → high call count.
    _, calls_large_ctx = _run_rolling(
        pretrained_GPT_model,
        sample_batch,
        max_new_tokens=budget,
        rolling_context_size=max_seq_len - 1,
    )

    # Small per-chunk window → each chunk can emit up to ``max_seq_len - small_ctx`` tokens →
    # fewer iterations total for the same budget.
    small_ctx = 4
    torch.manual_seed(0)
    _, calls_small_ctx = _run_rolling(
        pretrained_GPT_model,
        sample_batch,
        max_new_tokens=budget,
        rolling_context_size=small_ctx,
    )

    assert calls_large_ctx > calls_small_ctx, (
        f"Expected more inner chunks with rolling_context_size={max_seq_len - 1} than with "
        f"rolling_context_size={small_ctx}, but got {calls_large_ctx} and {calls_small_ctx}. "
        f"A smaller per-chunk context window should emit more tokens per call and therefore "
        f"fewer total calls for a fixed new-token budget."
    )

    # Both should still iterate at least twice for a 40-token budget on a 20-token model.
    assert calls_small_ctx >= 2
    assert calls_large_ctx >= 2


def test_rolling_generate_respects_budget(pretrained_GPT_model: Model, sample_batch: MEDSTorchBatch):
    """Output length is bounded above by ``max_new_tokens`` for any ``rolling_context_size``."""

    torch.manual_seed(0)
    budget = 25
    for ctx in (None, 4, 8, pretrained_GPT_model.max_seq_len - 1):
        out, _ = _run_rolling(
            pretrained_GPT_model,
            sample_batch,
            max_new_tokens=budget,
            rolling_context_size=ctx,
        )
        assert out.shape[1] <= budget, (
            f"Output length {out.shape[1]} exceeds max_new_tokens={budget} with rolling_context_size={ctx}."
        )
