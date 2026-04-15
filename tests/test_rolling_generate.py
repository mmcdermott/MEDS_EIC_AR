"""Direct unit tests for ``Model._rolling_generate``.

These tests complement the CLI-driven integration tests in ``test_generate_trajectories.py`` by
exercising the rolling loop directly on a randomly-initialized ``Model`` instance. They cover two
dimensions that the CLI test cannot easily parameterize:

1. **Multi-round iteration** with ``max_new_tokens`` well above ``max_seq_len`` — many sliding
   boundaries, not just one — verified by spying on ``HF_model.generate`` and asserting the call
   count grows with the budget.
2. **Non-default ``rolling_context_size`` values** — a smaller per-chunk window increases the
   per-chunk new-token budget (each chunk can emit up to ``max_seq_len - rolling_context_size``
   tokens), so it yields a strictly **smaller** number of inner ``HF_model.generate`` calls for
   the same total new-token budget. The test below verifies this monotone relationship.

We use a fresh random-init model per test and pick an ``eos_token_id`` (``37`` = ``TIMELINE//END``
in the demo vocab) that a randomly-initialized model will essentially never greedily emit, so the
rolling loop terminates on budget exhaustion rather than on EOS. This makes the call-count math
deterministic and independent of whatever the session-scoped ``pretrained_GPT_model`` fixture's
greedy behavior happens to be on any given run — the earlier version of this test was sensitive to
greedy-decoding EOS drift from the real pretrained model.
"""

from unittest.mock import MagicMock, Mock

import pytest
import torch
from meds_torchdata import MEDSTorchBatch, MEDSTorchDataConfig

from MEDS_EIC_AR.model.model import Model


@pytest.fixture
def rolling_model(dataset_config: MEDSTorchDataConfig) -> Model:
    """A small random-init ``Model`` with ``max_seq_len=20`` and a safe ``eos_token_id``."""
    torch.manual_seed(0)
    model = Model(
        {
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_size": 4,
            "max_position_embeddings": 20,
            "vocab_size": dataset_config.vocab_size,
        },
        precision="32-true",
    )
    model.HF_model.config.eos_token_id = 37  # TIMELINE//END; random-init model ~never greedy-emits 37
    return model


@pytest.fixture
def rolling_batch() -> MEDSTorchBatch:
    """A tiny fake batch whose codes avoid the ``eos_token_id=37`` so the rolling loop won't stop."""
    return Mock(
        code=torch.LongTensor([[38, 22, 36], [38, 22, 36]]),
        PAD_INDEX=0,
        mode="SM",
    )


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


def test_rolling_generate_multi_round(rolling_model: Model, rolling_batch: MEDSTorchBatch):
    """Budget well above ``max_seq_len`` must trigger many sliding-window iterations."""

    max_seq_len = rolling_model.max_seq_len  # 20
    budget = 3 * max_seq_len  # 60 — guarantees many rounds

    out, call_count = _run_rolling(
        rolling_model,
        rolling_batch,
        max_new_tokens=budget,
        rolling_context_size=None,
    )

    # Output shape: exactly budget tokens per row, since eos=37 is never emitted by this random-init
    # model and the full budget is available.
    assert out.shape == (rolling_batch.code.shape[0], budget), (
        f"Expected output shape ({rolling_batch.code.shape[0]}, {budget}), got {tuple(out.shape)}."
    )

    # With the default ``rolling_context_size = max_seq_len - 1 = 19`` and an input prompt of length
    # 3, the first chunk emits up to ``max_seq_len - 3 = 17`` tokens (well, this reaches steady state
    # faster because the window saturates). Once saturated, each subsequent chunk emits exactly
    # ``max_seq_len - (max_seq_len - 1) = 1`` new token. So a 60-token budget takes at least
    # ``60 - 17 = 43`` post-saturation chunks, plus some pre-saturation chunks — well over any small
    # threshold. We use ``>= 30`` as a safe lower bound that proves rolling iterated many times.
    assert call_count >= 30, (
        f"Rolling loop only made {call_count} inner HF_model.generate call(s) for a budget of "
        f"{budget} tokens on a model with max_seq_len={max_seq_len} — expected many rounds."
    )


def test_rolling_generate_smaller_context_yields_fewer_chunks(
    rolling_model: Model, rolling_batch: MEDSTorchBatch
):
    """Shrinking ``rolling_context_size`` must strictly decrease the inner call count.

    Each chunk can emit at most ``max_seq_len - ctx_size`` new tokens once the window saturates.
    Smaller ``ctx_size`` → larger per-chunk budget → **fewer** chunks. Larger ``ctx_size`` →
    smaller per-chunk budget → **more** chunks. This test verifies the monotone relationship by
    running the loop twice on the same model/batch and comparing call counts.
    """

    max_seq_len = rolling_model.max_seq_len  # 20
    budget = 40  # > 2 * max_seq_len, so both settings must iterate several times

    # Large per-chunk window → each chunk emits 1 new token post-saturation → many calls.
    _, calls_large_ctx = _run_rolling(
        rolling_model,
        rolling_batch,
        max_new_tokens=budget,
        rolling_context_size=max_seq_len - 1,  # 19
    )

    # Small per-chunk window → each chunk emits up to ``max_seq_len - 4 = 16`` tokens → few calls.
    small_ctx = 4
    _, calls_small_ctx = _run_rolling(
        rolling_model,
        rolling_batch,
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


def test_rolling_generate_respects_budget(rolling_model: Model, rolling_batch: MEDSTorchBatch):
    """Output length is bounded above by ``max_new_tokens`` for any ``rolling_context_size``."""

    budget = 25
    for ctx in (None, 4, 8, rolling_model.max_seq_len - 1):
        out, _ = _run_rolling(
            rolling_model,
            rolling_batch,
            max_new_tokens=budget,
            rolling_context_size=ctx,
        )
        assert out.shape[1] <= budget, (
            f"Output length {out.shape[1]} exceeds max_new_tokens={budget} with rolling_context_size={ctx}."
        )
