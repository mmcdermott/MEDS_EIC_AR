"""Tests for trajectory generation — both the single-chunk CLI path and the rolling-window path.

Two flavors of tests live here:

1. **CLI integration tests** (``test_generate_trajectories_runs``,
   ``test_generate_trajectories_rolling_runs``) drive the full ``MEICAR_generate_trajectories`` CLI
   via session-scoped fixtures and inspect the output parquet files.
2. **Direct unit tests on ``Model._rolling_generate``**
   (``test_rolling_generate_multi_round``, ``test_rolling_generate_smaller_context_yields_fewer_chunks``,
   ``test_rolling_generate_respects_budget``) exercise the rolling loop directly on a fresh
   random-init ``Model`` instance. These cover multi-round iteration and non-default
   ``rolling_context_size`` values — dimensions the CLI test cannot easily parameterize — by
   wrapping ``HF_model.generate`` with a ``MagicMock`` spy and asserting on the inner call count.

   We use a random-init model with ``eos_token_id = 37`` (the ``TIMELINE//END`` index in the demo
   vocab, a token a randomly-initialized model essentially never greedy-emits) so the rolling loop
   terminates on budget exhaustion rather than on EOS. That keeps the call-count math deterministic
   and independent of whatever greedy behavior the real pretrained demo model happens to learn on
   any given run.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import polars as pl
import pytest
import torch
from meds import held_out_split, train_split, tuning_split
from meds_torchdata import MEDSTorchBatch, MEDSTorchDataConfig
from polars.testing import assert_frame_equal

from MEDS_EIC_AR.model.model import Model


def _load_trajectories_by_split(root: Path) -> dict[str, dict[str, pl.DataFrame]]:
    trajectories_by_split: dict[str, dict[str, pl.DataFrame]] = {}
    for split in (train_split, tuning_split, held_out_split):
        split_dir = root / split
        trajectories_by_split[split] = {}
        for fp in split_dir.glob("*.parquet"):
            df = pl.read_parquet(fp, use_pyarrow=True)
            assert len(df) > 0, f"Parquet file {fp} is empty"
            trajectories_by_split[split][fp.stem] = df
        assert trajectories_by_split[split], (
            f"No generated parquet files found under {split_dir}. The generation CLI produced no "
            f"output for split {split!r}."
        )
    return trajectories_by_split


def test_generate_trajectories_runs(generated_trajectories: Path):
    trajectories_by_split = _load_trajectories_by_split(generated_trajectories)

    for sp, samps in trajectories_by_split.items():
        assert len(samps) == 2, f"Expected 2 trajectories for split {sp}, but found {len(samps)}."

        try:
            assert_frame_equal(samps["0"], samps["1"], check_exact=True)
            samps_equal = True
        except AssertionError:
            samps_equal = False

        assert not samps_equal, f"Trajectories for distinct samples in split {sp} are equal!"

        subjects = {samp: set(df["subject_id"]) for samp, df in samps.items()}
        assert subjects["0"] == subjects["1"], f"Subjects in samples for split {sp} do not match!"


def test_generate_trajectories_rolling_runs(
    generated_trajectories: Path, generated_trajectories_rolling: Path
):
    """End-to-end check that the sliding-window path actually produces multi-chunk output.

    The rolling fixture requests ``rolling_generation.max_new_tokens=50`` on the demo model, whose
    ``pretrained_max_seq_len=20``. That puts the rolling budget at ``> 2 * max_seq_len``, which forces the
    loop to cross several sliding boundaries (not just one), independent of per-subject input length. The
    single-chunk (non-rolling) fixture, by contrast, caps the new-token budget at
    ``max_seq_len - input_len <= 19`` — strictly less than 50. So any (split, sample, subject) whose rolling
    output has **strictly more** generated rows than its non-rolling counterpart is direct evidence that
    the rolling path emitted output the single-chunk path mathematically cannot. This is the actual
    integration signal: the shape-only assertions below (non-empty parquet files, subjects line up across
    samples) would pass even if the rolling kwarg were silently ignored and the CLI fell through to the
    legacy path. The ``max_rolling_rows > max_single_rows`` check is what proves the plumbing.
    """

    rolling_by_split = _load_trajectories_by_split(generated_trajectories_rolling)
    single_by_split = _load_trajectories_by_split(generated_trajectories)

    max_rolling_rows = 0
    max_single_rows = 0
    for sp, samps in rolling_by_split.items():
        assert len(samps) == 2, f"Expected 2 trajectories for split {sp}, but found {len(samps)}."
        subjects = {samp: set(df["subject_id"]) for samp, df in samps.items()}
        assert subjects["0"] == subjects["1"], f"Subjects in samples for split {sp} do not match!"

        for samp, df in samps.items():
            for subject_id, sub_df in df.group_by("subject_id"):
                max_rolling_rows = max(max_rolling_rows, len(sub_df))
                single_df = single_by_split[sp][samp]
                single_sub_df = single_df.filter(pl.col("subject_id") == subject_id[0])
                max_single_rows = max(max_single_rows, len(single_sub_df))

    # Non-rolling single-chunk path is strictly bounded above by ``pretrained_max_seq_len - 1 = 19`` new
    # tokens per subject (since the prompt consumes at least one slot). Rolling asked for 50, which the
    # single-chunk path cannot produce. If ``max_rolling_rows`` exceeds the non-rolling ceiling, the
    # rolling loop must have iterated across multiple chunk boundaries and written its output through.
    assert max_rolling_rows > max_single_rows, (
        f"Rolling-generation output is not longer than non-rolling output for any subject "
        f"(max rolling rows: {max_rolling_rows}, max non-rolling rows: {max_single_rows}). "
        f"Expected rolling to emit tokens beyond the single-chunk cap with max_new_tokens=50 / "
        f"max_seq_len=20 — suggests the rolling kwarg was silently dropped and the CLI fell through "
        f"to the legacy single-chunk path."
    )


# ---------------------------------------------------------------------------
# Direct unit tests on Model._rolling_generate
# ---------------------------------------------------------------------------


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
    # 3, the first chunk emits up to ``max_seq_len - 3 = 17`` tokens, then the window saturates and
    # every subsequent chunk emits ``max_seq_len - (max_seq_len - 1) = 1`` new token. So a 60-token
    # budget takes at least ``60 - 17 = 43`` post-saturation chunks plus a few pre-saturation ones —
    # well over any small threshold. ``>= 30`` is a safe lower bound that proves many rounds.
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


# ---------------------------------------------------------------------------
# Direct unit tests on RepeatedPredictionDataset + collate_with_meta (#89)
# ---------------------------------------------------------------------------


def test_collate_with_meta_round_trip_through_dataloader():
    """Run the expanded dataset through a real DataLoader and confirm the per-row metadata makes it onto the
    batched ``MEDSTorchBatch`` and unscrambles back to the right (subject, sample) indices.

    This is the wire-up test for the issue #89 path: ``RepeatedPredictionDataset`` →
    ``collate_with_meta`` → ``extract_meta`` is the chain ``predict_step`` and the regrouping
    code in ``MEICAR_generate_trajectories`` rely on, so it's worth testing end-to-end with a
    real dataloader rather than just calling the helpers in isolation.
    """
    from functools import partial

    from torch.utils.data import DataLoader

    from MEDS_EIC_AR.generation.repeated_dataset import (
        RepeatedPredictionDataset,
        collate_with_meta,
        extract_meta,
    )

    # Build a fake dataset whose items are MEDSTorchBatch-shaped (single-row) and whose ``code``
    # encodes the subject_idx so we can verify which underlying base item each row came from.
    class FakeBaseDataset:
        def __init__(self, n: int) -> None:
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, i: int) -> torch.Tensor:
            return torch.tensor([i, 100 + i, 200 + i], dtype=torch.long)

        def collate(self, items):
            # Return a Mock that quacks like MEDSTorchBatch enough for the metadata sidecar to
            # attach via object.__setattr__.
            codes = torch.stack(items, dim=0)
            batch = Mock(code=codes, PAD_INDEX=0, mode="SM")
            return batch

    base = FakeBaseDataset(n=3)
    n_samples = 4
    expanded = RepeatedPredictionDataset(base, n_samples=n_samples)
    loader = DataLoader(
        expanded,
        batch_size=5,  # deliberately not a multiple of n_samples so we hit a cross-subject batch
        shuffle=False,
        collate_fn=partial(collate_with_meta, base_collate=base.collate),
    )

    all_subject_idxs: list[int] = []
    all_sample_idxs: list[int] = []
    all_first_codes: list[int] = []
    for batch in loader:
        subject_idxs, sample_idxs = extract_meta(batch)
        # Every row in the batch should have a metadata entry of the right shape.
        assert subject_idxs.shape == (batch.code.shape[0],)
        assert sample_idxs.shape == (batch.code.shape[0],)
        all_subject_idxs.extend(subject_idxs.tolist())
        all_sample_idxs.extend(sample_idxs.tolist())
        all_first_codes.extend(batch.code[:, 0].tolist())

    # 3 subjects * 4 samples = 12 rows total, in subject-changes-slow order.
    expected_subject_idxs = [s for s in range(3) for _ in range(n_samples)]
    expected_sample_idxs = [k for _ in range(3) for k in range(n_samples)]
    # The fake dataset returns code[0] = subject_idx for whichever base item is being rendered,
    # so the per-row first code should equal the subject_idx that row carries.
    assert all_subject_idxs == expected_subject_idxs
    assert all_sample_idxs == expected_sample_idxs
    assert all_first_codes == expected_subject_idxs


def test_extract_meta_raises_on_unwrapped_batch():
    """``extract_meta`` should fail loudly when handed a batch that didn't come through the
    ``collate_with_meta`` path.

    This catches CLI wiring bugs where someone forgets to swap the collate function on the dataloader.
    """
    from types import SimpleNamespace

    from MEDS_EIC_AR.generation.repeated_dataset import extract_meta

    # SimpleNamespace, not Mock — Mock auto-creates attributes on access, which would falsely
    # pass the ``hasattr(batch, META_ATTR)`` check inside ``extract_meta``.
    bare_batch = SimpleNamespace(code=torch.zeros((2, 3), dtype=torch.long), PAD_INDEX=0, mode="SM")
    with pytest.raises(AttributeError, match="missing per-row metadata"):
        extract_meta(bare_batch)
