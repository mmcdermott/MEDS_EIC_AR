from pathlib import Path

import polars as pl
from meds import held_out_split, train_split, tuning_split
from polars.testing import assert_frame_equal


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
