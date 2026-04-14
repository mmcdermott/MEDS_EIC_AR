from pathlib import Path

import polars as pl
from meds import held_out_split, train_split, tuning_split
from polars.testing import assert_frame_equal


def _load_trajectories_by_split(root: Path) -> dict[str, dict[str, pl.DataFrame]]:
    generated_files = list(root.glob("*/*.parquet"))
    assert len(generated_files) > 0, f"No generated files found under {root}."

    trajectories_by_split: dict[str, dict[str, pl.DataFrame]] = {}
    for split in (train_split, tuning_split, held_out_split):
        split_dir = root / split
        trajectories_by_split[split] = {}
        for fp in split_dir.glob("*.parquet"):
            df = pl.read_parquet(fp, use_pyarrow=True)
            assert len(df) > 0, f"Parquet file {fp} is empty"
            trajectories_by_split[split][fp.stem] = df
    return trajectories_by_split


def test_generate_trajectories_runs(generated_trajectories: Path):
    trajectories_by_split = _load_trajectories_by_split(generated_trajectories)

    assert len(trajectories_by_split) == 3, "Not all splits have generated trajectories."

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


def test_generate_trajectories_rolling_runs(generated_trajectories_rolling: Path):
    """End-to-end check that the sliding-window path wires through the CLI correctly.

    The fixture requests ``rolling_generation.max_new_tokens=15`` on the demo model, which has
    ``max_position_embeddings`` small enough that a single HF generate call can't emit 15 new tokens in one
    chunk — so the loop must cross at least one chunk boundary. We don't assert exact trajectory length
    (model output depends on whether EOS was sampled) — only that the pipeline produces non-empty parquet
    files for each split/sample, matching the shape of the legacy test.
    """

    trajectories_by_split = _load_trajectories_by_split(generated_trajectories_rolling)
    assert len(trajectories_by_split) == 3, "Not all splits have rolling-generated trajectories."

    for sp, samps in trajectories_by_split.items():
        assert len(samps) == 2, f"Expected 2 trajectories for split {sp}, but found {len(samps)}."
        subjects = {samp: set(df["subject_id"]) for samp, df in samps.items()}
        assert subjects["0"] == subjects["1"], f"Subjects in samples for split {sp} do not match!"
