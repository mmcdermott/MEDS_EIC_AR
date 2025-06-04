from pathlib import Path

import polars as pl
from meds import held_out_split, train_split, tuning_split
from polars.testing import assert_frame_equal


def test_generate_trajectories_runs(generated_trajectories: Path):
    generated_files = generated_trajectories.glob("*/*.parquet")
    assert len(list(generated_files)) > 0, "No generated files found in the specified directory."

    trajectories_by_split = {}
    for split in (train_split, tuning_split, held_out_split):
        split_dir = generated_trajectories / split

        trajectories_by_split[split] = {}

        for fp in split_dir.glob("*.parquet"):
            df = pl.read_parquet(fp, use_pyarrow=True)
            assert len(df) > 0, f"Parquet file {fp} is empty"
            trajectories_by_split[split][fp.stem] = df

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
