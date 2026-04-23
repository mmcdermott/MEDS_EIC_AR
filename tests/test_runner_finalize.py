"""Unit tests for ``MEDS_EIC_AR.generation.runner.finalize_predictions`` — the rank-0 step.

``finalize_predictions`` is the critical correctness step for DDP generation: it reads all
rank shards per trajectory, sorts by ``dataset_row_idx``, validates the index set is the full
permutation ``{0, ..., n_dataset_rows - 1}``, and hands off to ``format_trajectories``. The
full end-to-end single-device path is covered by ``test_generate_trajectories.py``; the tests
in this module specifically target the *multi-shard* merge + validation logic (the DDP case,
which no other test exercises because the CI host has no GPUs).

``format_trajectories`` and ``GeneratedTrajectorySchema.align`` are patched out so we can
focus on the merge + validation — those are covered elsewhere and pulling in a real
``MEDSPytorchDataset`` here would add a lot of setup for no additional signal.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch

from MEDS_EIC_AR.generation.runner import (
    finalize_predictions,
    write_predictions_shards,
)


def _captured_format_trajectories():
    """Return a fake ``format_trajectories`` that records the per-row token batches it receives.

    The real ``format_trajectories`` consumes a list of ``[1, L_i]`` tensors (the 1-row
    batches ``finalize_predictions`` emits after its merge + sort) and produces a polars
    DataFrame. We capture the tensors and return a trivial DataFrame so we can assert the
    merge ordering without spinning up a real base dataset.
    """
    captured: list[torch.Tensor] = []

    def fake(base_dataset, one_row_batches):
        captured.extend(one_row_batches)
        return pl.DataFrame({"placeholder": list(range(len(one_row_batches)))})

    return captured, fake


def _patch_finalize_io():
    """Patch ``format_trajectories`` + ``GeneratedTrajectorySchema.align`` as identity passthroughs.

    Returns the context-manager plus the captured-batches list so tests can assert what
    ``finalize_predictions`` fed into the formatter in what order.
    """
    captured, fake_format = _captured_format_trajectories()
    return captured, patch.multiple(
        "MEDS_EIC_AR.generation.runner",
        format_trajectories=fake_format,
        GeneratedTrajectorySchema=MagicMock(align=lambda t: t),
    )


def test_two_ranks_interleaved_shards_merge_in_dataset_row_order(tmp_path: Path):
    """Two ranks writing interleaved shards merge into per-row batches in ``dataset_row_idx`` order,
    regardless of which rank wrote which rows."""
    shard_dir = tmp_path / "_shards"
    shard_dir.mkdir()

    # Rank 0 takes rows [0, 2] (even DistributedSampler stripe for world_size=2)
    write_predictions_shards(
        [
            {
                "tokens": torch.tensor([[10, 11], [30, 31]], dtype=torch.long),
                "dataset_row_idxs": torch.tensor([0, 2]),
                "trajectory_idxs": torch.tensor([0, 0]),
            }
        ],
        n_trajectories=1,
        shard_dir=shard_dir,
        rank=0,
    )
    # Rank 1 takes rows [1, 3] (odd stripe)
    write_predictions_shards(
        [
            {
                "tokens": torch.tensor([[20, 21], [40, 41]], dtype=torch.long),
                "dataset_row_idxs": torch.tensor([1, 3]),
                "trajectory_idxs": torch.tensor([0, 0]),
            }
        ],
        n_trajectories=1,
        shard_dir=shard_dir,
        rank=1,
    )

    captured, patcher = _patch_finalize_io()
    out_fp = tmp_path / "trajectory_0.parquet"
    with patcher:
        finalize_predictions(
            n_dataset_rows=4,
            shard_dir=shard_dir,
            trajectory_paths={0: out_fp},
            base_dataset=MagicMock(),
            do_overwrite=True,
            cleanup_shards=False,
        )

    # Four rows, in dataset_row_idx order [0, 1, 2, 3] regardless of which rank wrote them:
    assert len(captured) == 4
    assert [t.flatten().tolist() for t in captured] == [
        [10, 11],  # row 0 from rank 0
        [20, 21],  # row 1 from rank 1
        [30, 31],  # row 2 from rank 0
        [40, 41],  # row 3 from rank 1
    ]
    assert out_fp.is_file()


def test_variable_length_across_shards_preserved_after_merge(tmp_path: Path):
    """Rows with different ``L`` from different ranks keep their native lengths after merge — no cross-row
    padding.

    The 1-row-batch emit shape is the mechanism.
    """
    shard_dir = tmp_path / "_shards"
    shard_dir.mkdir()

    write_predictions_shards(
        [
            {
                "tokens": torch.tensor([[1, 2]], dtype=torch.long),
                "dataset_row_idxs": torch.tensor([0]),
                "trajectory_idxs": torch.tensor([0]),
            }
        ],
        n_trajectories=1,
        shard_dir=shard_dir,
        rank=0,
    )
    write_predictions_shards(
        [
            {
                "tokens": torch.tensor([[3, 4, 5, 6]], dtype=torch.long),
                "dataset_row_idxs": torch.tensor([1]),
                "trajectory_idxs": torch.tensor([0]),
            }
        ],
        n_trajectories=1,
        shard_dir=shard_dir,
        rank=1,
    )

    captured, patcher = _patch_finalize_io()
    with patcher:
        finalize_predictions(
            n_dataset_rows=2,
            shard_dir=shard_dir,
            trajectory_paths={0: tmp_path / "trajectory_0.parquet"},
            base_dataset=MagicMock(),
            do_overwrite=True,
            cleanup_shards=False,
        )

    assert [t.flatten().tolist() for t in captured] == [[1, 2], [3, 4, 5, 6]]


def test_missing_row_fails_set_completeness(tmp_path: Path):
    """Shards together cover only 3 of 4 expected rows → RuntimeError."""
    shard_dir = tmp_path / "_shards"
    shard_dir.mkdir()
    write_predictions_shards(
        [
            {
                "tokens": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
                "dataset_row_idxs": torch.tensor([0, 1, 2]),  # row 3 missing
                "trajectory_idxs": torch.tensor([0, 0, 0]),
            }
        ],
        n_trajectories=1,
        shard_dir=shard_dir,
        rank=0,
    )
    _, patcher = _patch_finalize_io()
    with patcher, pytest.raises(RuntimeError, match="merged row count 3 != n_dataset_rows 4"):
        finalize_predictions(
            n_dataset_rows=4,
            shard_dir=shard_dir,
            trajectory_paths={0: tmp_path / "trajectory_0.parquet"},
            base_dataset=MagicMock(),
            do_overwrite=True,
            cleanup_shards=False,
        )


def test_duplicate_row_across_ranks_fails_set_completeness(tmp_path: Path):
    """Two ranks both claim ``dataset_row_idx = 0`` — length passes but set-equality fails."""
    shard_dir = tmp_path / "_shards"
    shard_dir.mkdir()
    for rank in (0, 1):
        write_predictions_shards(
            [
                {
                    "tokens": torch.tensor([[1, 2]], dtype=torch.long),
                    "dataset_row_idxs": torch.tensor([0]),
                    "trajectory_idxs": torch.tensor([0]),
                }
            ],
            n_trajectories=1,
            shard_dir=shard_dir,
            rank=rank,
        )
    _, patcher = _patch_finalize_io()
    with patcher, pytest.raises(RuntimeError, match="dataset-row-index set is not"):
        finalize_predictions(
            n_dataset_rows=2,
            shard_dir=shard_dir,
            trajectory_paths={0: tmp_path / "trajectory_0.parquet"},
            base_dataset=MagicMock(),
            do_overwrite=True,
            cleanup_shards=False,
        )


def test_empty_split_skips_without_crash(tmp_path: Path):
    """``n_dataset_rows == 0`` short-circuits with no write and no IndexError on ``idxs[0]``."""
    shard_dir = tmp_path / "_shards"
    shard_dir.mkdir()
    out_fp = tmp_path / "trajectory_0.parquet"
    _, patcher = _patch_finalize_io()
    with patcher:
        # Does not raise despite no shards in shard_dir and no rank writes — the empty-split
        # short-circuit fires before the glob + validation.
        finalize_predictions(
            n_dataset_rows=0,
            shard_dir=shard_dir,
            trajectory_paths={0: out_fp},
            base_dataset=MagicMock(),
            do_overwrite=True,
            cleanup_shards=False,
        )
    assert not out_fp.exists()


def test_no_rank_shards_fails_with_filesystem_hint(tmp_path: Path):
    """If ``shard_dir`` is empty (no rank wrote) the error points at the multi-node-DDP shared-filesystem mis-
    configuration as a possible cause."""
    shard_dir = tmp_path / "_shards"
    shard_dir.mkdir()
    with pytest.raises(RuntimeError, match="shared filesystem"):
        finalize_predictions(
            n_dataset_rows=4,
            shard_dir=shard_dir,
            trajectory_paths={0: tmp_path / "trajectory_0.parquet"},
            base_dataset=MagicMock(),
            do_overwrite=True,
            cleanup_shards=False,
        )
