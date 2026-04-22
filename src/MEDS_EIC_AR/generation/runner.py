"""Rank-aware shard-then-merge pipeline for finalizing generation output.

The generation CLI calls two functions from this module, one before and one after a barrier:

- ``write_predictions_shards`` — runs on *every* rank (single-device or DDP). Takes that rank's
  ``trainer.predict`` output and writes one parquet shard per trajectory under a shared
  ``shard_dir`` (name: ``trajectory_{t}.rank_{rank}.parquet``). Shards carry the raw per-row
  generated tokens alongside the base-dataset row index.
- ``finalize_predictions`` — runs on **rank 0 only**, after an inter-rank barrier. Reads all
  rank shards for each trajectory, concatenates + sorts by ``dataset_row_idx``, validates the
  index set is the complete permutation ``{0, ..., n_dataset_rows - 1}``, feeds the merged
  per-row tokens to ``format_trajectories``, and writes the final per-trajectory parquet via
  ``GeneratedTrajectorySchema``. Shards are deleted after the final parquets land.

The filesystem is the coordination mechanism: no ``BasePredictionWriter`` callback, no
``torch.distributed.all_gather``. This works for any ``world_size`` including 1 (rank 0 writes
one shard, rank 0 merges one shard — same code path). The shard format (``dataset_row_idx``
column + ``tokens`` list column) makes the merge a single ``pl.concat(...).sort(...)`` call.
"""

import contextlib
import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import torch
from MEDS_trajectory_evaluation.schema import GeneratedTrajectorySchema

from .format_trajectories import format_trajectories

logger = logging.getLogger(__name__)

#: Shard column names. Written by ``write_predictions_shards``, consumed by
#: ``finalize_predictions``. Kept as module constants so the two stay in sync.
SHARD_ROW_IDX_COL = "dataset_row_idx"
SHARD_TOKENS_COL = "tokens"


def _demux_predictions_per_trajectory(
    predictions: list[dict[str, torch.Tensor]],
    n_trajectories: int,
) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]]]:
    """Demux a flat list of per-batch predict outputs into per-trajectory token + dataset-row-idx lists.

    Each batch in ``predictions`` carries three tensors: ``tokens`` (``[B, L]``), ``dataset_row_idxs``
    (``[B]``), and ``trajectory_idxs`` (``[B]``, values in ``0..n_trajectories-1``). We demux by
    trajectory so that downstream formatting can process each trajectory's rows as one stream. The
    per-batch grouping is preserved (rather than flattened) because each batch has its own ``L``
    from the generator — ``model.generate`` stops at the batch-local stopping condition (all rows
    EOS or ``max_new_tokens``), which varies batch-to-batch.

    Iterates over ``trajectory_idxs.unique()`` per batch rather than doing ``n_trajectories``
    boolean compares unconditionally, so tail batches with fewer than ``n_trajectories`` rows
    don't pay for empty buckets.

    Examples:
        >>> preds = [
        ...     {"tokens": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        ...      "dataset_row_idxs": torch.tensor([0, 0, 1, 1]),
        ...      "trajectory_idxs": torch.tensor([0, 1, 0, 1])},
        ... ]
        >>> tok, sidx = _demux_predictions_per_trajectory(preds, n_trajectories=2)
        >>> [b.tolist() for b in tok[0]]
        [[[1, 2], [5, 6]]]
        >>> [b.tolist() for b in tok[1]]
        [[[3, 4], [7, 8]]]
        >>> [b.tolist() for b in sidx[0]]
        [[0, 1]]
        >>> [b.tolist() for b in sidx[1]]
        [[0, 1]]

        A trajectory absent from a batch simply has no entry appended for that batch:

        >>> preds = [
        ...     {"tokens": torch.tensor([[1, 2], [3, 4]]),
        ...      "dataset_row_idxs": torch.tensor([0, 1]),
        ...      "trajectory_idxs": torch.tensor([0, 0])},
        ... ]
        >>> _, sidx = _demux_predictions_per_trajectory(preds, n_trajectories=2)
        >>> [b.tolist() for b in sidx[0]]
        [[0, 1]]
        >>> [b.tolist() for b in sidx[1]]
        []
    """
    per_trajectory_batches: dict[int, list[torch.Tensor]] = {t: [] for t in range(n_trajectories)}
    per_trajectory_dataset_row_idxs: dict[int, list[torch.Tensor]] = {t: [] for t in range(n_trajectories)}
    for pred in predictions:
        tokens = pred["tokens"]
        trajectory_idxs = pred["trajectory_idxs"]
        dataset_row_idxs = pred["dataset_row_idxs"]
        for t in trajectory_idxs.unique().tolist():
            mask = trajectory_idxs == t
            per_trajectory_batches[t].append(tokens[mask])
            per_trajectory_dataset_row_idxs[t].append(dataset_row_idxs[mask])
    return per_trajectory_batches, per_trajectory_dataset_row_idxs


def write_predictions_shards(
    predictions: list[dict[str, torch.Tensor]],
    *,
    n_trajectories: int,
    shard_dir: Path,
    rank: int,
) -> None:
    """Write this rank's predictions as per-trajectory parquet shards under ``shard_dir``.

    Called by every rank (single-device = rank 0 with ``world_size=1``, or each rank in DDP).
    Writes one shard per trajectory with filename ``trajectory_{t}.rank_{rank}.parquet``, carrying
    columns ``dataset_row_idx`` (Int64) and ``tokens`` (``List[Int64]``). Under DDP each rank
    sees only its local slice of predictions — the shards for a single trajectory together span
    ``{0, ..., n_dataset_rows - 1}`` across all rank files. Rows within a rank's shard are NOT
    sorted; the final sort happens in ``finalize_predictions`` after concatenating all ranks.

    Empty shards (a rank that received no rows for a trajectory) are still written so
    ``finalize_predictions``'s glob-and-concat loop doesn't have to special-case missing files.

    Examples:
        >>> import tempfile
        >>> preds = [{
        ...     "tokens": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
        ...     "dataset_row_idxs": torch.tensor([0, 1, 2]),
        ...     "trajectory_idxs": torch.tensor([0, 1, 0]),
        ... }]
        >>> with tempfile.TemporaryDirectory() as d:
        ...     shard_dir = Path(d)
        ...     write_predictions_shards(preds, n_trajectories=2, shard_dir=shard_dir, rank=0)
        ...     names = sorted(p.name for p in shard_dir.glob("*.parquet"))
        ...     df = pl.read_parquet(shard_dir / "trajectory_0.rank_0.parquet")
        >>> names
        ['trajectory_0.rank_0.parquet', 'trajectory_1.rank_0.parquet']
        >>> df.to_dict(as_series=False)
        {'dataset_row_idx': [0, 2], 'tokens': [[1, 2], [5, 6]]}
    """
    per_t_batches, per_t_idxs = _demux_predictions_per_trajectory(predictions, n_trajectories)
    for t in range(n_trajectories):
        token_batches = per_t_batches[t]
        idx_batches = per_t_idxs[t]
        if token_batches:
            all_tokens = [row.tolist() for batch in token_batches for row in batch]
            all_idxs = torch.cat(idx_batches).tolist()
        else:
            all_tokens = []
            all_idxs = []
        shard_df = pl.DataFrame(
            {SHARD_ROW_IDX_COL: all_idxs, SHARD_TOKENS_COL: all_tokens},
            schema={SHARD_ROW_IDX_COL: pl.Int64, SHARD_TOKENS_COL: pl.List(pl.Int64)},
        )
        shard_path = shard_dir / f"trajectory_{t}.rank_{rank}.parquet"
        shard_df.write_parquet(shard_path)


def finalize_predictions(
    *,
    n_dataset_rows: int,
    shard_dir: Path,
    trajectory_paths: dict[int, Path],
    base_dataset,
    do_overwrite: bool,
    cleanup_shards: bool = True,
) -> None:
    """Merge rank shards per trajectory into the final parquets. Call on rank 0 only.

    For each trajectory in ``trajectory_paths``:

    1. ``pl.read_parquet`` + ``pl.concat`` + ``.sort("dataset_row_idx")`` across all
       ``trajectory_{t}.rank_*.parquet`` shards under ``shard_dir``.
    2. Validate the concatenated, sorted ``dataset_row_idx`` column is exactly
       ``[0, 1, ..., n_dataset_rows - 1]`` — catches missing rows, duplicates, and
       out-of-range values as a loud error instead of silently misaligning output.
    3. Reconstitute per-row generated-token tensors from the shard's ``tokens`` list column,
       emit as a list of ``[1, L_i]`` single-row tensors (variable ``L`` per row is preserved
       — ``format_trajectories`` already iterates per-batch), and hand off to
       ``format_trajectories`` + ``GeneratedTrajectorySchema.align`` + ``pq.write_table``.

    Per-file idempotent: if a trajectory's final parquet already exists and ``do_overwrite`` is
    ``False``, that trajectory is skipped (its shards are still cleaned up at the end unless
    ``cleanup_shards=False``). After all trajectories finish, shard files are removed and the
    shard directory itself is removed if empty.
    """
    expected_first = 0
    expected_last = n_dataset_rows - 1
    for t, out_fp in trajectory_paths.items():
        if out_fp.is_file() and not do_overwrite:
            logger.info(f"Skipping {out_fp} as it already exists.")
            continue

        shard_paths = sorted(shard_dir.glob(f"trajectory_{t}.rank_*.parquet"))
        if not shard_paths:
            raise RuntimeError(
                f"Trajectory {t}: no rank shards at {shard_dir}/trajectory_{t}.rank_*.parquet. "
                "write_predictions_shards must run on every rank before finalize_predictions."
            )

        merged = pl.concat([pl.read_parquet(p) for p in shard_paths]).sort(SHARD_ROW_IDX_COL)

        # Validate the concatenated, sorted idx column is the full permutation ``[0, ..., n-1]``.
        # After ``.sort``, completeness checks reduce to: right length, first == 0, last == n-1,
        # and every consecutive diff is 1 (no duplicates, no gaps). Done in polars-native ops so
        # we never materialize a Python list of N indices.
        idxs = merged[SHARD_ROW_IDX_COL]
        if len(idxs) != n_dataset_rows:
            raise RuntimeError(
                f"Trajectory {t} merged row count {len(idxs)} != n_dataset_rows {n_dataset_rows}. "
                "Check that every rank wrote its shard and that no rows were dropped upstream."
            )
        diffs = idxs.diff().drop_nulls()
        if idxs[0] != expected_first or idxs[-1] != expected_last or (diffs.len() > 0 and (diffs != 1).any()):
            preview = idxs.head(10).to_list()
            raise RuntimeError(
                f"Trajectory {t} dataset-row-index set is not "
                f"{{0, 1, ..., {expected_last}}}: got {preview}... (len {len(idxs)}). "
                "Every base-dataset row index must appear exactly once across all rank shards. "
                "A missing or duplicate index indicates a sampler bug, an accidental dataloader "
                "shuffle, or a missing / misnamed rank shard."
            )

        # Reconstitute per-row token tensors; emit as 1-row batches so ``format_trajectories``
        # can preserve each row's native ``L`` (no cross-row padding).
        token_rows = [torch.tensor(row, dtype=torch.long) for row in merged[SHARD_TOKENS_COL].to_list()]
        one_row_batches = [r.unsqueeze(0) for r in token_rows]

        logger.info(f"Writing trajectory {t} to {out_fp}.")
        predictions_df = format_trajectories(base_dataset, one_row_batches)
        pa_table = GeneratedTrajectorySchema.align(predictions_df.to_arrow())
        pq.write_table(pa_table, out_fp)

    if cleanup_shards:
        for p in shard_dir.glob("trajectory_*.rank_*.parquet"):
            p.unlink()
        # ``rmdir`` fails loudly if the dir has other contents or doesn't exist — that's the
        # safe default, we just don't care in either case.
        with contextlib.suppress(OSError):
            shard_dir.rmdir()
