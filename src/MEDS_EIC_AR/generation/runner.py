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

**Filesystem visibility requirement.** For multi-node DDP runs, ``shard_dir`` (i.e. the
generation ``output_dir``) must live on a filesystem visible to every rank / every node —
otherwise rank 0's ``finalize_predictions`` will only see its own node's shards and raise a
"no rank shards" or set-completeness error. Node-local scratch is a classic pitfall here.
Single-node runs (including multi-GPU on a single host) have no such requirement since all
ranks share the same filesystem by construction.
"""

import contextlib
import logging
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
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

    For multi-node DDP, ``shard_dir`` must be on a filesystem visible to every rank (rank 0's
    finalize glob must see every rank's shard). See the module docstring.

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

        # Build the shard's columns tensor-side so we never materialize ``total_tokens``
        # Python ints just to hand them to polars. For each ``[B_i, L_i]`` batch, every row
        # has length ``L_i``, so row lengths are a single ``torch.full(B_i, L_i)`` per batch;
        # the flat token stream is ``batch.reshape(-1)`` concatenated across batches in row-
        # major order. pyarrow ``ListArray.from_arrays`` then wraps offsets + flat values
        # into the nested-list column polars/parquet expects, zero-copy from ``.numpy()`` on
        # CPU tensors.
        if token_batches:
            row_lens = torch.cat(
                [torch.full((batch.shape[0],), batch.shape[1], dtype=torch.int64) for batch in token_batches]
            )
            flat_tokens = torch.cat([batch.reshape(-1) for batch in token_batches]).to(torch.int64)
            all_idxs = torch.cat(idx_batches).to(torch.int64)
        else:
            row_lens = torch.empty(0, dtype=torch.int64)
            flat_tokens = torch.empty(0, dtype=torch.int64)
            all_idxs = torch.empty(0, dtype=torch.int64)

        # 1:1 correspondence between token rows and dataset_row_idxs. Both are demuxed from
        # the same predict_step output dict by the same mask in
        # ``_demux_predictions_per_trajectory``, so this should always hold. Guarding
        # explicitly surfaces any upstream plumbing drift as a loud ``RuntimeError`` here
        # instead of a silently corrupt shard on disk.
        if row_lens.numel() != all_idxs.numel():
            raise RuntimeError(
                f"Trajectory {t} (rank {rank}): token-row count {row_lens.numel()} != "
                f"dataset_row_idx count {all_idxs.numel()}. Demux invariant violated — "
                "check for upstream masking / metadata drift in predict_step."
            )

        # pyarrow ``ListArray`` uses int32 offsets. For realistic cohort scale (e.g. 100k
        # rows of 1000 tokens each = 1e8 total, well under 2^31 - 1 ~= 2.1e9) this is plenty;
        # guard loudly against the theoretical overflow so an unexpectedly huge run doesn't
        # silently corrupt offset arithmetic.
        total_tokens = int(flat_tokens.numel())
        if total_tokens > 2**31 - 1:
            raise RuntimeError(
                f"Trajectory {t} (rank {rank}): total token count {total_tokens} exceeds "
                "int32 ListArray offset limit. Upgrade to LargeListArray if this scale is "
                "real; currently this path assumes int32 offsets are sufficient."
            )
        offsets_i32 = (
            torch.cat([torch.zeros(1, dtype=torch.int64), row_lens.cumsum(0)]).to(torch.int32).numpy()
        )
        tokens_arr = pa.ListArray.from_arrays(
            pa.array(offsets_i32, type=pa.int32()),
            pa.array(flat_tokens.numpy(), type=pa.int64()),
        )
        idxs_arr = pa.array(all_idxs.numpy(), type=pa.int64())
        table = pa.table({SHARD_ROW_IDX_COL: idxs_arr, SHARD_TOKENS_COL: tokens_arr})
        shard_path = shard_dir / f"trajectory_{t}.rank_{rank}.parquet"
        pq.write_table(table, shard_path)


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

        # Empty split (``n_dataset_rows == 0``). Nothing to validate, nothing to format, and
        # ``format_trajectories`` would raise on an empty batch list anyway. Log + skip so
        # downstream consumers see a missing parquet (same contract as "nothing to generate")
        # rather than an ``IndexError`` from the completeness check below.
        if n_dataset_rows == 0:
            logger.info(
                f"Trajectory {t}: split is empty (n_dataset_rows == 0); skipping finalize "
                f"and not writing {out_fp}."
            )
            continue

        shard_paths = sorted(shard_dir.glob(f"trajectory_{t}.rank_*.parquet"))
        if not shard_paths:
            raise RuntimeError(
                f"Trajectory {t}: no rank shards at {shard_dir}/trajectory_{t}.rank_*.parquet. "
                "Either (a) write_predictions_shards was not run on every rank before "
                "finalize_predictions, or (b) under multi-node DDP, shard_dir is on a "
                "filesystem not visible to all ranks (e.g. node-local scratch). Generation "
                "requires shared storage across ranks; point output_dir at a shared filesystem."
            )

        merged = pl.concat([pl.read_parquet(p) for p in shard_paths]).sort(SHARD_ROW_IDX_COL)

        # Validate the concatenated, sorted idx column is the full permutation ``[0, ..., n-1]``.
        # After ``.sort``, completeness checks reduce to: right length, first == 0, last == n-1,
        # and every consecutive diff is 1 (no duplicates, no gaps). Done in polars-native ops so
        # we never materialize a Python list of N indices. (The empty-split case has already
        # been short-circuited above, so ``idxs[0]`` / ``idxs[-1]`` are safe here.)
        idxs = merged[SHARD_ROW_IDX_COL]
        if len(idxs) != n_dataset_rows:
            raise RuntimeError(
                f"Trajectory {t} merged row count {len(idxs)} != n_dataset_rows {n_dataset_rows}. "
                "Check that every rank wrote its shard and that no rows were dropped upstream. "
                "Under multi-node DDP, also verify shard_dir is on shared storage."
            )
        diffs = idxs.diff().drop_nulls()
        if idxs[0] != expected_first or idxs[-1] != expected_last or (diffs.len() > 0 and (diffs != 1).any()):
            preview = idxs.head(10).to_list()
            raise RuntimeError(
                f"Trajectory {t} dataset-row-index set is not "
                f"{{0, 1, ..., {expected_last}}}: got {preview}... (len {len(idxs)}). "
                "Every base-dataset row index must appear exactly once across all rank shards. "
                "A missing or duplicate index indicates a sampler bug, an accidental dataloader "
                "shuffle, a missing / misnamed rank shard, or — on multi-node DDP — a "
                "shard_dir on node-local storage that rank 0 can't fully see."
            )

        # Reconstitute per-row token tensors as zero-copy views into the shard's Arrow flat
        # values buffer rather than going through ``Series.to_list`` (which would materialize
        # ``total_tokens`` Python ints on rank 0). ``torch.from_numpy`` is zero-copy for CPU
        # arrays, and slicing ``values_np[offsets[i]:offsets[i+1]]`` is a view, so each per-
        # row tensor shares the one underlying buffer. Emit as 1-row batches so
        # ``format_trajectories`` preserves each row's native ``L`` (no cross-row padding).
        tokens_arrow = merged[SHARD_TOKENS_COL].to_arrow()
        if isinstance(tokens_arrow, pa.ChunkedArray):
            tokens_arrow = tokens_arrow.combine_chunks()
        offsets_np = np.asarray(tokens_arrow.offsets)
        values_np = np.asarray(tokens_arrow.values)
        one_row_batches = [
            torch.from_numpy(values_np[offsets_np[i] : offsets_np[i + 1]]).unsqueeze(0)
            for i in range(len(offsets_np) - 1)
        ]

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
