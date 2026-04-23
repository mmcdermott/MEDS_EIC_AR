"""Rank-aware shard-then-merge pipeline for finalizing ``MEICAR_generate_trajectories`` output.

The generation CLI calls two functions from this module, one before and one after a barrier:

- :func:`write_rank_output` — runs on **every** rank. Takes that rank's ``trainer.predict``
  output (a list of :class:`PredictStepOutput`) and writes one parquet per trajectory into a
  shared ``rank_outputs_dir``, named ``trajectory_{t}.rank_{rank}.parquet``. Shards carry
  ``dataset_row_idx`` (Int64) and ``tokens`` (List[Int64]) — the per-row trimmed token sequence
  (post-EOS ``PAD_INDEX`` stripping done at write time).
- :func:`finalize_predictions` — runs on **rank 0 only**, after an inter-rank barrier. For
  each trajectory, reads all rank outputs, concatenates + sorts by ``dataset_row_idx``, and
  feeds the merged rows to :func:`format_trajectories` for the final MEDS-shaped parquet write.

The filesystem is the coordination mechanism: no ``BasePredictionWriter`` callback, no explicit
``torch.distributed.all_gather``. Works uniformly for single-device (``world_size=1``: one
rank output per trajectory, rank-0 merge is effectively a passthrough) and DDP (one rank output
per ``(trajectory, rank)``; rank 0's ``pl.concat + sort`` stitches them together).

**Filesystem visibility requirement.** Under multi-node DDP, ``rank_outputs_dir`` (i.e. the
generation ``output_dir``) must live on a filesystem visible to every rank / every node.
Node-local scratch is the classic pitfall — rank 0 would only see its own node's rank outputs
and raise a "no rank outputs" or row-count error. Single-node runs (including multi-GPU on a
single host) have no such requirement since all ranks share the same filesystem by
construction.
"""

import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch
from MEDS_trajectory_evaluation.schema import GeneratedTrajectorySchema
from MEDS_transforms.stages.add_time_derived_measurements.utils import normalize_time_unit

from .repeated_dataset import PredictStepOutput
from .utils import get_code_information

logger = logging.getLogger(__name__)

TIMELINE_DELTA_TOKEN = "TIMELINE//DELTA"


def _trim_post_pad(row: torch.Tensor) -> list[int]:
    """Truncate a 1-D token tensor at the first ``MEDSTorchBatch.PAD_INDEX`` and return as a Python list.

    The generator pads each batch row to the per-batch max ``L`` with ``PAD_INDEX`` after the
    row's EOS. Those padding tokens are meaningless downstream (``format_trajectories`` skips
    them) and just bloat the shard. Strip them at write time.

    Examples:
        >>> _trim_post_pad(torch.tensor([1, 2, 3, 0, 0], dtype=torch.long))
        [1, 2, 3]
        >>> _trim_post_pad(torch.tensor([0, 0, 0], dtype=torch.long))
        []
        >>> _trim_post_pad(torch.tensor([5, 6, 7], dtype=torch.long))
        [5, 6, 7]
    """
    pad_at = (row == MEDSTorchBatch.PAD_INDEX).nonzero(as_tuple=False)
    end = pad_at[0, 0].item() if pad_at.numel() > 0 else row.numel()
    return row[:end].tolist()


def write_rank_output(
    outputs: list[PredictStepOutput],
    *,
    rank: int,
    rank_outputs_dir: Path,
) -> None:
    """Write this rank's predict outputs as per-trajectory parquet files.

    Called by every rank (single-device = rank 0 with ``world_size=1``, or each rank in DDP).
    For each trajectory index present in ``outputs``, writes a file
    ``trajectory_{t}.rank_{rank}.parquet`` under ``rank_outputs_dir`` with columns:

    - ``dataset_row_idx`` (Int64): the base-dataset row index for each prediction.
    - ``tokens`` (List[Int64]): the generated token sequence, trimmed to the row's native
      length (post-EOS ``PAD_INDEX`` stripped).

    Under DDP each rank sees only its local slice of predictions — the rank outputs for a
    trajectory together span ``{0, ..., n_dataset_rows - 1}`` across all rank files. Rows
    within a rank's output are *not* sorted; the final sort happens in
    :func:`finalize_predictions` after concatenating all ranks.

    No ``n_trajectories`` argument: the set of trajectory indices is discovered from the
    ``trajectory_idxs`` field of the inputs.

    For multi-node DDP, ``rank_outputs_dir`` must be on a filesystem visible to every rank
    (rank 0's finalize has to see every rank's file). See the module docstring.

    Examples:
        >>> outs = [
        ...     PredictStepOutput(
        ...         tokens=torch.tensor([[1, 2, 0], [3, 4, 5], [6, 7, 8]], dtype=torch.long),
        ...         dataset_row_idxs=torch.tensor([0, 1, 2]),
        ...         trajectory_idxs=torch.tensor([0, 1, 0]),
        ...     ),
        ... ]
        >>> with tempfile.TemporaryDirectory() as d:
        ...     rod = Path(d)
        ...     write_rank_output(outs, rank=0, rank_outputs_dir=rod)
        ...     names = sorted(p.name for p in rod.glob("*.parquet"))
        ...     df0 = pl.read_parquet(rod / "trajectory_0.rank_0.parquet")
        ...     df1 = pl.read_parquet(rod / "trajectory_1.rank_0.parquet")
        >>> names
        ['trajectory_0.rank_0.parquet', 'trajectory_1.rank_0.parquet']
        >>> df0
        shape: (2, 2)
        ┌─────────────────┬───────────┐
        │ dataset_row_idx ┆ tokens    │
        │ ---             ┆ ---       │
        │ i64             ┆ list[i64] │
        ╞═════════════════╪═══════════╡
        │ 0               ┆ [1, 2]    │
        │ 2               ┆ [6, 7, 8] │
        └─────────────────┴───────────┘
        >>> df1
        shape: (1, 2)
        ┌─────────────────┬───────────┐
        │ dataset_row_idx ┆ tokens    │
        │ ---             ┆ ---       │
        │ i64             ┆ list[i64] │
        ╞═════════════════╪═══════════╡
        │ 1               ┆ [3, 4, 5] │
        └─────────────────┴───────────┘
    """
    per_trajectory = PredictStepOutput.split_by_trajectory(outputs)
    for t, t_outputs in per_trajectory.items():
        idxs: list[int] = []
        tokens: list[list[int]] = []
        for batch in t_outputs:
            for i in range(batch.tokens.shape[0]):
                idxs.append(int(batch.dataset_row_idxs[i].item()))
                tokens.append(_trim_post_pad(batch.tokens[i]))
        shard_df = pl.DataFrame(
            {"dataset_row_idx": idxs, "tokens": tokens},
            schema={"dataset_row_idx": pl.Int64, "tokens": pl.List(pl.Int64)},
        )
        shard_path = rank_outputs_dir / f"trajectory_{t}.rank_{rank}.parquet"
        shard_df.write_parquet(shard_path)


def format_trajectories(
    base_dataset: MEDSPytorchDataset,
    merged: pl.DataFrame,
) -> pl.DataFrame:
    """Translate merged per-trajectory generated tokens into a MEDS-shaped DataFrame.

    Args:
        base_dataset: The un-expanded dataset used for generation. ``schema_df`` supplies
            ``(subject_id, prediction_time, last-observed-time)`` per base-dataset row;
            ``config.code_metadata_fp`` supplies the vocabulary.
        merged: A polars DataFrame sorted by ``dataset_row_idx``, with columns
            ``dataset_row_idx`` (Int64) and ``tokens`` (List[Int64]) — one row per base-dataset
            row covered by this trajectory. This is what
            :func:`finalize_predictions`'s ``pl.concat([...rank outputs...]).sort(...)`` produces.

    Returns:
        A polars DataFrame with the standard MEDS columns (``subject_id``, ``time``,
        ``prediction_time``, ``code``, ``numeric_value``) — one output row per generated code
        across all trajectories' rows.

    Examples:
        >>> merged = pl.DataFrame(
        ...     {
        ...         "dataset_row_idx": [0, 1],
        ...         "tokens": [
        ...             [31, 4, 14, 4, 3, 4, 14, 14, 33, 15],
        ...             [32, 16, 33, 15, 1, 37],
        ...         ],
        ...     },
        ...     schema={"dataset_row_idx": pl.Int64, "tokens": pl.List(pl.Int64)},
        ... )
        >>> _ = pl.Config().set_tbl_rows(-1)
        >>> format_trajectories(pytorch_dataset_with_task, merged)
        shape: (16, 5)
        ┌────────────┬─────────────────┬─────────────────────┬─────────────────────────────┬───────────────┐
        │ subject_id ┆ time            ┆ prediction_time     ┆ code                        ┆ numeric_value │
        │ ---        ┆ ---             ┆ ---                 ┆ ---                         ┆ ---           │
        │ i64        ┆ datetime[μs]    ┆ datetime[μs]        ┆ str                         ┆ f32           │
        ╞════════════╪═════════════════╪═════════════════════╪═════════════════════════════╪═══════════════╡
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ TIMELINE//DELTA//years//val ┆ 0.000003      │
        │            ┆ 17:50:27.999999 ┆                     ┆ ue_…                        ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ DISCHARGE                   ┆ null          │
        │            ┆ 17:50:27.999999 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ HR//value_[105.1,107.5)     ┆ 105.099998    │
        │            ┆ 17:50:27.999999 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ DISCHARGE                   ┆ null          │
        │            ┆ 17:50:27.999999 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ ADMISSION//PULMONARY        ┆ null          │
        │            ┆ 17:50:27.999999 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ DISCHARGE                   ┆ null          │
        │            ┆ 17:50:27.999999 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ HR//value_[105.1,107.5)     ┆ 105.099998    │
        │            ┆ 17:50:27.999999 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ HR//value_[105.1,107.5)     ┆ 105.099998    │
        │            ┆ 17:50:27.999999 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ TIMELINE//DELTA//years//val ┆ 0.00004       │
        │            ┆ 18:11:40.400032 ┆                     ┆ ue_…                        ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:00:00 ┆ HR//value_[107.5,107.7)     ┆ 107.5         │
        │            ┆ 18:11:40.400032 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:30:00 ┆ TIMELINE//DELTA//years//val ┆ 0.000015      │
        │            ┆ 18:33:18.999982 ┆                     ┆ ue_…                        ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:30:00 ┆ HR//value_[107.7,112.5)     ┆ 108.349998    │
        │            ┆ 18:33:18.999982 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:30:00 ┆ TIMELINE//DELTA//years//val ┆ 0.00004       │
        │            ┆ 18:54:31.400015 ┆                     ┆ ue_…                        ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:30:00 ┆ HR//value_[107.5,107.7)     ┆ 107.5         │
        │            ┆ 18:54:31.400015 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:30:00 ┆ ADMISSION//CARDIAC          ┆ null          │
        │            ┆ 18:54:31.400015 ┆                     ┆                             ┆               │
        │ 239684     ┆ 2010-05-11      ┆ 2010-05-11 18:30:00 ┆ TIMELINE//END               ┆ null          │
        │            ┆ 18:54:31.400015 ┆                     ┆                             ┆               │
        └────────────┴─────────────────┴─────────────────────┴─────────────────────────────┴───────────────┘
    """
    # Flatten the nested ``tokens`` column to one row per generated code, then join against
    # schema (for subject_id / prediction_time / last-time per base-dataset row) and against
    # the code-metadata DataFrame (for code string + value_mean per vocab index). Everything
    # after is vectorized polars ops — no Python iter_rows loop.
    schema_df = base_dataset.schema_df.select(
        DataSchema.subject_id_name,
        LabelSchema.prediction_time_name,
        MEDSPytorchDataset.LAST_TIME,
    ).with_row_index("dataset_row_idx")
    # ``with_row_index`` materializes a UInt32 column; cast to Int64 to match ``merged``.
    schema_df = schema_df.with_columns(pl.col("dataset_row_idx").cast(pl.Int64))
    code_metadata = get_code_information(base_dataset)

    long = (
        merged.explode("tokens")
        .rename({"tokens": "code_idx"})
        # ``_trim_post_pad`` strips post-EOS padding at shard-write time, so in normal operation
        # there should be no PAD_INDEX rows here. Filter belt-and-suspenders in case a
        # hypothetical left-padded generator ever inserts them.
        .filter(pl.col("code_idx") != MEDSTorchBatch.PAD_INDEX)
        .join(code_metadata, on="code_idx", how="left")
        .join(schema_df, on="dataset_row_idx", how="left")
    )

    # ``TIMELINE//DELTA`` tokens increment each row's running time by
    # ``value_mean * seconds_per_unit``. The unit is the second-to-last segment of the code
    # string (e.g. ``TIMELINE//DELTA//years//value_[...]`` → ``years``). Extract the unit,
    # look up seconds-per-unit via a small mapping DataFrame built from the units actually
    # present in the data (calling ``normalize_time_unit`` once per unique unit — typically
    # a handful), and compute the per-row delta. Non-delta rows contribute zero seconds.
    long = long.with_columns(
        # ``null_on_oob`` is needed because polars evaluates both branches of
        # ``when-then-otherwise``; non-delta codes can have fewer than 2 ``//``-separated
        # segments and would otherwise raise ``ComputeError`` on the ``.list.get(-2)``.
        pl.when(pl.col("code").str.starts_with(TIMELINE_DELTA_TOKEN))
        .then(pl.col("code").str.split("//").list.get(-2, null_on_oob=True))
        .otherwise(None)
        .alias("unit"),
    )
    unique_units = long["unit"].drop_nulls().unique().to_list()
    unit_to_seconds = {u: normalize_time_unit(u)[1] for u in unique_units}
    unit_mapping = pl.DataFrame(
        {"unit": list(unit_to_seconds.keys()), "seconds_per_unit": list(unit_to_seconds.values())},
        schema={"unit": pl.Utf8, "seconds_per_unit": pl.Float64},
    )
    long = long.join(unit_mapping, on="unit", how="left").with_columns(
        # Per-row microsecond delta. Casting per-step BEFORE the cumsum matches the precision
        # of Python's ``timedelta(seconds=float)`` addition, which truncates each ``seconds``
        # float to integer microseconds individually before summing.
        delta_us=(pl.col("value_mean") * pl.col("seconds_per_unit").fill_null(0.0) * 1_000_000)
        .cast(pl.Int64)
        .fill_null(0),
    )

    # Cumulative per-row: time = last_time + sum of all prior (and current) delta_us within
    # this dataset_row_idx. ``explode`` preserves source order within each list, and
    # ``merged`` within a single ``dataset_row_idx`` has its tokens in generation order, so
    # ``cum_sum().over("dataset_row_idx")`` walks them in the right order.
    long = long.with_columns(
        cum_delta_us=pl.col("delta_us").cum_sum().over("dataset_row_idx"),
    )

    return long.select(
        pl.col(DataSchema.subject_id_name).cast(pl.Int64),
        (pl.col(MEDSPytorchDataset.LAST_TIME) + pl.duration(microseconds=pl.col("cum_delta_us")))
        .cast(pl.Datetime)
        .alias(DataSchema.time_name),
        pl.col(LabelSchema.prediction_time_name).cast(pl.Datetime),
        pl.col("code").alias(DataSchema.code_name),
        pl.col("value_mean").cast(pl.Float32).alias(DataSchema.numeric_value_name),
    )


def finalize_predictions(
    *,
    rank_outputs_dir: Path,
    trajectory_paths: dict[int, Path],
    base_dataset: MEDSPytorchDataset,
    n_dataset_rows: int,
    do_overwrite: bool,
    cleanup_rank_outputs: bool = True,
) -> None:
    """Merge rank outputs per trajectory into the final parquets. Call on rank 0 only.

    For each trajectory in ``trajectory_paths``:

    1. ``pl.read_parquet`` + ``pl.concat`` across all ``trajectory_{t}.rank_*.parquet``
       rank outputs under ``rank_outputs_dir``. No sort needed — :func:`format_trajectories`
       does a join against ``schema_df`` on ``dataset_row_idx``, so arrival order is
       irrelevant.
    2. Sanity check: ``len(merged) == n_dataset_rows``.
    3. :func:`format_trajectories` on the merged DataFrame → final polars DataFrame.
    4. :class:`GeneratedTrajectorySchema.align` + ``pq.write_table`` to the trajectory's
       output path.

    Per-file idempotent: if a trajectory's final parquet already exists and ``do_overwrite`` is
    ``False``, it's skipped (rank outputs still clean up at the end unless
    ``cleanup_rank_outputs=False``).

    The callers assume:

    - :func:`write_rank_output` has already run on every rank into ``rank_outputs_dir`` (enforced
      by a ``trainer.strategy.barrier()`` between those two calls in the CLI).
    - The empty-split case (``n_dataset_rows == 0``) is handled at the CLI entry — we're never
      called with an empty split.

    Examples:
        Build a single-rank predict output for two base-dataset rows, merge it, and finalize
        to a per-trajectory parquet:

        >>> preds = [
        ...     PredictStepOutput(
        ...         tokens=torch.tensor([[31, 4, 14], [32, 16, 33]], dtype=torch.long),
        ...         dataset_row_idxs=torch.tensor([0, 1]),
        ...         trajectory_idxs=torch.tensor([0, 0]),
        ...     ),
        ... ]
        >>> with tempfile.TemporaryDirectory() as d:
        ...     rod = Path(d) / "_rank_outputs"
        ...     rod.mkdir()
        ...     write_rank_output(preds, rank=0, rank_outputs_dir=rod)
        ...     out_fp = Path(d) / "trajectory_0.parquet"
        ...     finalize_predictions(
        ...         rank_outputs_dir=rod,
        ...         trajectory_paths={0: out_fp},
        ...         base_dataset=pytorch_dataset_with_task,
        ...         n_dataset_rows=2,
        ...         do_overwrite=True,
        ...     )
        ...     df = pl.read_parquet(out_fp)
        >>> df.shape
        (6, 5)
        >>> df.columns
        ['subject_id', 'time', 'code', 'numeric_value', 'prediction_time']
    """
    for t, out_fp in trajectory_paths.items():
        if out_fp.is_file() and not do_overwrite:
            logger.info(f"Skipping {out_fp} as it already exists.")
            continue

        rank_paths = sorted(rank_outputs_dir.glob(f"trajectory_{t}.rank_*.parquet"))
        if not rank_paths:
            raise RuntimeError(
                f"Trajectory {t}: no rank outputs at "
                f"{rank_outputs_dir}/trajectory_{t}.rank_*.parquet. Either (a) write_rank_output "
                "was not run on every rank before finalize_predictions, or (b) under multi-node "
                "DDP, rank_outputs_dir is on a filesystem not visible to all ranks (e.g. node-"
                "local scratch). Point output_dir at shared storage."
            )

        merged = pl.concat([pl.read_parquet(p) for p in rank_paths])
        if len(merged) != n_dataset_rows:
            raise RuntimeError(
                f"Trajectory {t}: merged row count {len(merged)} != n_dataset_rows "
                f"{n_dataset_rows}. Check that every rank wrote its output and that no rows "
                "were dropped upstream. Under multi-node DDP, also verify rank_outputs_dir is on "
                "shared storage."
            )

        logger.info(f"Writing trajectory {t} to {out_fp}.")
        predictions_df = format_trajectories(base_dataset, merged)
        pa_table = GeneratedTrajectorySchema.align(predictions_df.to_arrow())
        pq.write_table(pa_table, out_fp)

    if cleanup_rank_outputs:
        for p in rank_outputs_dir.glob("trajectory_*.rank_*.parquet"):
            p.unlink()
        # ``rmdir`` fails if the dir has other contents or doesn't exist. Neither is a
        # problem — the rank-output files we wrote are already unlinked above — but log
        # at debug so an unexpected failure (permission issue, racing process) leaves a
        # breadcrumb instead of vanishing silently.
        try:
            rank_outputs_dir.rmdir()
        except OSError as exc:
            logger.debug(f"rmdir({rank_outputs_dir}) did not succeed: {exc}")


__all__ = ["finalize_predictions", "format_trajectories", "write_rank_output"]
