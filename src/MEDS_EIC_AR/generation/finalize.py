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

import contextlib
import logging
from datetime import timedelta
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch
from MEDS_trajectory_evaluation.schema import GeneratedTrajectorySchema
from MEDS_transforms.stages.add_time_derived_measurements.utils import normalize_time_unit

from .repeated_dataset import PredictStepOutput

# ``CodeInformation`` is referenced only in the ``format_trajectories`` doctest — ruff
# otherwise auto-prunes it as unused. Keep it accessible in the module namespace so doctests
# can construct fake code-information dicts without a separate import line.
from .utils import CodeInformation, get_code_information  # noqa: F401

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
        ...         tokens=torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 8, 9]], dtype=torch.long),
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
        >>> df0.to_dict(as_series=False)
        {'dataset_row_idx': [0, 2], 'tokens': [[1, 2], [6, 7, 8, 9]]}
        >>> df1.to_dict(as_series=False)
        {'dataset_row_idx': [1], 'tokens': [[3, 4, 5]]}
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

    Raises:
        ValueError: If ``base_dataset``'s code metadata has any code with ``value_prob`` not in
            ``{0.0, 1.0}`` (this model assumes each code either always or never carries a value).

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
        ┌────────────┬───────────────────────┬─────────────────────┬───────────────────────┬───────────────┐
        │ subject_id ┆ time                  ┆ prediction_time     ┆ code                  ┆ numeric_value │
        │ ---        ┆ ---                   ┆ ---                 ┆ ---                   ┆ ---           │
        │ i64        ┆ datetime[μs]          ┆ datetime[μs]        ┆ str                   ┆ f32           │
        ╞════════════╪═══════════════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
        │ 239684     ┆ 2010-05-11 17:50:28   ┆ 2010-05-11 18:00:00 ┆ TIMELINE//DELTA//year ┆ 0.000003      │
        │            ┆                       ┆                     ┆ s//value_…            ┆               │
        │ 239684     ┆ 2010-05-11 17:50:28   ┆ 2010-05-11 18:00:00 ┆ DISCHARGE             ┆ null          │
        │ 239684     ┆ 2010-05-11 17:50:28   ┆ 2010-05-11 18:00:00 ┆ HR//value_[105.1,107. ┆ 105.099998    │
        │            ┆                       ┆                     ┆ 5)                    ┆               │
        │ 239684     ┆ 2010-05-11 17:50:28   ┆ 2010-05-11 18:00:00 ┆ DISCHARGE             ┆ null          │
        │ 239684     ┆ 2010-05-11 17:50:28   ┆ 2010-05-11 18:00:00 ┆ ADMISSION//PULMONARY  ┆ null          │
        │ 239684     ┆ 2010-05-11 17:50:28   ┆ 2010-05-11 18:00:00 ┆ DISCHARGE             ┆ null          │
        │ 239684     ┆ 2010-05-11 17:50:28   ┆ 2010-05-11 18:00:00 ┆ HR//value_[105.1,107. ┆ 105.099998    │
        │            ┆                       ┆                     ┆ 5)                    ┆               │
        │ 239684     ┆ 2010-05-11 17:50:28   ┆ 2010-05-11 18:00:00 ┆ HR//value_[105.1,107. ┆ 105.099998    │
        │            ┆                       ┆                     ┆ 5)                    ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:00:00 ┆ TIMELINE//DELTA//year ┆ 0.00004       │
        │            ┆ 18:11:40.400010       ┆                     ┆ s//value_…            ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:00:00 ┆ HR//value_[107.5,107. ┆ 107.5         │
        │            ┆ 18:11:40.400010       ┆                     ┆ 7)                    ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ TIMELINE//DELTA//year ┆ 0.000015      │
        │            ┆ 18:33:18.999983       ┆                     ┆ s//value_…            ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ HR//value_[107.7,112. ┆ 108.349998    │
        │            ┆ 18:33:18.999983       ┆                     ┆ 5)                    ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ TIMELINE//DELTA//year ┆ 0.00004       │
        │            ┆ 18:54:31.399993       ┆                     ┆ s//value_…            ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ HR//value_[107.5,107. ┆ 107.5         │
        │            ┆ 18:54:31.399993       ┆                     ┆ 7)                    ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ ADMISSION//CARDIAC    ┆ null          │
        │            ┆ 18:54:31.399993       ┆                     ┆                       ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ TIMELINE//END         ┆ null          │
        │            ┆ 18:54:31.399993       ┆                     ┆                       ┆               │
        └────────────┴───────────────────────┴─────────────────────┴───────────────────────┴───────────────┘

        Invalid code metadata raises:

        >>> with patch("MEDS_EIC_AR.generation.finalize.get_code_information") as mock:
        ...     mock.return_value = {1: CodeInformation(code='HR', value_prob=0.5, value_mean=106.0)}
        ...     format_trajectories("fake dataset", merged)
        Traceback (most recent call last):
          ...
        ValueError: Code HR has a value probability of 0.5, which is not 0.0 or 1.0. This is not supported.
    """
    code_information = get_code_information(base_dataset)

    for code_info in code_information.values():
        if code_info.value_prob not in {0.0, 1.0}:
            raise ValueError(
                f"Code {code_info.code} has a value probability of {code_info.value_prob}, "
                "which is not 0.0 or 1.0. This is not supported."
            )

    schema_df = base_dataset.schema_df.select(
        DataSchema.subject_id_name,
        LabelSchema.prediction_time_name,
        MEDSPytorchDataset.LAST_TIME,
    )

    rows: list[dict] = []
    for record in merged.iter_rows(named=True):
        schema_row = schema_df.row(record["dataset_row_idx"], named=True)
        subject_id = schema_row[DataSchema.subject_id_name]
        prediction_time = schema_row[LabelSchema.prediction_time_name]
        time = schema_row[MEDSPytorchDataset.LAST_TIME]

        for code_idx in record["tokens"]:
            if code_idx == MEDSTorchBatch.PAD_INDEX:
                continue
            code_info = code_information[code_idx]
            code = code_info.code
            value_mean = code_info.value_mean

            if code.startswith(TIMELINE_DELTA_TOKEN):
                unit = code.split("//")[-2]
                _, seconds_in_unit = normalize_time_unit(unit)
                time += timedelta(seconds=value_mean * seconds_in_unit)

            rows.append(
                {
                    DataSchema.subject_id_name: subject_id,
                    DataSchema.time_name: time,
                    LabelSchema.prediction_time_name: prediction_time,
                    DataSchema.code_name: code,
                    DataSchema.numeric_value_name: value_mean,
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            DataSchema.subject_id_name: pl.Int64,
            DataSchema.time_name: pl.Datetime,
            LabelSchema.prediction_time_name: pl.Datetime,
            DataSchema.code_name: pl.Utf8,
            DataSchema.numeric_value_name: pl.Float32,
        },
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

    1. ``pl.read_parquet`` + ``pl.concat`` + ``.sort("dataset_row_idx")`` across all
       ``trajectory_{t}.rank_*.parquet`` rank outputs under ``rank_outputs_dir``.
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

        merged = pl.concat([pl.read_parquet(p) for p in rank_paths]).sort("dataset_row_idx")
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
        # ``rmdir`` fails loudly if the dir has other contents or doesn't exist — don't care in
        # either case.
        with contextlib.suppress(OSError):
            rank_outputs_dir.rmdir()


__all__ = ["finalize_predictions", "format_trajectories", "write_rank_output"]
