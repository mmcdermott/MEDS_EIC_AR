"""Rank-aware shard-then-merge pipeline for finalizing ``MEICAR_generate_trajectories`` output.

The generation CLI calls two functions from this module, one before and one after a barrier:

- :func:`write_rank_output` — runs on **every** rank. Takes that rank's ``trainer.predict``
  output (a list of :class:`PredictStepOutput`) and writes one parquet per trajectory into a
  shared ``rank_outputs_dir``, named ``trajectory_{t}.rank_{rank}.parquet``. Shards carry
  ``dataset_row_idx`` (Int64) and ``tokens`` (List[Int64]) — the per-row trimmed token sequence
  (post-EOS ``PAD_INDEX`` stripping done at write time).
- :func:`finalize_predictions` — runs on **rank 0 only**, after an inter-rank barrier. For
  each trajectory, reads all rank outputs, concatenates them, validates
  ``dataset_row_idx`` coverage, and feeds the merged rows to :func:`format_trajectories` for
  the final MEDS-shaped parquet write.

The filesystem is the coordination mechanism: no ``BasePredictionWriter`` callback, no explicit
``torch.distributed.all_gather``. Works uniformly for single-device (``world_size=1``: one
rank output per trajectory, rank-0 merge is effectively a passthrough) and DDP (one rank output
per ``(trajectory, rank)``; rank 0's ``pl.concat`` stitches them together). Output row order is
driven by the ``dataset_row_idx`` join in :func:`format_trajectories` — arrival/concat order
across ranks is not semantically meaningful, so no sort is performed.

**Filesystem visibility requirement.** Under multi-node DDP, ``rank_outputs_dir`` (i.e. the
generation ``output_dir``) must live on a filesystem visible to every rank / every node.
Node-local scratch is the classic pitfall — rank 0 would only see its own node's rank outputs
and raise a "no rank outputs" or coverage error. Single-node runs (including multi-GPU on a
single host) have no such requirement since all ranks share the same filesystem by
construction.

**Timeline-delta preprocessing coupling.** Finalize hardcodes
:data:`TIMELINE_DELTA_TOKEN` = ``"TIMELINE//DELTA"`` to match the preprocessing pipeline's
default ``add_time_derived_measurements.timeline_tokens.time_delta_code``. Overriding that
prefix in preprocessing is not supported — :func:`_timeline_delta_seconds_per_unit` will raise
because no codes match. See the ``TIMELINE_DELTA_TOKEN`` comment below and
mmcdermott/MEDS_transforms#391 for the upstream follow-up.
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

logger = logging.getLogger(__name__)

# Must match the ``add_time_derived_measurements.timeline_tokens.time_delta_code`` default used
# by the preprocessing pipeline in ``preprocessing/configs/_data.yaml`` and
# ``preprocessing/configs/_reshard_data.yaml``. MEDS-transforms' ``timeline_tokens`` stage lets
# that prefix be overridden, but finalize has no way to recover a renamed prefix from the code
# metadata — so a non-default ``time_delta_code`` would silently skip all delta tokens here.
# See mmcdermott/MEDS_transforms#391 for the upstream follow-up to surface delta-code identity
# and unit via code metadata instead of this out-of-band coupling.
TIMELINE_DELTA_TOKEN = "TIMELINE//DELTA"


def _get_code_metadata(dataset: MEDSPytorchDataset) -> pl.DataFrame:
    """Return a DataFrame mapping code indices to their code strings and mean numeric values.

    Reads ``dataset.config.code_metadata_fp`` and produces a polars DataFrame with one row per
    vocabulary entry and columns:

    - ``code_idx`` (Int64): the vocabulary index used in generated tokens.
    - ``code`` (Utf8): the MEDS code string (e.g. ``TIMELINE//END``, ``HR//value_[102.6,105.1)``).
    - ``value_mean`` (Float32, nullable): the mean numeric value across observed occurrences,
      or ``null`` for codes that never carry a numeric value. The preprocessing pipeline
      (``quantile_binning`` + ``bin_numeric_values`` stages) guarantees that every code either
      always or never carries a value, so this column is a complete binary signal.

    **Note on bin-edge precision in code names.** For value-carrying bin codes like
    ``HR//value_[102.6,105.1)``, the ``[lo,hi)`` endpoints in the code *name* are rendered at
    limited decimal precision for readability by the upstream quantile-binning stage — they're
    not the exact bin edges. The underlying bin edges and ``value_mean`` are stored at full
    float32 precision and can differ from the displayed endpoint in the low-order digits
    (e.g. ``156.485596`` for a code named ``HEIGHT//value_[156.4856,...)``). This is not a bug;
    it's the upstream naming convention.

    Used only inside :func:`format_trajectories`; private to this module.

    Args:
        dataset: The dataset used for generation.

    Examples:
        >>> with pl.Config(tbl_rows=-1):
        ...     print(_get_code_metadata(pytorch_dataset).sort("code_idx"))
        shape: (38, 3)
        ┌──────────┬─────────────────────────────────┬────────────┐
        │ code_idx ┆ code                            ┆ value_mean │
        │ ---      ┆ ---                             ┆ ---        │
        │ i64      ┆ str                             ┆ f32        │
        ╞══════════╪═════════════════════════════════╪════════════╡
        │ 1        ┆ ADMISSION//CARDIAC              ┆ null       │
        │ 2        ┆ ADMISSION//ORTHOPEDIC           ┆ null       │
        │ 3        ┆ ADMISSION//PULMONARY            ┆ null       │
        │ 4        ┆ DISCHARGE                       ┆ null       │
        │ 5        ┆ EYE_COLOR//BLUE                 ┆ null       │
        │ 6        ┆ EYE_COLOR//BROWN                ┆ null       │
        │ 7        ┆ EYE_COLOR//HAZEL                ┆ null       │
        │ 8        ┆ HEIGHT//value_[156.4856,160.39… ┆ 156.485596 │
        │ 9        ┆ HEIGHT//value_[160.39531,164.6… ┆ 160.395309 │
        │ 10       ┆ HEIGHT//value_[164.68689,175.2… ┆ 164.68689  │
        │ 11       ┆ HEIGHT//value_[175.27112,inf)   ┆ 175.271118 │
        │ 12       ┆ HR//value_[-inf,102.6)          ┆ 86.0       │
        │ 13       ┆ HR//value_[102.6,105.1)         ┆ 102.599998 │
        │ 14       ┆ HR//value_[105.1,107.5)         ┆ 105.099998 │
        │ 15       ┆ HR//value_[107.5,107.7)         ┆ 107.5      │
        │ 16       ┆ HR//value_[107.7,112.5)         ┆ 108.349998 │
        │ 17       ┆ HR//value_[112.5,112.6)         ┆ 112.5      │
        │ 18       ┆ HR//value_[112.6,113.4)         ┆ 112.599998 │
        │ 19       ┆ HR//value_[113.4,114.1)         ┆ 113.400002 │
        │ 20       ┆ HR//value_[114.1,119.8)         ┆ 114.099998 │
        │ 21       ┆ HR//value_[119.8,inf)           ┆ 145.0      │
        │ 22       ┆ MEDS_BIRTH                      ┆ null       │
        │ 23       ┆ TEMP//value_[-inf,95.8)         ┆ 95.5       │
        │ 24       ┆ TEMP//value_[100.0,100.1)       ┆ 100.0      │
        │ 25       ┆ TEMP//value_[100.1,inf)         ┆ 100.25     │
        │ 26       ┆ TEMP//value_[95.8,96.0)         ┆ 95.800003  │
        │ 27       ┆ TEMP//value_[96.0,96.2)         ┆ 96.0       │
        │ 28       ┆ TEMP//value_[96.2,97.8)         ┆ 96.199997  │
        │ 29       ┆ TEMP//value_[97.8,99.9)         ┆ 98.800003  │
        │ 30       ┆ TEMP//value_[99.9,100.0)        ┆ 99.900002  │
        │ 31       ┆ TIMELINE//DELTA//years//value_… ┆ 0.000003   │
        │ 32       ┆ TIMELINE//DELTA//years//value_… ┆ 0.000015   │
        │ 33       ┆ TIMELINE//DELTA//years//value_… ┆ 0.00004    │
        │ 34       ┆ TIMELINE//DELTA//years//value_… ┆ 0.000065   │
        │ 35       ┆ TIMELINE//DELTA//years//value_… ┆ 0.000198   │
        │ 36       ┆ TIMELINE//DELTA//years//value_… ┆ 31.861664  │
        │ 37       ┆ TIMELINE//END                   ┆ null       │
        │ 38       ┆ TIMELINE//START                 ┆ null       │
        └──────────┴─────────────────────────────────┴────────────┘
    """
    columns = ["code", "code/vocab_index", "values/n_occurrences", "values/sum"]
    metadata = pl.read_parquet(dataset.config.code_metadata_fp, columns=columns, use_pyarrow=True)
    return metadata.select(
        pl.col("code/vocab_index").cast(pl.Int64).alias("code_idx"),
        pl.col("code"),
        pl.when(pl.col("values/n_occurrences") > 0)
        .then(pl.col("values/sum") / pl.col("values/n_occurrences"))
        .otherwise(None)
        .alias("value_mean"),
    )


def _timeline_delta_seconds_per_unit(code_metadata: pl.DataFrame) -> float:
    """Return the seconds-per-unit scalar for this dataset's ``TIMELINE//DELTA`` codes.

    The preprocessing pipeline's ``add_time_derived_measurements`` stage is configured with a
    single ``time_unit`` (currently ``years`` in
    ``preprocessing/configs/_data.yaml``), so every ``TIMELINE//DELTA//<unit>//value_[...]``
    code in the vocabulary shares the same unit segment. This helper extracts that unit from
    the vocabulary, asserts uniqueness (catches any future pipeline change that would
    introduce mixed units), and returns its seconds-per-unit scalar.

    Raises ``ValueError`` when no ``TIMELINE//DELTA`` codes exist in the vocabulary: without
    them, generation cannot advance per-row time, so the output would be degenerate. The most
    likely cause is a non-default ``add_time_derived_measurements.timeline_tokens.time_delta_code``
    in the preprocessing pipeline — :data:`TIMELINE_DELTA_TOKEN` hardcodes the default prefix
    and a renamed prefix is not supported at present.
    """
    delta_codes = code_metadata.filter(pl.col("code").str.starts_with(TIMELINE_DELTA_TOKEN))
    if delta_codes.height == 0:
        raise ValueError(
            f"No codes starting with {TIMELINE_DELTA_TOKEN!r} were found in the vocabulary. "
            "Generation needs delta codes to advance per-row time. This likely means the "
            "preprocessing pipeline's ``add_time_derived_measurements.timeline_tokens.time_delta_code`` "
            "was changed from the default; overriding that prefix is not supported at present "
            "because finalize has no way to recover the renamed prefix from the code metadata."
        )
    units = {c.split("//")[-2] for c in delta_codes["code"].to_list()}
    if len(units) != 1:
        raise ValueError(
            f"Expected exactly one TIMELINE//DELTA unit across the vocabulary, got {sorted(units)}. "
            "The preprocessing pipeline should be configured with a single "
            "``add_time_derived_measurements.timeline_tokens.time_unit``."
        )
    return normalize_time_unit(units.pop())[1]


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
    within a rank's output are *not* sorted, and :func:`finalize_predictions` doesn't sort
    either after concatenating: final output order is driven by ``format_trajectories``'s
    ``dataset_row_idx`` join, not by cross-rank arrival order.

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
        merged: A polars DataFrame with columns ``dataset_row_idx`` (Int64) and
            ``tokens`` (List[Int64]) — one row per base-dataset row covered by this
            trajectory. Row order is unspecified (this is what
            :func:`finalize_predictions`'s ``pl.concat([...rank outputs...])`` yields — no
            sort); final output ordering is driven by the ``dataset_row_idx`` join below.

    Returns:
        A polars DataFrame with the standard MEDS columns (``subject_id``, ``time``,
        ``prediction_time``, ``code``, ``numeric_value``) — one output row per generated code
        across all trajectories' rows.

    Examples:
        Set up two base-dataset rows, each with a generated token sequence:

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

        Per :func:`_get_code_metadata`'s doctest, in this vocabulary token IDs 31-36 are the
        ``TIMELINE//DELTA//years//...`` codes and the rest are non-delta. Delta tokens
        advance the per-row running time by ``value_mean * seconds_per_year`` (the single
        unit pinned by the preprocessing pipeline, resolved once by
        :func:`_timeline_delta_seconds_per_unit`); non-delta tokens just append an output
        row at the current time. The running time is the cumulative sum of delta
        microseconds across the token sequence within each ``dataset_row_idx``. So for row
        0's tokens ``[31, 4, 14, 4, 3, 4, 14, 14, 33, 15]``:

        - Token ``31`` (``TIMELINE//DELTA//years//...``, ``value_mean ≈ 3.17e-6``) advances
          time by ``3.17e-6 * 31556926 s/year ~= 100 s``.
        - Tokens ``4, 14, 4, 3, 4, 14, 14`` are non-delta — output rows at the current time.
        - Token ``33`` (``TIMELINE//DELTA//years//...``, ``value_mean ≈ 4.03e-5``) advances
          time by ``4.03e-5 * 31556926 s/year ~= 1272 s ~= 21 min 12 s``.
        - Token ``15`` (non-delta) at the advanced time.

        Row 1 follows the same pattern from the second base-dataset row's ``last_time``,
        with ``32`` then ``33`` being deltas and ``37`` a terminating ``TIMELINE//END``.
        (The residual ≤30µs drift from what pen-and-paper arithmetic gives is Int64 cumsum
        precision vs the Python ``timedelta`` reference; harmless at clinical time scales.)

        >>> with pl.Config(tbl_rows=-1):
        ...     print(format_trajectories(pytorch_dataset_with_task, merged))
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
        │            ┆ 18:11:40.400          ┆                     ┆ s//value_…            ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:00:00 ┆ HR//value_[107.5,107. ┆ 107.5         │
        │            ┆ 18:11:40.400          ┆                     ┆ 7)                    ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ TIMELINE//DELTA//year ┆ 0.000015      │
        │            ┆ 18:33:18.999968       ┆                     ┆ s//value_…            ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ HR//value_[107.7,112. ┆ 108.349998    │
        │            ┆ 18:33:18.999968       ┆                     ┆ 5)                    ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ TIMELINE//DELTA//year ┆ 0.00004       │
        │            ┆ 18:54:31.399968       ┆                     ┆ s//value_…            ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ HR//value_[107.5,107. ┆ 107.5         │
        │            ┆ 18:54:31.399968       ┆                     ┆ 7)                    ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ ADMISSION//CARDIAC    ┆ null          │
        │            ┆ 18:54:31.399968       ┆                     ┆                       ┆               │
        │ 239684     ┆ 2010-05-11            ┆ 2010-05-11 18:30:00 ┆ TIMELINE//END         ┆ null          │
        │            ┆ 18:54:31.399968       ┆                     ┆                       ┆               │
        └────────────┴───────────────────────┴─────────────────────┴───────────────────────┴───────────────┘
    """
    # Flatten the nested ``tokens`` column to one row per generated code, then join against
    # schema (for subject_id / prediction_time / last-time per base-dataset row) and against
    # the code-metadata DataFrame (for code string + value_mean per vocab index).
    schema_df = (
        base_dataset.schema_df.select(
            DataSchema.subject_id_name,
            LabelSchema.prediction_time_name,
            MEDSPytorchDataset.LAST_TIME,
        )
        .with_row_index("dataset_row_idx")
        # ``with_row_index`` materializes a UInt32 column; cast to Int64 to match ``merged``.
        .with_columns(pl.col("dataset_row_idx").cast(pl.Int64))
    )
    code_metadata = _get_code_metadata(base_dataset)
    seconds_per_unit = _timeline_delta_seconds_per_unit(code_metadata)

    # Explode the per-row list of tokens, strip any residual PAD_INDEX, and resolve each
    # ``code_idx`` to a code string + ``value_mean`` via left join. ``_trim_post_pad`` already
    # drops post-EOS padding at shard-write time; the explicit PAD filter is a belt-and-
    # suspenders guard for any hypothetical left-padded generator.
    with_codes = (
        merged.explode("tokens")
        .rename({"tokens": "code_idx"})
        .filter(pl.col("code_idx") != MEDSTorchBatch.PAD_INDEX)
        .join(code_metadata, on="code_idx", how="left")
    )
    # A null ``code`` means a generated ``code_idx`` had no row in the code-metadata parquet.
    # In normal use this can't happen — tokens are sampled from the model's own vocabulary.
    # When it does happen the most likely cause is a checkpoint/dataset mismatch: the
    # generation CLI was pointed at a preprocessing output whose vocabulary doesn't line up
    # with the one the model was trained on. Fail loudly instead of producing rows with null
    # ``code`` that would silently pass through GeneratedTrajectorySchema.align. Gate on a
    # scalar ``null_count()`` so the happy path avoids allocating a filtered frame.
    missing_code_count = with_codes["code"].null_count()
    if missing_code_count:
        missing_codes = with_codes.filter(pl.col("code").is_null())
        unknown = sorted(missing_codes["code_idx"].unique().to_list())[:20]
        raise RuntimeError(
            f"{missing_code_count} generated token(s) have code_idx values not present in "
            f"the code metadata (first 20: {unknown}). This usually indicates a checkpoint / "
            "code-metadata mismatch: the checkpoint passed to MEICAR_generate_trajectories "
            "was trained on a different preprocessing output than the one pointed at by "
            "``datamodule.config``."
        )

    with_rows = with_codes.join(schema_df, on="dataset_row_idx", how="left")
    # A null ``subject_id`` after the schema join means a merged row carries a
    # ``dataset_row_idx`` outside ``[0, len(base_dataset))``. ``finalize_predictions``'s
    # coverage check is supposed to catch this upstream — reaching it here means
    # ``format_trajectories`` was called directly on bad input, so the error is targeted at
    # that path. Same scalar-gate pattern — especially important here because ``with_rows``
    # is post-``explode`` and can be very large.
    missing_subject_id_count = with_rows[DataSchema.subject_id_name].null_count()
    if missing_subject_id_count:
        missing_rows = with_rows.filter(pl.col(DataSchema.subject_id_name).is_null())
        # ``missing_subject_id_count`` counts *token* rows, not base-dataset rows —
        # a single bad ``dataset_row_idx`` with many tokens would inflate the number.
        # Report the unique-id count alongside the raw token-row count so both signals
        # are visible.
        unknown_ids = sorted(missing_rows["dataset_row_idx"].unique().to_list())[:20]
        n_unknown_ids = missing_rows["dataset_row_idx"].n_unique()
        raise RuntimeError(
            f"{n_unknown_ids} unresolved dataset_row_idx value(s) across "
            f"{missing_subject_id_count} token-row(s) (first 20 ids: {unknown_ids}). These "
            "are not present in base_dataset.schema_df — expected the caller "
            "(finalize_predictions) to have already validated coverage over "
            "[0, len(base_dataset))."
        )

    # ``TIMELINE//DELTA`` tokens increment each row's running time by
    # ``value_mean * seconds_per_unit``. The preprocessing pipeline pins a single unit across
    # the whole vocabulary (see :func:`_timeline_delta_seconds_per_unit`), so this is just a
    # scalar multiplication — no per-row unit parsing, no unit-mapping join. Non-delta rows
    # contribute zero microseconds. Casting per-step BEFORE the cumsum matches Python's
    # ``timedelta(seconds=float)`` precision (it truncates each step to microseconds before
    # summing).
    return (
        with_rows.with_columns(
            delta_us=pl.when(pl.col("code").str.starts_with(TIMELINE_DELTA_TOKEN))
            .then((pl.col("value_mean") * seconds_per_unit * 1_000_000).cast(pl.Int64))
            .otherwise(0)
        )
        # Cumulative per-row: time = last_time + sum of all prior (and current) delta_us within
        # this dataset_row_idx. ``explode`` preserves source order within each list, so
        # ``cum_sum().over("dataset_row_idx")`` walks tokens in generation order.
        .with_columns(cum_delta_us=pl.col("delta_us").cum_sum().over("dataset_row_idx"))
        .select(
            pl.col(DataSchema.subject_id_name).cast(pl.Int64),
            (pl.col(MEDSPytorchDataset.LAST_TIME) + pl.duration(microseconds=pl.col("cum_delta_us")))
            .cast(pl.Datetime)
            .alias(DataSchema.time_name),
            pl.col(LabelSchema.prediction_time_name).cast(pl.Datetime),
            pl.col("code").alias(DataSchema.code_name),
            pl.col("value_mean").cast(pl.Float32).alias(DataSchema.numeric_value_name),
        )
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
       rank outputs under ``rank_outputs_dir``. No sort — :func:`format_trajectories` joins
       against ``schema_df`` on ``dataset_row_idx`` and order is not semantically meaningful.
    2. Coverage check: the merged ``dataset_row_idx`` values must equal
       ``{0, ..., n_dataset_rows - 1}`` exactly once each. Catches duplicate/missing rows
       with the same total count (e.g. two ranks both claiming row 0) which a pure length
       check would silently accept.
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
        Build per-trajectory predict outputs distributed across two ranks, merge them, and
        finalize to a per-trajectory parquet. Each rank sees a disjoint slice of
        ``dataset_row_idx`` — mirroring how Lightning's ``DistributedSampler`` partitions the
        predict dataloader under DDP — and writes its own
        ``trajectory_{t}.rank_{rank}.parquet``:

        >>> preds_r0 = [
        ...     PredictStepOutput(
        ...         tokens=torch.tensor([[31, 4, 14]], dtype=torch.long),
        ...         dataset_row_idxs=torch.tensor([0]),
        ...         trajectory_idxs=torch.tensor([0]),
        ...     ),
        ... ]
        >>> preds_r1 = [
        ...     PredictStepOutput(
        ...         tokens=torch.tensor([[32, 16, 33]], dtype=torch.long),
        ...         dataset_row_idxs=torch.tensor([1]),
        ...         trajectory_idxs=torch.tensor([0]),
        ...     ),
        ... ]
        >>> with tempfile.TemporaryDirectory() as d:
        ...     rod = Path(d) / "_rank_outputs"
        ...     rod.mkdir()
        ...     write_rank_output(preds_r0, rank=0, rank_outputs_dir=rod)
        ...     write_rank_output(preds_r1, rank=1, rank_outputs_dir=rod)
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

        Single-rank (``world_size=1``) is the same code path with one rank file —
        ``pl.concat`` over a one-element list is a passthrough:

        >>> preds_solo = [
        ...     PredictStepOutput(
        ...         tokens=torch.tensor([[31, 4, 14], [32, 16, 33]], dtype=torch.long),
        ...         dataset_row_idxs=torch.tensor([0, 1]),
        ...         trajectory_idxs=torch.tensor([0, 0]),
        ...     ),
        ... ]
        >>> with tempfile.TemporaryDirectory() as d:
        ...     rod = Path(d) / "_rank_outputs"
        ...     rod.mkdir()
        ...     write_rank_output(preds_solo, rank=0, rank_outputs_dir=rod)
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

        Duplicate or missing ``dataset_row_idx`` coverage across ranks raises a clear error
        instead of silently producing malformed output. Here rank 1 accidentally re-claims
        row 0, so row 1 is missing and row 0 is duplicated — the total count is still 2 and
        would pass a naive length check:

        >>> preds_r1_bad = [
        ...     PredictStepOutput(
        ...         tokens=torch.tensor([[5, 6, 7]], dtype=torch.long),
        ...         dataset_row_idxs=torch.tensor([0]),
        ...         trajectory_idxs=torch.tensor([0]),
        ...     ),
        ... ]
        >>> with tempfile.TemporaryDirectory() as d:
        ...     rod = Path(d) / "_rank_outputs"
        ...     rod.mkdir()
        ...     write_rank_output(preds_r0, rank=0, rank_outputs_dir=rod)
        ...     write_rank_output(preds_r1_bad, rank=1, rank_outputs_dir=rod)
        ...     finalize_predictions(
        ...         rank_outputs_dir=rod,
        ...         trajectory_paths={0: Path(d) / "bad.parquet"},
        ...         base_dataset=pytorch_dataset_with_task,
        ...         n_dataset_rows=2,
        ...         do_overwrite=True,
        ...     )
        Traceback (most recent call last):
            ...
        RuntimeError: Trajectory 0: rank outputs should cover dataset_row_idx ...
        First 20 missing: [1]; ...
    """
    for t, out_fp in trajectory_paths.items():
        if out_fp.is_file() and not do_overwrite:
            logger.info(f"Skipping {out_fp} as it already exists.")
            continue

        # Glob order is unspecified; that's fine — ``format_trajectories`` joins by
        # ``dataset_row_idx`` and the final output order is driven by that join, not by
        # rank-file arrival order.
        rank_paths = list(rank_outputs_dir.glob(f"trajectory_{t}.rank_*.parquet"))
        if not rank_paths:
            raise RuntimeError(
                f"Trajectory {t}: no rank outputs at "
                f"{rank_outputs_dir}/trajectory_{t}.rank_*.parquet. Either (a) write_rank_output "
                "was not run on every rank before finalize_predictions, or (b) under multi-node "
                "DDP, rank_outputs_dir is on a filesystem not visible to all ranks (e.g. node-"
                "local scratch). Point output_dir at shared storage."
            )

        merged = pl.concat([pl.read_parquet(p) for p in rank_paths])
        # Under DDP each rank writes a disjoint slice of dataset_row_idx; a correct merge
        # covers ``{0, ..., n_dataset_rows - 1}`` exactly once per trajectory. A pure
        # ``len(merged) == n_dataset_rows`` check misses duplicate/missing pairs (e.g.
        # ``[0, 1, 1, 3]`` vs ``[0, 1, 2, 3]`` — same length, wrong coverage). Validate the
        # three failure modes (wrong count / out-of-range / duplicate) using scalar
        # aggregates only: ``min``/``max`` for range, ``n_unique`` for duplicates. That
        # avoids allocating a filtered frame or a ``set(range(n_dataset_rows))`` on the
        # happy path — for 10M+ row cohorts those allocations alone would cost hundreds of
        # MB of transient overhead stacked on ``merged``. Only on the failure branch do we
        # materialize the out-of-range / missing samples for the error message.
        observed = merged["dataset_row_idx"]
        observed_count = observed.len()
        observed_min = observed.min() if observed_count else 0
        observed_max = observed.max() if observed_count else -1
        all_in_range = observed_min >= 0 and observed_max < n_dataset_rows
        observed_unique_count = observed.n_unique()
        if observed_count != n_dataset_rows or not all_in_range or observed_unique_count != n_dataset_rows:
            # Failure path — keep diagnostics bounded so a very large cohort can still raise
            # a clear error rather than cascading into an OOM. Don't build a Python ``set`` of
            # all ``n_dataset_rows`` ids; instead, sort the unique in-range observed ids and
            # scan them for gaps, stopping after the first 20. Out-of-range sampling uses
            # polars' own ``.head(20)`` for the same reason.
            out_of_range = (
                observed.filter((observed < 0) | (observed >= n_dataset_rows))
                .unique()
                .sort()
                .head(20)
                .to_list()
            )
            in_range_unique_sorted = (
                observed.filter((observed >= 0) & (observed < n_dataset_rows)).unique().sort()
            )
            missing: list[int] = []
            next_expected = 0
            for idx_val in in_range_unique_sorted:
                idx_val = int(idx_val)
                while next_expected < idx_val and len(missing) < 20:
                    missing.append(next_expected)
                    next_expected += 1
                if len(missing) >= 20:
                    break
                next_expected = idx_val + 1
            while next_expected < n_dataset_rows and len(missing) < 20:
                missing.append(next_expected)
                next_expected += 1
            raise RuntimeError(
                f"Trajectory {t}: rank outputs should cover dataset_row_idx exactly once "
                f"each over {{0, ..., {n_dataset_rows - 1}}}; got {observed_count} rows with "
                f"{observed_unique_count} unique ids. First 20 missing: {missing}; first 20 "
                f"out-of-range: {out_of_range}. Possible causes: (a) write_rank_output was "
                "not run on every rank before finalize_predictions; (b) under multi-node "
                "DDP, rank_outputs_dir is on a filesystem not visible to all ranks (e.g. "
                "node-local scratch); (c) a rank wrote stale shards from a previous run."
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
