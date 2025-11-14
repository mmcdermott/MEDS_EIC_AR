from datetime import timedelta
from typing import NamedTuple

import polars as pl
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset
from MEDS_transforms.stages.add_time_derived_measurements.utils import normalize_time_unit

TIMELINE_DELTA_TOKEN = "TIMELINE//DELTA"


class CodeInformation(NamedTuple):
    code: str
    value_prob: float
    value_mean: float | None


def get_code_information(dataset: MEDSPytorchDataset) -> dict[int, CodeInformation]:
    """Returns a dictionary mapping code indices to their code strings and numeric value means.

    Args:
        dataset: The dataset used for generation.

    Returns:
        A dictionary mapping code indices to their code strings and numeric value means.

    Examples:
        >>> get_code_information(pytorch_dataset)
        {1: CodeInformation(code='ADMISSION//CARDIAC', value_prob=0.0, value_mean=None),
         2: CodeInformation(code='ADMISSION//ORTHOPEDIC', value_prob=0.0, value_mean=None),
         3: CodeInformation(code='ADMISSION//PULMONARY', value_prob=0.0, value_mean=None),
         4: CodeInformation(code='DISCHARGE', value_prob=0.0, value_mean=None),
         5: CodeInformation(code='EYE_COLOR//BLUE', value_prob=0.0, value_mean=None),
         6: CodeInformation(code='EYE_COLOR//BROWN', value_prob=0.0, value_mean=None),
         7: CodeInformation(code='EYE_COLOR//HAZEL', value_prob=0.0, value_mean=None),
         8: CodeInformation(code='HEIGHT//value_[156.4856,160.39531)', value_prob=1.0, value_mean=156.4...),
         9: CodeInformation(code='HEIGHT//value_[160.39531,164.68689)', value_prob=1.0, value_mean=160.3...),
         10: CodeInformation(code='HEIGHT//value_[164.68689,175.27112)', value_prob=1.0, value_mean=164.6...),
         11: CodeInformation(code='HEIGHT//value_[175.27112,inf)', value_prob=1.0, value_mean=175.2...),
         12: CodeInformation(code='HR//value_[-inf,102.6)', value_prob=1.0, value_mean=86.0),
         13: CodeInformation(code='HR//value_[102.6,105.1)', value_prob=1.0, value_mean=102.5999984741211),
         14: CodeInformation(code='HR//value_[105.1,107.5)', value_prob=1.0, value_mean=105.0999984741211),
         15: CodeInformation(code='HR//value_[107.5,107.7)', value_prob=1.0, value_mean=107.5),
         16: CodeInformation(code='HR//value_[107.7,112.5)', value_prob=1.0, value_mean=108.3499984741211),
         17: CodeInformation(code='HR//value_[112.5,112.6)', value_prob=1.0, value_mean=112.5),
         18: CodeInformation(code='HR//value_[112.6,113.4)', value_prob=1.0, value_mean=112.5999984741211),
         19: CodeInformation(code='HR//value_[113.4,114.1)', value_prob=1.0, value_mean=113.4000015258789),
         20: CodeInformation(code='HR//value_[114.1,119.8)', value_prob=1.0, value_mean=114.0999984741211),
         21: CodeInformation(code='HR//value_[119.8,inf)', value_prob=1.0, value_mean=145.0),
         22: CodeInformation(code='MEDS_BIRTH', value_prob=0.0, value_mean=None),
         23: CodeInformation(code='TEMP//value_[-inf,95.8)', value_prob=1.0, value_mean=95.5),
         24: CodeInformation(code='TEMP//value_[100.0,100.1)', value_prob=1.0, value_mean=100.0),
         25: CodeInformation(code='TEMP//value_[100.1,inf)', value_prob=1.0, value_mean=100.25),
         26: CodeInformation(code='TEMP//value_[95.8,96.0)', value_prob=1.0, value_mean=95.80000305175781),
         27: CodeInformation(code='TEMP//value_[96.0,96.2)', value_prob=1.0, value_mean=96.0),
         28: CodeInformation(code='TEMP//value_[96.2,97.8)', value_prob=1.0, value_mean=96.19999694824219),
         29: CodeInformation(code='TEMP//value_[97.8,99.9)', value_prob=1.0, value_mean=98.80000305175781),
         30: CodeInformation(code='TEMP//value_[99.9,100.0)', value_prob=1.0, value_mean=99.9000015258789),
         31: CodeInformation(code='TIMELINE//DELTA//years//value_...', value_prob=1.0, value_mean=3...e-06),
         32: CodeInformation(code='TIMELINE//DELTA//years//value_...', value_prob=1.0, value_mean=1...e-05),
         33: CodeInformation(code='TIMELINE//DELTA//years//value_...', value_prob=1.0, value_mean=4...e-05),
         34: CodeInformation(code='TIMELINE//DELTA//years//value_...', value_prob=1.0, value_mean=6...e-05),
         35: CodeInformation(code='TIMELINE//DELTA//years//value_...', value_prob=1.0, value_mean=0...),
         36: CodeInformation(code='TIMELINE//DELTA//years//value_...', value_prob=1.0, value_mean=31...),
         37: CodeInformation(code='TIMELINE//END', value_prob=0.0, value_mean=None),
         38: CodeInformation(code='TIMELINE//START', value_prob=0.0, value_mean=None)}
    """
    code_information = {}

    columns = ["code", "code/vocab_index", "code/n_occurrences", "values/n_occurrences", "values/sum"]
    code_metadata_df = pl.read_parquet(dataset.config.code_metadata_fp, columns=columns, use_pyarrow=True)

    for row in code_metadata_df.to_dicts():
        has_value_prob = row["values/n_occurrences"] / row["code/n_occurrences"]
        value_mean = (row["values/sum"] / row["values/n_occurrences"]) if has_value_prob else None
        code_information[row["code/vocab_index"]] = CodeInformation(
            code=row["code"],
            value_prob=has_value_prob,
            value_mean=value_mean,
        )

    return code_information


def format_trajectory_batch(
    schema_chunk: pl.DataFrame,
    generated_code_indices: torch.LongTensor,
    code_information: dict[int, CodeInformation],
) -> pl.DataFrame:
    """Formats a single batch of generated outputs into a MEDS-like dataframe format.

    Args:
        schema_chunk: The chunk of the dataset's schema dataframe corresponding to this generated batch. This
            dataframe must have the following columns:
              - `"subject_id"`: The subject ID of the patient.
              - `"prediction_time"`: The time after which no data can be ingested for this prediction. This
                is, implicitly, assumed to be the start time of the generated window. TODO(mmd): This
                assumption may not always be valid, and we don't have an exposed endpoint for the true time of
                the last event in the input!
              - `"task_sample_id"`: The task sample ID of the patient. This is a unique identifier for the
                task sample, which is useful as there may be different task samples for the same patient, and
                we don't wish our generated trajectories to intersect.
              - `"window_last_observed"`: The last time of an event observed in the input data for this
                subject sample.
        generated_code_indices: The generated codes for this batch.
        code_information: The code information mapping from code indices to their string representations and
            numeric value means.

    Returns:
        A polars dataframe containing this batch of generated trajectory data in a MEDS-like format.

    Examples:
        >>> schema_df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3, 1, 4],
        ...     "prediction_time": [
        ...         datetime(1993, 1, 1),
        ...         datetime(2000, 1, 2),
        ...         datetime(1973, 1, 3),
        ...         datetime(2002, 10, 12),
        ...         datetime(2002, 10, 12),
        ...     ],
        ...     "window_last_observed": [
        ...         datetime(1993, 1, 1),
        ...         datetime(2000, 1, 1, 22, 13),
        ...         datetime(1973, 1, 2),
        ...         datetime(2002, 10, 12),
        ...         datetime(2002, 10, 11, 23, 59),
        ...     ],
        ... })
        >>> generated_code_indices = torch.LongTensor([
        ...     [2, 4, 5, 4, 3, 4, 5, 5, 1, 6],
        ...     [1, 7, 9, 0, 0, 0, 0, 0, 0, 0],
        ...     [4, 7, 3, 6, 1, 9, 0, 0, 0, 0],
        ...     [6, 1, 8, 9, 0, 0, 0, 0, 0, 0],
        ...     [9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ... ])
        >>> code_information = {
        ...     1: CodeInformation(code='TIMELINE//DELTA//years//value_A', value_prob=1.0, value_mean=1.0),
        ...     2: CodeInformation(code='TIMELINE//DELTA//years//value_A', value_prob=1.0, value_mean=0.1),
        ...     3: CodeInformation(code='TIMELINE//DELTA//years//value_A', value_prob=1.0, value_mean=0.001),
        ...     4: CodeInformation(code='HR', value_prob=0.0, value_mean=None),
        ...     5: CodeInformation(code='TEMP//A', value_prob=1.0, value_mean=99.0),
        ...     6: CodeInformation(code='TEMP//B', value_prob=1.0, value_mean=101.0),
        ...     7: CodeInformation(code='TEMP//C', value_prob=1.0, value_mean=96.8),
        ...     8: CodeInformation(code='DX//1', value_prob=0.0, value_mean=None),
        ...     9: CodeInformation(code='TIMELINE//END', value_prob=0.0, value_mean=None)
        ... }

    With this setup, we are generating the following strings of events for these patients:

        >>> schema_df
        shape: (5, 3)
        ┌────────────┬─────────────────────┬──────────────────────┐
        │ subject_id ┆ prediction_time     ┆ window_last_observed │
        │ ---        ┆ ---                 ┆ ---                  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]         │
        ╞════════════╪═════════════════════╪══════════════════════╡
        │ 1          ┆ 1993-01-01 00:00:00 ┆ 1993-01-01 00:00:00  │
        │ 2          ┆ 2000-01-02 00:00:00 ┆ 2000-01-01 22:13:00  │
        │ 3          ┆ 1973-01-03 00:00:00 ┆ 1973-01-02 00:00:00  │
        │ 1          ┆ 2002-10-12 00:00:00 ┆ 2002-10-12 00:00:00  │
        │ 4          ┆ 2002-10-12 00:00:00 ┆ 2002-10-11 23:59:00  │
        └────────────┴─────────────────────┴──────────────────────┘

    For task sample 1 (starting at 1993-01-01), we generate:
      - (2) A timeline delta of 0.1 years, which is 35.6 days. Event time: ~1993-02-06, 12:30
      - (4) A HR event, with no numeric value.
      - (5) A TEMP//A event, with a numeric value of 99.0.
      - (4) A HR event, with no numeric value.
      - (3) A timeline delta of 0.001 years, which is 0.0365 days (8.7 hours). Event time: ~1993-02-06, 21:20
      - (4) A HR event, with no numeric value.
      - (5) A TEMP//A event, with a numeric value of 99.0.
      - (5) A TEMP//A event, with a numeric value of 99.0.
      - (1) A timeline delta of 1.0 year. Event time: ~1994-02-06, 21:20
      - (6) A TEMP//B event, with a numeric value of 101.0.
    For task sample 2 (starting at 2000-01-01, 22:13), we generate:
      - (1) A timeline delta of 1.0 year. Event time: ~2001-01-01
      - (7) A TEMP//C event, with a numeric value of 96.8.
      - (9) A TIMELINE//END event, with no numeric value.
    For task sample 3 (starting at 1973-01-02), we generate the following:
      - (4) A HR event, with no numeric value.
      - (7) A TEMP//C event, with a numeric value of 96.8.
      - (3) A timeline delta of 0.001 years, which is 0.0365 days (8.7 hours). Event time: ~1973-01-02, 08:45
      - (6) A TEMP//B event, with a numeric value of 101.0.
      - (1) A timeline delta of 1.0 year. Event time: ~1974-01-02, 08:45
      - (9) A TIMELINE//END event, with no numeric value.
    For task sample 4 (starting at 2002-10-12), we generate:
      - (6) A TEMP//B event, with a numeric value of 101.0.
      - (1) A timeline delta of 1.0 year. Event time: ~2003-10-12
      - (8) A DX//1 event, with no numeric value.
      - (9) A TIMELINE//END event, with no numeric value.
    For task sample 5 (starting at 2002-10-11, 23:59), we generate:
      - (9) A TIMELINE//END event, with no numeric value.

        >>> _ = pl.Config().set_tbl_rows(-1)
        >>> format_trajectory_batch(schema_df, generated_code_indices, code_information)
        shape: (24, 5)
        ┌────────────┬───────────────────────┬─────────────────────┬───────────────────────┬───────────────┐
        │ subject_id ┆ time                  ┆ prediction_time     ┆ code                  ┆ numeric_value │
        │ ---        ┆ ---                   ┆ ---                 ┆ ---                   ┆ ---           │
        │ i64        ┆ datetime[μs]          ┆ datetime[μs]        ┆ str                   ┆ f32           │
        ╞════════════╪═══════════════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
        │ 1          ┆ 1993-02-06            ┆ 1993-01-01 00:00:00 ┆ TIMELINE//DELTA//year ┆ 0.1           │
        │            ┆ 12:34:52.608          ┆                     ┆ s//value_…            ┆               │
        │ 1          ┆ 1993-02-06            ┆ 1993-01-01 00:00:00 ┆ HR                    ┆ null          │
        │            ┆ 12:34:52.608          ┆                     ┆                       ┆               │
        │ 1          ┆ 1993-02-06            ┆ 1993-01-01 00:00:00 ┆ TEMP//A               ┆ 99.0          │
        │            ┆ 12:34:52.608          ┆                     ┆                       ┆               │
        │ 1          ┆ 1993-02-06            ┆ 1993-01-01 00:00:00 ┆ HR                    ┆ null          │
        │            ┆ 12:34:52.608          ┆                     ┆                       ┆               │
        │ 1          ┆ 1993-02-06            ┆ 1993-01-01 00:00:00 ┆ TIMELINE//DELTA//year ┆ 0.001         │
        │            ┆ 21:20:49.534080       ┆                     ┆ s//value_…            ┆               │
        │ 1          ┆ 1993-02-06            ┆ 1993-01-01 00:00:00 ┆ HR                    ┆ null          │
        │            ┆ 21:20:49.534080       ┆                     ┆                       ┆               │
        │ 1          ┆ 1993-02-06            ┆ 1993-01-01 00:00:00 ┆ TEMP//A               ┆ 99.0          │
        │            ┆ 21:20:49.534080       ┆                     ┆                       ┆               │
        │ 1          ┆ 1993-02-06            ┆ 1993-01-01 00:00:00 ┆ TEMP//A               ┆ 99.0          │
        │            ┆ 21:20:49.534080       ┆                     ┆                       ┆               │
        │ 1          ┆ 1994-02-07            ┆ 1993-01-01 00:00:00 ┆ TIMELINE//DELTA//year ┆ 1.0           │
        │            ┆ 03:09:35.614080       ┆                     ┆ s//value_…            ┆               │
        │ 1          ┆ 1994-02-07            ┆ 1993-01-01 00:00:00 ┆ TEMP//B               ┆ 101.0         │
        │            ┆ 03:09:35.614080       ┆                     ┆                       ┆               │
        │ 2          ┆ 2001-01-01            ┆ 2000-01-02 00:00:00 ┆ TIMELINE//DELTA//year ┆ 1.0           │
        │            ┆ 04:01:46.080          ┆                     ┆ s//value_…            ┆               │
        │ 2          ┆ 2001-01-01            ┆ 2000-01-02 00:00:00 ┆ TEMP//C               ┆ 96.800003     │
        │            ┆ 04:01:46.080          ┆                     ┆                       ┆               │
        │ 2          ┆ 2001-01-01            ┆ 2000-01-02 00:00:00 ┆ TIMELINE//END         ┆ null          │
        │            ┆ 04:01:46.080          ┆                     ┆                       ┆               │
        │ 3          ┆ 1973-01-02 00:00:00   ┆ 1973-01-03 00:00:00 ┆ HR                    ┆ null          │
        │ 3          ┆ 1973-01-02 00:00:00   ┆ 1973-01-03 00:00:00 ┆ TEMP//C               ┆ 96.800003     │
        │ 3          ┆ 1973-01-02            ┆ 1973-01-03 00:00:00 ┆ TIMELINE//DELTA//year ┆ 0.001         │
        │            ┆ 08:45:56.926080       ┆                     ┆ s//value_…            ┆               │
        │ 3          ┆ 1973-01-02            ┆ 1973-01-03 00:00:00 ┆ TEMP//B               ┆ 101.0         │
        │            ┆ 08:45:56.926080       ┆                     ┆                       ┆               │
        │ 3          ┆ 1974-01-02            ┆ 1973-01-03 00:00:00 ┆ TIMELINE//DELTA//year ┆ 1.0           │
        │            ┆ 14:34:43.006080       ┆                     ┆ s//value_…            ┆               │
        │ 3          ┆ 1974-01-02            ┆ 1973-01-03 00:00:00 ┆ TIMELINE//END         ┆ null          │
        │            ┆ 14:34:43.006080       ┆                     ┆                       ┆               │
        │ 1          ┆ 2002-10-12 00:00:00   ┆ 2002-10-12 00:00:00 ┆ TEMP//B               ┆ 101.0         │
        │ 1          ┆ 2003-10-12            ┆ 2002-10-12 00:00:00 ┆ TIMELINE//DELTA//year ┆ 1.0           │
        │            ┆ 05:48:46.080          ┆                     ┆ s//value_…            ┆               │
        │ 1          ┆ 2003-10-12            ┆ 2002-10-12 00:00:00 ┆ DX//1                 ┆ null          │
        │            ┆ 05:48:46.080          ┆                     ┆                       ┆               │
        │ 1          ┆ 2003-10-12            ┆ 2002-10-12 00:00:00 ┆ TIMELINE//END         ┆ null          │
        │            ┆ 05:48:46.080          ┆                     ┆                       ┆               │
        │ 4          ┆ 2002-10-11 23:59:00   ┆ 2002-10-12 00:00:00 ┆ TIMELINE//END         ┆ null          │
        └────────────┴───────────────────────┴─────────────────────┴───────────────────────┴───────────────┘

    This function is robust to the unit string used for the timeline delta tokens, provided it conforms to the
    set recognized in
    [`MEDS_transforms.stages.add_time_derived_measurements.utils.normalize_time_unit`](https://meds-transforms.readthedocs.io/en/latest/api/MEDS_transforms/stages/add_time_derived_measurements/utils/#MEDS_transforms.stages.add_time_derived_measurements.utils.normalize_time_unit).
    However, note that as time deltas are based on aggregate units (not calendar units), they won't
    necessarily universally correspond to true calendar months or years.

        >>> code_information = {
        ...     1: CodeInformation(code='TIMELINE//DELTA//s//A', value_prob=1.0, value_mean=1.0),
        ...     2: CodeInformation(code='TIMELINE//DELTA//days//A', value_prob=1.0, value_mean=1.0),
        ...     3: CodeInformation(code='TIMELINE//DELTA//wks//B', value_prob=1.0, value_mean=1.0),
        ...     4: CodeInformation(code='TIMELINE//DELTA//mos//C', value_prob=1.0, value_mean=1.0),
        ...     5: CodeInformation(code='TIMELINE//DELTA//yrs//C', value_prob=1.0, value_mean=1.0),
        ... }
        >>> schema_df = pl.DataFrame({
        ...     "subject_id": [1],
        ...     "prediction_time": [datetime(1993, 1, 1)],
        ...     "window_last_observed": [datetime(1993, 1, 1)],
        ... })
        >>> generated_code_indices = torch.LongTensor([[1, 2, 3, 4, 5]])
        >>> format_trajectory_batch(schema_df, generated_code_indices, code_information)
        shape: (5, 5)
        ┌────────────┬───────────────────────┬─────────────────────┬───────────────────────┬───────────────┐
        │ subject_id ┆ time                  ┆ prediction_time     ┆ code                  ┆ numeric_value │
        │ ---        ┆ ---                   ┆ ---                 ┆ ---                   ┆ ---           │
        │ i64        ┆ datetime[μs]          ┆ datetime[μs]        ┆ str                   ┆ f32           │
        ╞════════════╪═══════════════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
        │ 1          ┆ 1993-01-01 00:00:01   ┆ 1993-01-01 00:00:00 ┆ TIMELINE//DELTA//s//A ┆ 1.0           │
        │ 1          ┆ 1993-01-02 00:00:01   ┆ 1993-01-01 00:00:00 ┆ TIMELINE//DELTA//days ┆ 1.0           │
        │            ┆                       ┆                     ┆ //A                   ┆               │
        │ 1          ┆ 1993-01-09 00:00:01   ┆ 1993-01-01 00:00:00 ┆ TIMELINE//DELTA//wks/ ┆ 1.0           │
        │            ┆                       ┆                     ┆ /B                    ┆               │
        │ 1          ┆ 1993-02-08 10:29:07   ┆ 1993-01-01 00:00:00 ┆ TIMELINE//DELTA//mos/ ┆ 1.0           │
        │            ┆                       ┆                     ┆ /C                    ┆               │
        │ 1          ┆ 1994-02-08            ┆ 1993-01-01 00:00:00 ┆ TIMELINE//DELTA//yrs/ ┆ 1.0           │
        │            ┆ 16:17:53.080          ┆                     ┆ /C                    ┆               │
        └────────────┴───────────────────────┴─────────────────────┴───────────────────────┴───────────────┘

    Note than an error is raised if the schema chunk does not have the same batch size as the generated
    tokens:

        >>> format_trajectory_batch(schema_df[:0], generated_code_indices, code_information)
        Traceback (most recent call last):
            ...
        ValueError: Batch size 1 does not match schema chunk size 0
    """

    batch_size = generated_code_indices.shape[0]
    if batch_size != schema_chunk.shape[0]:
        raise ValueError(f"Batch size {batch_size} does not match schema chunk size {schema_chunk.shape[0]}")

    rows = []
    for i in range(batch_size):
        subject_id = schema_chunk.select(DataSchema.subject_id_name)[i].item()
        prediction_time = schema_chunk.select(LabelSchema.prediction_time_name)[i].item()
        time = schema_chunk.select(MEDSPytorchDataset.LAST_TIME)[i].item()

        for code_idx in generated_code_indices[i]:
            if code_idx == 0:
                continue

            code_info = code_information[code_idx.item()]
            code = code_info.code
            value_mean = code_info.value_mean

            if code.startswith(TIMELINE_DELTA_TOKEN):
                unit = code.split("//")[-2]
                _, seconds_in_unit = normalize_time_unit(unit)
                seconds = value_mean * seconds_in_unit
                time += timedelta(seconds=seconds)

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


def format_trajectories(
    dataset: MEDSPytorchDataset,
    generated_outputs: list[torch.LongTensor],
) -> pl.DataFrame:
    """Transfomrs the generated outputs into a MEDS-like dataframe format of continued trajectories.

    Args:
        dataset: The dataset used for generation.
        generated_outputs: The generated outputs. This is formatted as a list of generated samples that should
            be of the same length as the dataframe.

    Returns:
        A polars dataframe containing the generated trajectories in a MEDS-like format.

    Raises:
        ValueError: If the passed dataset does not yield code info with values strictly either always
            occurring or never occurring.

    Examples:
        >>> generated_code_indices = [
        ...     torch.LongTensor([[31, 4, 14, 4, 3, 4, 14, 14, 33, 15], [32, 16, 33, 15, 1, 37, 0, 0, 0, 0]]),
        ...     torch.LongTensor([[36, 16, 37], [37, 0, 0]]),
        ... ]
        >>> _ = pl.Config().set_tbl_rows(-1)
        >>> format_trajectories(pytorch_dataset_with_task, generated_code_indices)
        shape: (20, 5)
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
        │ 239684     ┆ 2042-03-22            ┆ 2010-05-11 19:00:00 ┆ TIMELINE//DELTA//year ┆ 31.861664     │
        │            ┆ 00:20:07.901777       ┆                     ┆ s//value_…            ┆               │
        │ 239684     ┆ 2042-03-22            ┆ 2010-05-11 19:00:00 ┆ HR//value_[107.7,112. ┆ 108.349998    │
        │            ┆ 00:20:07.901777       ┆                     ┆ 5)                    ┆               │
        │ 239684     ┆ 2042-03-22            ┆ 2010-05-11 19:00:00 ┆ TIMELINE//END         ┆ null          │
        │            ┆ 00:20:07.901777       ┆                     ┆                       ┆               │
        │ 1195293    ┆ 2010-06-20 19:25:32   ┆ 2010-06-20 19:30:00 ┆ TIMELINE//END         ┆ null          │
        └────────────┴───────────────────────┴─────────────────────┴───────────────────────┴───────────────┘

    If the dataset yields invalid code information, an error will be thrown:

        >>> with patch("MEDS_EIC_AR.generation.format_trajectories.get_code_information") as mock:
        ...     mock.return_value = {1: CodeInformation(code='HR', value_prob=0.5, value_mean=106.0)}
        ...     format_trajectories("fake dataset", generated_code_indices)
        Traceback (most recent call last):
          ...
        ValueError: Code HR has a value probability of 0.5, which is not 0.0 or 1.0. This is not supported.
    """

    code_information = get_code_information(dataset)

    for code_info in code_information.values():
        if code_info.value_prob not in {0.0, 1.0}:
            raise ValueError(
                f"Code {code_info.code} has a value probability of {code_info.value_prob}, "
                "which is not 0.0 or 1.0. This is not supported."
            )

    output_schema = dataset.schema_df.select(
        DataSchema.subject_id_name, LabelSchema.prediction_time_name, dataset.LAST_TIME
    )

    batches_as_df = []
    st_i = 0
    for generated_codes in generated_outputs:
        batch_size = generated_codes.shape[0]

        # Get the schema chunk for this batch
        schema_chunk = output_schema.slice(st_i, batch_size).clone()
        st_i += batch_size

        # Format the generated codes into a MEDS-like dataframe
        batch_df = format_trajectory_batch(schema_chunk, generated_codes, code_information)
        batches_as_df.append(batch_df)

    return pl.concat(batches_as_df)
