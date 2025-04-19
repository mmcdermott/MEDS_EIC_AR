from datetime import timedelta
from typing import NamedTuple

import polars as pl
import torch
from meds import code_field, numeric_value_field, prediction_time_field, subject_id_field, time_field
from meds_torchdata import MEDSPytorchDataset

TIME_DELTA_UNIT = "year"
TIMELINE_DELTA_TOKEN = "TIMELINE//DELTA//years"
TASK_SAMPLE_ID_COL = "_task_sample_id"


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
         5: CodeInformation(code='DOB', value_prob=0.0, value_mean=None),
         6: CodeInformation(code='EYE_COLOR//BLUE', value_prob=0.0, value_mean=None),
         7: CodeInformation(code='EYE_COLOR//BROWN', value_prob=0.0, value_mean=None),
         8: CodeInformation(code='EYE_COLOR//HAZEL', value_prob=0.0, value_mean=None),
         9: CodeInformation(code='HEIGHT//value_[156.4856,160.39531)', value_prob=1.0, value_mean=156.4...),
         10: CodeInformation(code='HEIGHT//value_[160.39531,164.68689)', value_prob=1.0, value_mean=160.3...),
         11: CodeInformation(code='HEIGHT//value_[164.68689,175.27112)', value_prob=1.0, value_mean=164.6...),
         12: CodeInformation(code='HEIGHT//value_[175.27112,inf)', value_prob=1.0, value_mean=175.2...),
         13: CodeInformation(code='HR//value_[-inf,102.6)', value_prob=1.0, value_mean=86.0),
         14: CodeInformation(code='HR//value_[102.6,105.1)', value_prob=1.0, value_mean=102.5999984741211),
         15: CodeInformation(code='HR//value_[105.1,107.5)', value_prob=1.0, value_mean=105.0999984741211),
         16: CodeInformation(code='HR//value_[107.5,107.7)', value_prob=1.0, value_mean=107.5),
         17: CodeInformation(code='HR//value_[107.7,112.5)', value_prob=1.0, value_mean=108.3499984741211),
         18: CodeInformation(code='HR//value_[112.5,112.6)', value_prob=1.0, value_mean=112.5),
         19: CodeInformation(code='HR//value_[112.6,113.4)', value_prob=1.0, value_mean=112.5999984741211),
         20: CodeInformation(code='HR//value_[113.4,114.1)', value_prob=1.0, value_mean=113.4000015258789),
         21: CodeInformation(code='HR//value_[114.1,119.8)', value_prob=1.0, value_mean=114.0999984741211),
         22: CodeInformation(code='HR//value_[119.8,inf)', value_prob=1.0, value_mean=145.0),
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
        schema_chunk: The chunk of the dataset's schema dataframe corresponding to this generated batch.
        generated_code_indices: The generated codes for this batch.
        code_information: The code information mapping from code indices to their string representations and
            numeric value means.

    Returns:
        A polars dataframe containing this batch of generated trajectory data in a MEDS-like format.
    """

    batch_size = generated_code_indices.shape[0]
    if batch_size != schema_chunk.shape[0]:
        raise ValueError(f"Batch size {batch_size} does not match schema chunk size {schema_chunk.shape[0]}")

    rows = []
    for i in range(batch_size):
        subject_id = schema_chunk.select(subject_id_field)[i]
        time = schema_chunk.select(time_field)[i]
        task_sample_id = schema_chunk.select(TASK_SAMPLE_ID_COL)[i]

        for code_idx in generated_code_indices[i]:
            if code_idx == 0:
                continue

            code_info = code_information[code_idx.item()]
            code = code_info.code
            value_mean = code_info.value_mean

            if code.startswith(TIMELINE_DELTA_TOKEN):
                time += timedelta(**{TIME_DELTA_UNIT: value_mean})

            rows.append(
                {
                    TASK_SAMPLE_ID_COL: task_sample_id,
                    subject_id_field: subject_id,
                    time_field: time,
                    code_field: code,
                    numeric_value_field: value_mean,
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            TASK_SAMPLE_ID_COL: pl.Int64,
            subject_id_field: pl.Int64,
            time_field: pl.Datetime,
            code_field: pl.Utf8,
            numeric_value_field: pl.Float32,
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
    """

    code_information = get_code_information(dataset)

    for code_info in code_information.values():
        if code_info.value_prob not in {0.0, 1.0}:
            raise ValueError(
                f"Code {code_info.code} has a value probability of {code_info.value_prob}, "
                "which is not 0.0 or 1.0. This is not supported."
            )

    output_schema = (
        dataset.schema_df.select(subject_id_field, pl.col(prediction_time_field).alias(time_field)).clone()
    ).with_row_index(TASK_SAMPLE_ID_COL)

    batches_as_df = []
    st_i = 0
    for generated_codes in generated_outputs:
        batch_size = generated_codes.shape[0]

        # Get the schema chunk for this batch
        schema_chunk = output_schema.slice(st_i, batch_size)
        st_i += batch_size

        # Format the generated codes into a MEDS-like dataframe
        batch_df = format_trajectory_batch(schema_chunk, generated_codes, code_information)
        batches_as_df.append(batch_df)

    return pl.concat(batches_as_df)
