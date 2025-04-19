import polars as pl
from meds_torchdata.config import MEDSTorchDataConfig


def get_timeline_end_token_idx(dataset_config: MEDSTorchDataConfig) -> int:
    """Get the index of the end token in the timeline vocabulary.

    Args:
        dataset (MEDSPytorchDataset): The dataset used for generation.

    Returns:
        int: The index of the end token in the timeline vocabulary.

    Examples:
        >>>
        37
    """
    columns = ["code", "code/vocab_index"]
    code_metadata_df = pl.read_parquet(dataset_config.code_metadata_fp, columns=columns, use_pyarrow=True)

    return code_metadata_df.filter(pl.col("code") == "TIMELINE//END").select("code/vocab_index").item()
