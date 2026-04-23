from typing import NamedTuple

import polars as pl
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import DictConfig, OmegaConf

_ROLLING_ALLOWED_KEYS = {"max_new_tokens", "rolling_context_size"}


class CodeInformation(NamedTuple):
    """Per-vocab-index metadata consumed by the trajectory formatter.

    - ``code``: the MEDS code string (e.g. ``TIMELINE//END``, ``HR//value_[102.6,105.1)``).
    - ``value_prob``: fraction of occurrences of this code that carry a numeric value. In
      practice this is always ``0.0`` (never has a value) or ``1.0`` (always has a value) by
      construction of the preprocessing pipeline — the ``quantile_binning`` +
      ``bin_numeric_values`` stages produce either always-value-carrying codes
      (``X//value_[lo,hi)`` bins) or sentinel never-value-carrying codes (``DISCHARGE``,
      ``TIMELINE//END``, etc.). The trajectory formatter assumes this invariant without
      validating it at runtime.
    - ``value_mean``: the mean numeric value across observed occurrences (``None`` when
      ``value_prob == 0``).
    """

    code: str
    value_prob: float
    value_mean: float | None


def get_code_information(dataset: MEDSPytorchDataset) -> dict[int, CodeInformation]:
    """Return a dictionary mapping code indices to their code strings and numeric-value means.

    Reads ``dataset.config.code_metadata_fp`` and builds a ``{vocab_index: CodeInformation}`` dict.
    Consumed by :func:`format_trajectories` for the token-index → MEDS-row translation.

    Args:
        dataset: The dataset used for generation.

    Returns:
        A dictionary mapping code indices to their code strings and numeric value means.

    Examples:
        >>> dict(sorted(get_code_information(pytorch_dataset).items()))
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


def validate_rolling_cfg(rolling_cfg: DictConfig | None) -> dict[str, int]:
    """Validate the ``rolling_generation`` Hydra config block and return a cleaned kwargs dict.

    Returns an empty dict if ``rolling_cfg`` is ``None`` or contains only ``None`` values. Otherwise
    returns a dict of positive-int kwargs ready to update ``MEICARModule.generation_kwargs``. Raises
    ``ValueError`` on any structural problem so the generation CLI fails before loading the
    checkpoint rather than surfacing bad config deep inside ``Model._rolling_generate``.

    Examples:
        ``None`` and all-``None`` configs both mean "rolling disabled":

        >>> validate_rolling_cfg(None)
        {}
        >>> validate_rolling_cfg(
        ...     OmegaConf.create({"max_new_tokens": None, "rolling_context_size": None})
        ... )
        {}

        A valid ``max_new_tokens`` alone enables rolling with the default per-chunk window:

        >>> validate_rolling_cfg(OmegaConf.create({"max_new_tokens": 50}))
        {'max_new_tokens': 50}
        >>> validate_rolling_cfg(
        ...     OmegaConf.create({"max_new_tokens": 50, "rolling_context_size": None})
        ... )
        {'max_new_tokens': 50}

        Both keys can be set together:

        >>> validate_rolling_cfg(
        ...     OmegaConf.create({"max_new_tokens": 50, "rolling_context_size": 8})
        ... )
        {'max_new_tokens': 50, 'rolling_context_size': 8}

        Unexpected keys are rejected so typos like ``max_tokens`` or ``rolling_window`` don't get
        silently accepted as no-ops:

        >>> validate_rolling_cfg(OmegaConf.create({"max_tokens": 50}))
        Traceback (most recent call last):
            ...
        ValueError: rolling_generation has unexpected key(s) ['max_tokens']; only ...

        Non-int, bool, and non-positive values are all rejected with an actionable message:

        >>> validate_rolling_cfg(OmegaConf.create({"max_new_tokens": "50"}))
        Traceback (most recent call last):
            ...
        ValueError: rolling_generation.max_new_tokens must be a positive integer when set; got '50'. ...
        >>> validate_rolling_cfg(OmegaConf.create({"max_new_tokens": True}))
        Traceback (most recent call last):
            ...
        ValueError: rolling_generation.max_new_tokens must be a positive integer when set; got True. ...
        >>> validate_rolling_cfg(OmegaConf.create({"max_new_tokens": 0}))
        Traceback (most recent call last):
            ...
        ValueError: rolling_generation.max_new_tokens must be a positive integer when set; got 0. ...
        >>> validate_rolling_cfg(OmegaConf.create({"max_new_tokens": -5}))
        Traceback (most recent call last):
            ...
        ValueError: rolling_generation.max_new_tokens must be a positive integer when set; got -5. ...

        Setting ``rolling_context_size`` without ``max_new_tokens`` would be a silent no-op on the
        legacy single-chunk path — reject it:

        >>> validate_rolling_cfg(OmegaConf.create({"rolling_context_size": 8}))
        Traceback (most recent call last):
            ...
        ValueError: rolling_generation.rolling_context_size is set but ...
    """
    if rolling_cfg is None:
        return {}
    raw = OmegaConf.to_container(rolling_cfg, resolve=True)
    extra = set(raw) - _ROLLING_ALLOWED_KEYS
    if extra:
        raise ValueError(
            f"rolling_generation has unexpected key(s) {sorted(extra)}; only "
            f"{sorted(_ROLLING_ALLOWED_KEYS)} are allowed."
        )
    kwargs = {k: v for k, v in raw.items() if v is not None}
    if not kwargs:
        return {}
    for k, v in kwargs.items():
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(
                f"rolling_generation.{k} must be a positive integer when set; got {v!r}. "
                f"Leave it null to disable rolling generation for {k!r}."
            )
    if "max_new_tokens" not in kwargs:
        raise ValueError(
            "rolling_generation.rolling_context_size is set but "
            "rolling_generation.max_new_tokens is null. `rolling_context_size` only takes effect "
            "on the rolling path, which is enabled by setting `max_new_tokens`. Either set "
            "`max_new_tokens` to a positive integer to enable rolling generation, or leave "
            "`rolling_context_size` null."
        )
    return kwargs


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
