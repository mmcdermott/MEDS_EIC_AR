import polars as pl
from meds_torchdata import MEDSPytorchDataset
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import DictConfig, OmegaConf

_ROLLING_ALLOWED_KEYS = {"max_new_tokens", "rolling_context_size"}


def get_code_information(dataset: MEDSPytorchDataset) -> pl.DataFrame:
    """Return a DataFrame mapping code indices to their code strings, value probabilities, and value means.

    Reads ``dataset.config.code_metadata_fp`` and produces a polars DataFrame with one row per
    vocabulary entry and columns:

    - ``code_idx`` (Int64): the vocabulary index used in generated tokens.
    - ``code`` (Utf8): the MEDS code string (e.g. ``TIMELINE//END``, ``HR//value_[102.6,105.1)``).
    - ``value_prob`` (Float64): fraction of occurrences of this code that carry a numeric value.
      By construction of the preprocessing pipeline (``quantile_binning`` + ``bin_numeric_values``
      stages) this is always 0.0 (never has a value) or 1.0 (always has a value) — the
      trajectory formatter assumes this invariant without validating it at runtime.
    - ``value_mean`` (Float64, nullable): the mean numeric value across observed occurrences
      (null when ``value_prob == 0``).

    Returning a DataFrame (rather than ``dict[int, CodeInformation]``) lets
    :func:`MEDS_EIC_AR.generation.finalize.format_trajectories` do a single polars join against
    the exploded generated-token stream instead of per-row Python dict lookups.

    Args:
        dataset: The dataset used for generation.

    Returns:
        A polars DataFrame with columns ``code_idx``, ``code``, ``value_prob``, ``value_mean``.

    Examples:
        >>> _ = pl.Config().set_tbl_rows(-1)
        >>> get_code_information(pytorch_dataset).sort("code_idx")
        shape: (38, 4)
        ┌──────────┬─────────────────────────────────┬────────────┬────────────┐
        │ code_idx ┆ code                            ┆ value_prob ┆ value_mean │
        │ ---      ┆ ---                             ┆ ---        ┆ ---        │
        │ i64      ┆ str                             ┆ f64        ┆ f32        │
        ╞══════════╪═════════════════════════════════╪════════════╪════════════╡
        │ 1        ┆ ADMISSION//CARDIAC              ┆ 0.0        ┆ null       │
        │ 2        ┆ ADMISSION//ORTHOPEDIC           ┆ 0.0        ┆ null       │
        │ 3        ┆ ADMISSION//PULMONARY            ┆ 0.0        ┆ null       │
        │ 4        ┆ DISCHARGE                       ┆ 0.0        ┆ null       │
        │ 5        ┆ EYE_COLOR//BLUE                 ┆ 0.0        ┆ null       │
        │ 6        ┆ EYE_COLOR//BROWN                ┆ 0.0        ┆ null       │
        │ 7        ┆ EYE_COLOR//HAZEL                ┆ 0.0        ┆ null       │
        │ 8        ┆ HEIGHT//value_[156.4856,160.39… ┆ 1.0        ┆ 156.485596 │
        │ 9        ┆ HEIGHT//value_[160.39531,164.6… ┆ 1.0        ┆ 160.395309 │
        │ 10       ┆ HEIGHT//value_[164.68689,175.2… ┆ 1.0        ┆ 164.68689  │
        │ 11       ┆ HEIGHT//value_[175.27112,inf)   ┆ 1.0        ┆ 175.271118 │
        │ 12       ┆ HR//value_[-inf,102.6)          ┆ 1.0        ┆ 86.0       │
        │ 13       ┆ HR//value_[102.6,105.1)         ┆ 1.0        ┆ 102.599998 │
        │ 14       ┆ HR//value_[105.1,107.5)         ┆ 1.0        ┆ 105.099998 │
        │ 15       ┆ HR//value_[107.5,107.7)         ┆ 1.0        ┆ 107.5      │
        │ 16       ┆ HR//value_[107.7,112.5)         ┆ 1.0        ┆ 108.349998 │
        │ 17       ┆ HR//value_[112.5,112.6)         ┆ 1.0        ┆ 112.5      │
        │ 18       ┆ HR//value_[112.6,113.4)         ┆ 1.0        ┆ 112.599998 │
        │ 19       ┆ HR//value_[113.4,114.1)         ┆ 1.0        ┆ 113.400002 │
        │ 20       ┆ HR//value_[114.1,119.8)         ┆ 1.0        ┆ 114.099998 │
        │ 21       ┆ HR//value_[119.8,inf)           ┆ 1.0        ┆ 145.0      │
        │ 22       ┆ MEDS_BIRTH                      ┆ 0.0        ┆ null       │
        │ 23       ┆ TEMP//value_[-inf,95.8)         ┆ 1.0        ┆ 95.5       │
        │ 24       ┆ TEMP//value_[100.0,100.1)       ┆ 1.0        ┆ 100.0      │
        │ 25       ┆ TEMP//value_[100.1,inf)         ┆ 1.0        ┆ 100.25     │
        │ 26       ┆ TEMP//value_[95.8,96.0)         ┆ 1.0        ┆ 95.800003  │
        │ 27       ┆ TEMP//value_[96.0,96.2)         ┆ 1.0        ┆ 96.0       │
        │ 28       ┆ TEMP//value_[96.2,97.8)         ┆ 1.0        ┆ 96.199997  │
        │ 29       ┆ TEMP//value_[97.8,99.9)         ┆ 1.0        ┆ 98.800003  │
        │ 30       ┆ TEMP//value_[99.9,100.0)        ┆ 1.0        ┆ 99.900002  │
        │ 31       ┆ TIMELINE//DELTA//years//value_… ┆ 1.0        ┆ 0.000003   │
        │ 32       ┆ TIMELINE//DELTA//years//value_… ┆ 1.0        ┆ 0.000015   │
        │ 33       ┆ TIMELINE//DELTA//years//value_… ┆ 1.0        ┆ 0.00004    │
        │ 34       ┆ TIMELINE//DELTA//years//value_… ┆ 1.0        ┆ 0.000065   │
        │ 35       ┆ TIMELINE//DELTA//years//value_… ┆ 1.0        ┆ 0.000198   │
        │ 36       ┆ TIMELINE//DELTA//years//value_… ┆ 1.0        ┆ 31.861664  │
        │ 37       ┆ TIMELINE//END                   ┆ 0.0        ┆ null       │
        │ 38       ┆ TIMELINE//START                 ┆ 0.0        ┆ null       │
        └──────────┴─────────────────────────────────┴────────────┴────────────┘
    """
    columns = ["code", "code/vocab_index", "code/n_occurrences", "values/n_occurrences", "values/sum"]
    metadata = pl.read_parquet(dataset.config.code_metadata_fp, columns=columns, use_pyarrow=True)

    # ``value_prob`` = ``values/n_occurrences / code/n_occurrences`` — fraction of observations
    # that carried a value. ``value_mean`` is ``values/sum / values/n_occurrences`` when any
    # value-carrying observations exist, else null.
    return metadata.select(
        pl.col("code/vocab_index").cast(pl.Int64).alias("code_idx"),
        pl.col("code"),
        (pl.col("values/n_occurrences") / pl.col("code/n_occurrences")).alias("value_prob"),
        pl.when(pl.col("values/n_occurrences") > 0)
        .then(pl.col("values/sum") / pl.col("values/n_occurrences"))
        .otherwise(None)
        .alias("value_mean"),
    )


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
