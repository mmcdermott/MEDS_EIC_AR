import polars as pl
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import DictConfig, OmegaConf

_ROLLING_ALLOWED_KEYS = {"max_new_tokens", "rolling_context_size"}


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
