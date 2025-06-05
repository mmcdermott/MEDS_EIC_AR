import logging
import multiprocessing
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import torch
from lightning.pytorch.loggers import Logger
from MEDS_transforms.configs.utils import OmegaConfResolver
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def is_mlflow_logger(logger: Logger) -> bool:
    """This function checks if a pytorch lightning logger is an MLFlow logger.

    It is protected against the case that mlflow is not installed.
    """

    try:
        from lightning.pytorch.loggers import MLFlowLogger

        return isinstance(logger, MLFlowLogger)
    except ImportError:
        return False


def is_wandb_logger(logger: Logger) -> bool:
    """Check whether a Lightning logger is a WandB logger.

    The import of :class:`~lightning.pytorch.loggers.WandbLogger` may fail if
    the optional ``wandb`` dependency is not installed. This helper safely
    returns ``False`` in that situation.
    """

    try:
        from lightning.pytorch.loggers import WandbLogger

        return isinstance(logger, WandbLogger)
    except ImportError:
        return False


def hash_based_seed(seed: int | None, split: str, sample: int) -> int:
    """Generates a hash-based seed for reproducibility.

    This function generates a hash-based seed using the provided seed, split, and sample values. It is
    designed to be used in conjunction with OmegaConf for configuration management.

    Args:
        seed: The original seed value. THIS WILL NOT OVERWRITE THE OUTPUT. Rather, this just ensures the
            sequence of seeds chosen can be deterministically updated by changing a base parameter.
        split: The split identifier.
        sample: The sample index.

    Returns:
        A hash-based seed value.

    Examples:
        >>> hash_based_seed(42, "train", 0)
        1508872876
        >>> hash_based_seed(None, "held_out", 1)
        3132876237
    """

    hash_str = f"{seed}_{split}_{sample}"
    return int(sha256(hash_str.encode()).hexdigest(), 16) % (2**32 - 1)


@OmegaConfResolver
def gpus_available() -> bool:
    """Returns True if GPUs are available on the machine (available as an OmegaConf resolver).

    Examples:
        >>> with patch("torch.cuda.is_available", return_value=True):
        ...     gpus_available()
        True
        >>> with patch("torch.cuda.is_available", return_value=False):
        ...     gpus_available()
        False
    """
    return torch.cuda.is_available()


@OmegaConfResolver
def int_prod(x: int, y: int) -> int:
    """Returns the closest integer to the product of x and y (available as an OmegaConf resolver).

    Examples:
        >>> int_prod(2, 3)
        6
        >>> int_prod(2, 3.5)
        7
        >>> int_prod(2.49, 3)
        7
    """
    return round(x * y)


@OmegaConfResolver
def oc_min(x: int, y: int) -> int:
    """Returns the minimum of x and y (available as an OmegaConf resolver).

    Examples:
        >>> oc_min(5, 1)
        1
    """
    return min(x, y)


@OmegaConfResolver
def sub(x: int, y: int) -> int:
    """Returns x - y (available as an OmegaConf resolver).

    Examples:
        >>> sub(5, 1)
        4
    """
    return x - y


@OmegaConfResolver
def num_gpus() -> int:
    """Returns the number of GPUs available on the machine (available as an OmegaConf resolver).

    Examples:
        >>> with patch("torch.cuda.device_count", return_value=2):
        ...     num_gpus()
        2
    """
    return torch.cuda.device_count()


@OmegaConfResolver
def num_cores() -> int:
    """Returns the number of CPU cores available on the machine (available as an OmegaConf resolver).

    Examples:
        >>> with patch("multiprocessing.cpu_count", return_value=8):
        ...     num_cores()
        8
    """
    return multiprocessing.cpu_count()


@OmegaConfResolver
def resolve_generation_context_size(seq_lens: DictConfig) -> int:
    """Resolves the target generation context (input) size for the model.

    This function can be used in omega conf configs as a resolved function.

    Args:
        seq_lens: A configuration object containing the following key/value pairs:
            - max_generated_trajectory_len: If set, this gives the maximum length of trajectories (outputs)
              that should be generated.
            - frac_seq_len_as_context: If set, this gives the fraction of the pre-trained model's maximum
              sequence length that should be used as the context (input) for generation.
            - generation_context_size: If set, this gives the exact context size to use for generation.
            - pretrained_max_seq_len: The maximum sequence length of the pre-trained model.

    Returns:
        The generation context size, which is the maximum length of the input sequences the dataloader will
        pass to the model. The remaining length of the sequence will be used for generation. This will take
        one of several values depending on what is set:
            - If `generation_context_size` is set, it is returned.
            - If `max_generated_trajectory_len` is set, then
              `pretrained_max_seq_len - max_generated_trajectory_len` is returned.
            - If `frac_seq_len_as_context` is set, then
              `round(pretrained_max_seq_len * frac_seq_len_as_context)` is returned.

    Raises:
        TypeError: If the input keys have the wrong types.
        ValueError: If none of `max_generated_trajectory_len`, `frac_seq_len_as_context`, or
            `generation_context_size` are set, if more than one of them are set, if
            `pretrained_max_seq_len` is not set, or if the returned value would not be a positive integer.

    Examples:
        >>> resolve_generation_context_size(
        ...     {"pretrained_max_seq_len": 1024, "max_generated_trajectory_len": 512}
        ... )
        512
        >>> resolve_generation_context_size(
        ...     {"pretrained_max_seq_len": 1024, "generation_context_size": 100}
        ... )
        100
        >>> resolve_generation_context_size(
        ...     {"pretrained_max_seq_len": 1024, "frac_seq_len_as_context": 0.75}
        ... )
        768

    Fractional resolution is guaranteed to never be greater than the maximum sequence length or less than 1:

        >>> resolve_generation_context_size(
        ...     {"pretrained_max_seq_len": 1024, "frac_seq_len_as_context": 0.9999999999}
        ... )
        1023
        >>> resolve_generation_context_size(
        ...     {"pretrained_max_seq_len": 1024, "frac_seq_len_as_context": 0.0000000001}
        ... )
        1

    Null values do not trigger errors nor are used:

        >>> resolve_generation_context_size(
        ...     {
        ...         "pretrained_max_seq_len": 1024, "frac_seq_len_as_context": 0.75,
        ...         "generation_context_size": None
        ...     }
        ... )
        768

    Errors are raised if the input is missing required keys...

        >>> resolve_generation_context_size({})
        Traceback (most recent call last):
            ...
        ValueError: Required key 'pretrained_max_seq_len' not found in input.
        >>> resolve_generation_context_size({"pretrained_max_seq_len": 1024})
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of 'max_generated_trajectory_len' or 'frac_seq_len_as_context' or
            'generation_context_size' must be set to a non-null value.

    or if it has too many keys...

        >>> resolve_generation_context_size(
        ...     {
        ...         "pretrained_max_seq_len": 1024, "frac_seq_len_as_context": 0.75,
        ...         "generation_context_size": 256
        ...     }
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of 'max_generated_trajectory_len' or 'frac_seq_len_as_context' or
            'generation_context_size' must be set to a non-null value.

    or if it has extra keys...
        >>> resolve_generation_context_size(
        ...     {
        ...         "pretrained_max_seq_len": 1024, "frac_seq_len_as_context": 0.75,
        ...         "foobar": 256
        ...     }
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Extra keys found in input: ['foobar']. Only 'max_generated_trajectory_len',
            'frac_seq_len_as_context', 'generation_context_size', 'pretrained_max_seq_len' are allowed.

    or if the keys have the wrong types:

        >>> resolve_generation_context_size(
        ...     {"pretrained_max_seq_len": 1024, "generation_context_size": "foobar"}
        ... )
        Traceback (most recent call last):
            ...
        TypeError: Expected 'generation_context_size' to be an int; got <class 'str'>.
        >>> resolve_generation_context_size(
        ...     {"pretrained_max_seq_len": 1024, "max_generated_trajectory_len": -10}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Expected 'max_generated_trajectory_len' to be positive; got -10.
        >>> resolve_generation_context_size(
        ...     {"pretrained_max_seq_len": 1024, "frac_seq_len_as_context": 1.25}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: If non-null, 'frac_seq_len_as_context' must be a float between 0 and 1. Got 1.25.

    Errors are also raised if the output would not be a positive integer:

        >>> resolve_generation_context_size({"pretrained_max_seq_len": 1, "max_generated_trajectory_len": 5})
        Traceback (most recent call last):
            ...
        ValueError: The maximum sequence length of the pre-trained model must be greater than the maximum
            generated trajectory length. Got 1 and 5.
    """

    if seq_lens.get("pretrained_max_seq_len", None) is None:
        raise ValueError("Required key 'pretrained_max_seq_len' not found in input.")

    allowed_keys = [
        "max_generated_trajectory_len",
        "frac_seq_len_as_context",
        "generation_context_size",
        "pretrained_max_seq_len",
    ]

    if extra_keys := set(seq_lens.keys()) - set(allowed_keys):
        allowed_keys_str = "', '".join(allowed_keys)
        raise ValueError(
            f"Extra keys found in input: {sorted(extra_keys)}. Only '{allowed_keys_str}' are allowed."
        )

    non_null_keys = {k: v for k, v in seq_lens.items() if v is not None}
    pretrained_seq_len = non_null_keys.pop("pretrained_max_seq_len")

    if len(non_null_keys) != 1:
        raise ValueError(
            "Exactly one of 'max_generated_trajectory_len' or 'frac_seq_len_as_context' or "
            "'generation_context_size' must be set to a non-null value."
        )

    for k in ["pretrained_max_seq_len", "max_generated_trajectory_len", "generation_context_size"]:
        if k not in non_null_keys:
            continue
        if not isinstance(seq_lens[k], int):
            raise TypeError(f"Expected '{k}' to be an int; got {type(seq_lens[k])}.")
        if seq_lens[k] <= 0:
            raise ValueError(f"Expected '{k}' to be positive; got {seq_lens[k]}.")

    if "generation_context_size" in non_null_keys:
        return non_null_keys["generation_context_size"]
    if "max_generated_trajectory_len" in non_null_keys:
        if pretrained_seq_len <= non_null_keys["max_generated_trajectory_len"]:
            raise ValueError(
                "The maximum sequence length of the pre-trained model must be greater than the maximum "
                f"generated trajectory length. Got {pretrained_seq_len} and "
                f"{non_null_keys['max_generated_trajectory_len']}."
            )
        return pretrained_seq_len - non_null_keys["max_generated_trajectory_len"]
    if "frac_seq_len_as_context" in non_null_keys:
        val = non_null_keys["frac_seq_len_as_context"]
        if not isinstance(val, float) or val < 0 or val > 1:
            raise ValueError(
                f"If non-null, 'frac_seq_len_as_context' must be a float between 0 and 1. Got {val}."
            )
        return min(max(round(pretrained_seq_len * val), 1), pretrained_seq_len - 1)


def save_resolved_config(cfg: DictConfig, fp: Path) -> bool:
    """Save a fully resolved version of an OmegaConf DictConfig.

    Args:
        cfg: The OmegaConf DictConfig to resolve and save.
        fp: The path where the resolved configuration should be saved.

    Returns:
        True if the configuration was successfully saved, False otherwise.

    This function resolves all interpolations in the provided DictConfig and saves it to the specified file
    path. If the resolution fails, it will log a warning and do nothing. This function will not error out.

    Examples:
        >>> cfg = DictConfig({"some_other_key": "value", "key": "${some_other_key}"})
        >>> with print_warnings(), tempfile.NamedTemporaryFile(suffix=".yaml") as tmp_file:
        ...     saved = save_resolved_config(cfg, Path(tmp_file.name))
        ...     contents = Path(tmp_file.name).read_text()
        ...     print(f"Saved: {saved}")
        ...     print("Contents:")
        ...     print(contents)
        Saved: True
        Contents:
        some_other_key: value
        key: value

    If the resolution fails, it will log a warning and return False:

        >>> cfg = DictConfig({"key": "${non_existent_key}"})
        >>> with print_warnings(), tempfile.NamedTemporaryFile(suffix=".yaml") as tmp_file:
        ...     saved = save_resolved_config(cfg, Path(tmp_file.name))
        ...     print(f"Saved: {saved}")
        Saved: False
        Warning: Could not save resolved config: Interpolation key 'non_existent_key' not found...
    """

    try:
        # Create a copy and resolve all interpolations
        resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(resolved_cfg, fp)
        return True
    except Exception as e:
        logger.warning(f"Could not save resolved config: {e}")
        return False


def apply_saved_logger_run_ids(trainer_cfg: DictConfig, run_dir: Path) -> None:
    """Populate logger configs with saved experiment IDs if present.

    This helper mutates the provided trainer configuration in-place. It is kept
    separate from OmegaConf resolvers to keep the configuration loading simple
    while still allowing logger IDs to be injected automatically when a run is
    resumed.
    """

    if trainer_cfg is None:
        return

    loggers = []
    if "logger" in trainer_cfg:
        loggers.append(trainer_cfg.logger)
    if "loggers" in trainer_cfg:
        loggers.extend(trainer_cfg.loggers)

    log_dir = Path(run_dir) / "loggers"

    for logger_cfg in loggers:
        target = str(logger_cfg.get("_target_", "")).lower()
        if "wandb" in target:
            fp = log_dir / "wandb_run_id.txt"
            if fp.is_file() and not logger_cfg.get("id"):
                logger_cfg["id"] = fp.read_text().strip()
                logger_cfg.setdefault("resume", "allow")
        elif "mlflow" in target:
            fp = log_dir / "mlflow_run_id.txt"
            if fp.is_file() and not logger_cfg.get("run_id"):
                logger_cfg["run_id"] = fp.read_text().strip()


def save_logger_run_ids(loggers: Sequence[Logger], run_dir: Path) -> None:
    """Save experiment IDs for MLFlow and WandB loggers."""

    log_dir = Path(run_dir) / "loggers"
    log_dir.mkdir(parents=True, exist_ok=True)

    for logger in loggers:
        if is_mlflow_logger(logger):
            (log_dir / "mlflow_run_id.txt").write_text(str(logger.run_id))
            continue

        if is_wandb_logger(logger):
            (log_dir / "wandb_run_id.txt").write_text(str(logger.experiment.id))
