"""Shared utilities that don't fit the training / preprocessing / generation module split.

Groups of helpers in this module:

- **OmegaConf resolvers** (``gpus_available``, ``num_cores``, ``num_gpus``, ``oc_min``, ``int_prod``,
  ``resolve_generation_context_size``) — registered as Hydra/OmegaConf resolvers so config
  interpolations can read hardware state and compute derived values. ``hash_based_seed`` lives alongside
  them as a regular Python helper (called from ``__main__`` at per-split seed time), not as a resolver.
- **Logger restore/save** (``save_logger_run_ids``, ``apply_saved_logger_run_ids``) — lets training
  resumes reuse the same MLflow / WandB run IDs so a paused-and-resumed run looks like a single run in
  the tracking backend.
- **Environment snapshotting** (``save_environment_snapshot``) — writes ``environment.txt`` to the run
  ``output_dir`` on initial run creation (not on resume), capturing Python version, platform, and every
  installed distribution and version. See issue #24 / PR #129.
- **Resolved-config persistence** (``save_resolved_config``).
- **Logger detection** (``is_mlflow_logger``, ``is_wandb_logger``) — optional-import-safe predicates
  used by the training hooks and by ``save_logger_run_ids`` to route each attached logger to its
  backend-specific run-id save path.
"""

import logging
import multiprocessing
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import torch
from lightning.pytorch.loggers import Logger
from MEDS_transforms.configs.utils import OmegaConfResolver
from omegaconf import DictConfig, ListConfig, OmegaConf

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

    Example:
        >>> class DummyLogger:
        ...     ...
        >>> is_wandb_logger(DummyLogger())
        False
        >>> import builtins
        >>> from unittest.mock import patch
        >>> original_import = builtins.__import__
        >>> def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        ...     if name == "lightning.pytorch.loggers" and "WandbLogger" in fromlist:
        ...         raise ImportError
        ...     return original_import(name, globals, locals, fromlist, level)
        >>> with patch.object(builtins, "__import__", fake_import):
        ...     is_wandb_logger(DummyLogger())
        False
    """

    try:
        from lightning.pytorch.loggers import WandbLogger

        return isinstance(logger, WandbLogger)
    except ImportError:
        return False


def hash_based_seed(seed: int | None, split: str) -> int:
    """Generates a hash-based seed for reproducibility.

    This function generates a hash-based seed using the provided seed and split values. It is
    designed to be used in conjunction with OmegaConf for configuration management.

    Args:
        seed: The original seed value. THIS WILL NOT OVERWRITE THE OUTPUT. Rather, this just ensures the
            sequence of seeds chosen can be deterministically updated by changing a base parameter.
        split: The split identifier.

    Returns:
        A hash-based seed value.

    Examples:
        >>> hash_based_seed(42, "train")
        1631825622
        >>> hash_based_seed(None, "held_out")
        1088888987
    """

    hash_str = f"{seed}_{split}"
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


def save_environment_snapshot(fp: Path) -> bool:
    """Save a snapshot of the Python environment a run is using.

    Writes a ``pip freeze``-style listing of installed packages plus a header with the
    Python version and platform string. Lets anyone returning to a run output directory
    later localize the exact codebase + dependency set that produced the result — useful
    for reproducing or debugging trajectories from a model trained weeks or months ago,
    when the underlying wheels on PyPI have moved on.

    Format:

    .. code-block:: text

        # MEDS_EIC_AR run environment snapshot
        # python: 3.12.3 (main, Jun  7 2024, 00:00:00) ...
        # platform: Linux-6.8.0-generic-x86_64-with-glibc2.39
        MEDS-EIC-AR==0.X.Y
        lightning==2.5.1
        ...

    Never raises — any failure (permission denied, disk full, etc.) logs a warning and
    returns ``False`` so the calling entry point can keep going. The snapshot is a
    nice-to-have, not a correctness invariant.

    Args:
        fp: The path where the snapshot should be written. The parent directory is
            created if it doesn't already exist.

    Returns:
        ``True`` if the snapshot was written successfully, ``False`` otherwise.

    Example:
        Case-insensitive alphabetical sort lets us anchor the doctest on a few
        known-always-present top-level deps (``lightning``, ``polars``, ``torch``)
        with ellipsis on their versions so CI doesn't flake on routine upstream
        version bumps. The ordering of the three is stable: ``l < p < t``.

        >>> with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
        ...     _ = save_environment_snapshot(Path(tmp_file.name))
        ...     print(Path(tmp_file.name).read_text())  # doctest: +ELLIPSIS
        # MEDS_EIC_AR run environment snapshot
        # python: ...
        # platform: ...
        ...
        lightning==...
        ...
        polars==...
        ...
        torch==...
        ...

    Per-invariant assertions (header format, pip-freeze line shape, sort order,
    missing-parent-dir handling, etc.) live in
    ``tests/test_environment_snapshot.py`` as readable pytest cases rather than
    cluttering the docstring.
    """
    import importlib.metadata
    import platform
    import sys

    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# MEDS_EIC_AR run environment snapshot",
            f"# python: {sys.version.splitlines()[0]}",
            f"# platform: {platform.platform()}",
        ]
        packages = []
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            if name is None:
                continue
            packages.append(f"{name}=={dist.version}")
        # Case-insensitive sort so ``pip freeze`` -style orderings match across platforms
        # (macOS distribution discovery returns mixed case; Linux usually canonicalizes).
        packages.sort(key=str.lower)
        lines.extend(packages)
        fp.write_text("\n".join(lines) + "\n")
        return True
    except Exception as e:  # pragma: no cover — best-effort, swallows any disk/IO failure
        logger.warning(f"Could not save environment snapshot: {e}")
        return False


def _read_saved_id(fp: Path) -> str | None:
    """Read a saved-id file defensively; return ``None`` if the file is missing, unreadable, or blank.

    Called by :func:`apply_saved_logger_run_ids` for every ``mlflow_run_id.txt`` /
    ``wandb_run_id.txt`` / ``mlflow_tracking_uri.txt`` lookup. Treating an empty or
    whitespace-only file as "no saved id" matches user intent: a sentinel with no content
    shouldn't silently poison downstream logger instantiation with an invalid ``run_id=""``.
    I/O errors (permission denied, concurrent delete, disk failures) are logged at warning
    level and treated as "no saved id" — the caller's run proceeds with the pre-restore
    config rather than crashing before any compute has run.
    """
    if not fp.is_file():
        return None
    try:
        content = fp.read_text().strip()
    except OSError as e:
        logger.warning(f"Could not read saved logger id at {fp}: {e}. Skipping restore.")
        return None
    return content or None


def apply_saved_logger_run_ids(trainer_cfg: DictConfig, run_dir: Path) -> None:
    """Populate logger configs with saved experiment IDs if present.

    This helper mutates the provided trainer configuration in-place and reads
    any saved run IDs from ``<run_dir>/loggers``. It is kept separate from
    OmegaConf resolvers so configuration loading remains straightforward.

    Example:
        >>> from yaml_to_disk import yaml_disk
        >>> cfg = DictConfig(
        ...     {"loggers": [{"_target_": "MLFlowLogger"}, {"_target_": "WandbLogger"}]}
        ... )
        >>> disk = '''
        ... loggers:
        ...   "mlflow_run_id.txt": abc
        ...   "mlflow_tracking_uri.txt": file:///tmp/mlruns_original
        ...   "wandb_run_id.txt": xyz
        ... '''
        >>> with yaml_disk(disk) as run_dir:
        ...     apply_saved_logger_run_ids(cfg, run_dir)
        ...     print(cfg.loggers[0]["run_id"], cfg.loggers[0]["tracking_uri"])
        ...     print(cfg.loggers[1]["id"], cfg.loggers[1]["resume"])
        abc file:///tmp/mlruns_original
        xyz allow

    The restore is all-or-nothing: when we apply a saved ``run_id``, we also override
    ``tracking_uri`` with the saved value — including when the current config sets
    ``tracking_uri`` to something else (the default ``configs/trainer/logger/mlflow.yaml``
    does). That is intentional: resuming a ``run_id`` in a store it wasn't created in is
    incoherent (the run doesn't exist there), and the repo default interpolates
    ``tracking_uri`` off the current ``${log_dir}``, so without this override a resumed run
    would 404 or log to a new store.

        >>> cfg = DictConfig(
        ...     {"loggers": [{"_target_": "MLFlowLogger", "tracking_uri": "file:///tmp/default_store"}]}
        ... )
        >>> with yaml_disk(disk) as run_dir:
        ...     apply_saved_logger_run_ids(cfg, run_dir)
        ...     print(cfg.loggers[0]["run_id"], cfg.loggers[0]["tracking_uri"])
        abc file:///tmp/mlruns_original

    To log a new run into a different store (no resume), set ``run_id`` explicitly in the
    current config — that suppresses the saved-``run_id`` restore, which in turn suppresses
    the saved-``tracking_uri`` restore:

        >>> cfg = DictConfig(
        ...     {"loggers": [{
        ...         "_target_": "MLFlowLogger",
        ...         "run_id": "fresh",
        ...         "tracking_uri": "file:///tmp/new_store",
        ...     }]}
        ... )
        >>> with yaml_disk(disk) as run_dir:
        ...     apply_saved_logger_run_ids(cfg, run_dir)
        ...     print(cfg.loggers[0]["run_id"], cfg.loggers[0]["tracking_uri"])
        fresh file:///tmp/new_store
    """

    if trainer_cfg is None:
        return

    # Lightning accepts ``logger: bool | Logger | Iterable[Logger]``; Hydra lets users disable
    # logging entirely with ``trainer.logger=false`` or ``trainer.logger=null``. Normalize all
    # three cases into a flat list of dict-like configs before we start indexing with ``.get``.
    # Anything non-mapping (``True``/``False``/``None``, concrete Logger instances from Hydra
    # ``_target_`` resolution — though this helper usually runs *before* instantiation, it can
    # also be called on already-resolved configs) is skipped silently.
    raw_entries: list = []
    if "logger" in trainer_cfg:
        raw_entries.append(trainer_cfg.logger)
    if "loggers" in trainer_cfg:
        loggers_field = trainer_cfg.loggers
        if isinstance(loggers_field, DictConfig | dict):
            raw_entries.append(loggers_field)
        elif isinstance(loggers_field, list | ListConfig):
            raw_entries.extend(loggers_field)
        # Any other shape (bool, None, concrete object) is just a single non-mapping entry;
        # the ``hasattr`` guard below skips it.

    loggers = [e for e in raw_entries if e is not None and hasattr(e, "get")]

    log_dir = Path(run_dir) / "loggers"

    for logger_cfg in loggers:
        target = str(logger_cfg.get("_target_", "")).lower()
        if "wandb" in target:
            fp = log_dir / "wandb_run_id.txt"
            saved = _read_saved_id(fp)
            if saved and not logger_cfg.get("id"):
                logger_cfg["id"] = saved
                logger_cfg.setdefault("resume", "allow")
        elif "mlflow" in target:
            fp = log_dir / "mlflow_run_id.txt"
            applied_saved_run_id = False
            saved = _read_saved_id(fp)
            if saved and not logger_cfg.get("run_id"):
                logger_cfg["run_id"] = saved
                applied_saved_run_id = True
            # Restore ``tracking_uri`` whenever we just applied a saved ``run_id``, overriding
            # whatever the current config had there. An MLflow ``run_id`` is only resolvable
            # against the tracking store it was created in; the in-repo default
            # (``configs/trainer/logger/mlflow.yaml``) sets ``tracking_uri: ${log_dir}/mlflow/mlruns``,
            # which derives from the *current* run's ``output_dir``, so gating the restore on
            # ``not logger_cfg.get("tracking_uri")`` (what this code did initially) would never
            # fire in the common case — the default already populates the field with the new
            # run's path, and we'd quietly resume a run_id against the wrong store.
            #
            # The escape hatch moves: to log to a different store for a genuinely new run, the
            # caller leaves ``run_id`` explicitly set (or absent from disk) so the restore above
            # doesn't fire, and we leave ``tracking_uri`` alone. That's coherent. The opposite —
            # "resume this run_id, but log to a new store" — is incoherent (the run_id doesn't
            # exist in the new store), and this code now prevents it by construction.
            if applied_saved_run_id:
                saved_uri = _read_saved_id(log_dir / "mlflow_tracking_uri.txt")
                if saved_uri:
                    logger_cfg["tracking_uri"] = saved_uri


def save_logger_run_ids(loggers: Sequence[Logger], run_dir: Path) -> None:
    """Save experiment IDs for MLFlow and WandB loggers.

    Args:
        loggers: Collection of :class:`~lightning.pytorch.loggers.Logger` objects
            used during the run.
        run_dir: Directory where run IDs should be stored.

    Example:
        >>> class DummyMLFlowLogger:
        ...     def __init__(self, run_id="foo"):
        ...         self.run_id = run_id
        >>> class DummyWandBExp:
        ...     def __init__(self, id="bar"):
        ...         self.id = id
        >>> class DummyWandbLogger:
        ...     def __init__(self, exp_id="bar"):
        ...         self.experiment = DummyWandBExp(exp_id)
        >>> import tempfile
        >>> from unittest.mock import patch
        >>> mlflow_patch = patch(
        ...     "lightning.pytorch.loggers.MLFlowLogger", DummyMLFlowLogger, create=True
        ... )
        >>> wandb_patch = patch(
        ...     "lightning.pytorch.loggers.WandbLogger", DummyWandbLogger, create=True
        ... )
        >>> with mlflow_patch, wandb_patch, tempfile.TemporaryDirectory() as tmp:
        ...     run_dir = Path(tmp)
        ...     save_logger_run_ids(
        ...         [DummyMLFlowLogger("mlflow"), DummyWandbLogger("wandb")], run_dir
        ...     )
        ...     print((run_dir / "loggers" / "mlflow_run_id.txt").read_text())
        ...     print((run_dir / "loggers" / "wandb_run_id.txt").read_text())
        mlflow
        wandb
    """

    log_dir = Path(run_dir) / "loggers"
    log_dir.mkdir(parents=True, exist_ok=True)

    for logger in loggers:
        if is_mlflow_logger(logger):
            (log_dir / "mlflow_run_id.txt").write_text(str(logger.run_id))
            # Persist ``tracking_uri`` too so ``apply_saved_logger_run_ids`` can attach the
            # resumed run back to the store it was created in. Attribute is
            # ``_tracking_uri`` (private) on Lightning's MLFlowLogger; fall back to any
            # public ``tracking_uri`` if a future Lightning version exposes one. Missing
            # attribute is survivable — the resume call will work in the common case where
            # the current tracking_uri happens to match the saved one, and fail with a
            # clear error otherwise.
            tracking_uri = getattr(logger, "_tracking_uri", None) or getattr(logger, "tracking_uri", None)
            if tracking_uri:
                (log_dir / "mlflow_tracking_uri.txt").write_text(str(tracking_uri))
            continue

        if is_wandb_logger(logger):
            (log_dir / "wandb_run_id.txt").write_text(str(logger.experiment.id))
