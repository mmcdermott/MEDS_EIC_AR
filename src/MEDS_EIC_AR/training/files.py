from pathlib import Path
from typing import Any

from meds_torchdata.types import PaddingSide, StaticInclusionMode, SubsequenceSamplingStrategy
from omegaconf import DictConfig, OmegaConf

ALLOWED_DIFFERENCE_KEYS = {
    "do_resume",
    "output_dir",
    "do_overwrite",
    "datamodule.num_workers",
    "trainer.logger.save_dir",
    "trainer.callbacks.model_checkpoint.dirpath",
    "trainer.default_root_dir",
    "log_dir",
}

STR_ENUM_PARAMS = {
    "seq_sampling_strategy": SubsequenceSamplingStrategy,
    "padding_side": PaddingSide,
    "static_inclusion_mode": StaticInclusionMode,
}


def resolve_enum_value(value: Any, enum_type: type) -> Any:
    """Resolves a value to its corresponding enum type, handling both string and enum inputs.

    Args:
        value: The value to resolve, which can be a string or an instance of the enum type.
        enum_type: The enum type to resolve the value to.

    Returns:
        The resolved enum value.

    Raises:
        ValueError: If the value cannot be resolved to the enum type.

    Examples:
        >>> resolve_enum_value("left", PaddingSide)
        <PaddingSide.LEFT: 'left'>
        >>> resolve_enum_value("LEFT", PaddingSide)
        <PaddingSide.LEFT: 'left'>
        >>> resolve_enum_value("RANDOM", SubsequenceSamplingStrategy)
        <SubsequenceSamplingStrategy.RANDOM: 'random'>
        >>> resolve_enum_value(SubsequenceSamplingStrategy.RANDOM, SubsequenceSamplingStrategy)
        <SubsequenceSamplingStrategy.RANDOM: 'random'>
        >>> resolve_enum_value(34, PaddingSide)
        Traceback (most recent call last):
            ...
        ValueError: Expected 34 to be a string or PaddingSide, got int.
    """
    match value:
        case str():
            return enum_type(value.lower())
        case enum_type():
            return value
        case _:
            raise ValueError(
                f"Expected {value} to be a string or {enum_type.__name__}, got {type(value).__name__}."
            )


def diff_configs(new_config: dict[str, Any], old_config: dict[str, Any]) -> dict[str, str]:
    """Compares two configurations and returns a dictionary mapping key names to any differences found.

    This does one additional check beyond pure exclusion -- it looks to see if enum names match for enum
    parameters that undergo upper-casing or lower-casing via hydra / omegaconf / strenum sheniganigans.

    Args:
        new_config: The new configuration to compare.
        old_config: The old configuration to compare against.

    Returns:
        A dictionary mapping key names to strings describing how the values differ between the two
        configurations. This dictionary is ordered by the order of the appearance of the keys in the old
        configuration followed by the new configuration.

    Raises:
        ValueError: If the configurations are identical.

    Examples:
        >>> old_cfg = {"c": 1, "b": 3}
        >>> new_cfg = {"c": 1, "b": 2}
        >>> diff_configs(new_cfg, old_cfg)
        {'b': 'has value 3 (int) in the old config, but 2 (int) in the new config.'}
        >>> old_cfg = {"c": 1, "foo": {"bar": 3}}
        >>> new_cfg = {"c": 1, "b": 2, "foo": {}}
        >>> diff_configs(new_cfg, old_cfg)
        {'foo.bar': 'is present in the old config, but not in the new config.',
         'b': 'is not present in the old config, but is in the new config.'}

    There are a few enum parameters that are allowed to differ in casing between the two configurations, shown
    below:

        >>> old_cfg = {
        ...     "padding_side": "left", "seq_sampling_strategy": "RANDOM", "static_inclusion_mode": "omit"
        ... }
        >>> new_cfg = {
        ...     "padding_side": "LEFT", "seq_sampling_strategy": SubsequenceSamplingStrategy.RANDOM,
        ...     "static_inclusion_mode": "omit"
        ... }
        >>> diff_configs(new_cfg, old_cfg)
        {}
    """

    differences = {}
    for key, old_val in old_config.items():
        if key not in new_config:
            differences[key] = "is present in the old config, but not in the new config."
            continue

        new_val = new_config[key]

        if isinstance(old_val, dict) and isinstance(new_config[key], dict):
            nested_diffs = diff_configs(new_val, old_val)
            for nested_key, nested_diff in nested_diffs.items():
                differences[f"{key}.{nested_key}"] = nested_diff
            continue

        different = old_val != new_val

        if key in STR_ENUM_PARAMS:
            enum_type = STR_ENUM_PARAMS[key]

            old_val_enum = resolve_enum_value(old_val, enum_type)
            new_val_enum = resolve_enum_value(new_val, enum_type)

            different = old_val_enum != new_val_enum

        if different:
            differences[key] = (
                f"has value {old_val} ({type(old_val).__name__}) in the old config, "
                f"but {new_val} ({type(new_val).__name__}) in the new config."
            )

    for key in new_config:
        if key not in old_config:
            differences[key] = "is not present in the old config, but is in the new config."

    return differences


def validate_resume_directory(output_dir: Path, cfg: DictConfig):
    """Validates that this is a valid directory to resume training from for the given configuration.

    A valid directory must contain a valid configuration file corresponding to an _identical_ run to the
    target run (save for non-model parameters such as `output_dir`, `do_resume`, etc.).

    Args:
        output_dir: The directory to validate.
        cfg: The configuration for the run attempting to resume from the given directory.

    Raises:
        FileNotFoundError: If the configuration file does not exist in the output directory.
        ValueError: If the directory is not valid for resuming training for any reason.

    Examples:
        >>> disk = '''
        ...   config.yaml:
        ...     max_seq_len: 10
        ...     vocab_size: 100
        ...     do_resume: false
        ...     output_dir: /tmp/output
        ...     datamodule:
        ...       batch_size: 32
        ... '''
        >>> input_cfg = DictConfig({
        ...     "max_seq_len": 10,
        ...     "vocab_size": 100,
        ...     "do_resume": True,
        ...     "datamodule": {"batch_size": 32},
        ... })
        >>> with yaml_disk(disk) as output_dir:
        ...     validate_resume_directory(output_dir, input_cfg) # No errors

    Errors will return informative messages about the ways in which the configurations differ:

        >>> input_cfg = DictConfig({
        ...     "max_seq_len": 20,
        ...     "hidden_size": 256,
        ...     "do_resume": True,
        ...     "datamodule": {"batch_size": 64, "num_workers": 4},
        ... })
        >>> with yaml_disk(disk) as output_dir:
        ...     validate_resume_directory(output_dir, input_cfg)
        Traceback (most recent call last):
            ...
        ValueError: The configuration in the output directory does not match the input:
          - key 'datamodule.batch_size' has value 32 (int) in the old config, but 64 (int) in the new config.
          - key 'max_seq_len' has value 10 (int) in the old config, but 20 (int) in the new config.
          - key 'vocab_size' is present in the old config, but not in the new config.
          - key 'hidden_size' is not present in the old config, but is in the new config.

    An error will also be raised if there is no configuration file:

        >>> with yaml_disk('foo.txt: hi') as output_dir:
        ...     validate_resume_directory(output_dir, input_cfg)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Configuration file /tmp/tmp.../config.yaml does not exist in the output directory.
    """
    old_cfg_fp = output_dir / "config.yaml"
    if not old_cfg_fp.is_file():
        raise FileNotFoundError(f"Configuration file {old_cfg_fp} does not exist in the output directory.")

    old_cfg = OmegaConf.load(old_cfg_fp)

    old_cfg = OmegaConf.to_container(old_cfg, resolve=True)
    new_cfg = OmegaConf.to_container(cfg, resolve=True)

    differences = diff_configs(new_cfg, old_cfg)

    err_lines = []
    for key, diff in differences.items():
        if key in ALLOWED_DIFFERENCE_KEYS:
            continue
        err_lines.append(f"  - key '{key}' {diff}")

    if err_lines:
        err_lines_str = "\n".join(err_lines)
        raise ValueError(
            f"The configuration in the output directory does not match the input:\n{err_lines_str}"
        )


def find_checkpoint_path(output_dir: Path) -> Path | None:
    """Finds and returns the latest checkpoint path in the given output directory.

    Args:
        output_dir: The directory to search for checkpoints.

    Returns:
        The latest checkpoint path if one exists, otherwise None. The latest checkpoint is determined by the
        file names, to conform to the naming convention used by this module; namely,
        `epoch=${epoch}-step=${step}.ckpt` or `last.ckpt`.

    Raises:
        NotADirectoryError: If the checkpoints directory is a file instead of a directory.

    Examples:
        >>> print_directory(pretrained_model)
        ├── .logs
        │   ├── .hydra
        │   │   ├── config.yaml
        │   │   ├── hydra.yaml
        │   │   └── overrides.yaml
        │   └── __main__.log
        ├── best_model.ckpt
        ├── checkpoints
        │   ├── epoch=0-step=1.ckpt
        │   ├── epoch=0-step=2.ckpt
        │   ├── epoch=1-step=3.ckpt
        │   ├── epoch=1-step=4.ckpt
        │   └── last.ckpt
        ├── config.yaml
        └── loggers
            └── csv
                └── version_0
                    ├── hparams.yaml
                    └── metrics.csv

    Given the structure of an output directory as shown above, this function will return the path to the
    latest checkpoint file, leveraging `last.ckpt` if it exists, or the checkpoint file with the highest epoch
    and step numbers if `last.ckpt` is not present:

        >>> find_checkpoint_path(pretrained_model)
        PosixPath('/tmp/.../checkpoints/last.ckpt')
        >>> with yaml_disk('''
        ...   checkpoints:
        ...     - 'epoch=0-step=2.ckpt'
        ...     - 'epoch=0-step=3.ckpt'
        ...     - 'epoch=0-step=4.ckpt'
        ...     - 'epoch=1-step=5.ckpt'
        ...     - 'epoch=1-step=6.ckpt'
        ... ''') as output_dir:
        ...     find_checkpoint_path(output_dir)
        PosixPath('/tmp/.../checkpoints/epoch=1-step=6.ckpt')

    An error will be raised if the checkpoints directory is a file instead of a directory:

        >>> with yaml_disk('  - "checkpoints"') as output_dir:
        ...     find_checkpoint_path(output_dir)
        Traceback (most recent call last):
            ...
        NotADirectoryError: Checkpoints directory /tmp/.../checkpoints is a file, not a directory.
    """

    checkpoints_dir = output_dir / "checkpoints"

    if checkpoints_dir.is_file():
        raise NotADirectoryError(f"Checkpoints directory {checkpoints_dir} is a file, not a directory.")
    elif not checkpoints_dir.exists():
        return None

    last_ckpt = checkpoints_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt

    checkpoint_fps = list(checkpoints_dir.glob("epoch=*-step=*.ckpt"))
    if not checkpoint_fps:
        return None

    def get_epoch(fp: Path) -> int:
        return int(fp.stem.split("-")[0].split("=")[1])

    def get_step(fp: Path) -> int:
        return int(fp.stem.split("-")[1].split("=")[1])

    sorted_checkpoints = sorted(checkpoint_fps, key=lambda fp: (get_epoch(fp), get_step(fp)))

    return sorted_checkpoints[-1] if sorted_checkpoints else None
