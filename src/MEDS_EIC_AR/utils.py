import multiprocessing
from hashlib import sha256

from MEDS_transforms.configs.utils import OmegaConfResolver
from omegaconf import DictConfig


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
def int_prod(x: int, y: int) -> int:
    """Returns the closest integer to the product of x and y.

    This function can be used in omega conf configs as a resolved function.

    Args:
        x: The first integer.
        y: The second integer.

    Returns:
        The closest integer to the product of x and y.

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
def sub(x: int, y: int) -> int:
    """Returns the difference of x and y.

    This function can be used in omega conf configs as a resolved function.

    Examples:
        >>> sub(5, 1)
        4
    """
    return x - y


@OmegaConfResolver
def num_cores() -> int:
    """Returns the number of CPU cores available on the machine.

    This function can be used in omega conf configs as a resolved function.

    Returns:
        The number of CPU cores available on the machine.

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
