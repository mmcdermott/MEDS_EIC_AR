from typing import Any

from omegaconf import DictConfig


def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())
