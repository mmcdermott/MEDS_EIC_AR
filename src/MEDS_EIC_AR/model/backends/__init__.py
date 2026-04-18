"""Pluggable generation backends — see :mod:`MEDS_EIC_AR.model.backends.base`."""

from .base import GenerationBackend
from .hf import HFBackend
from .sglang import SGLangBackend

__all__ = ["GenerationBackend", "HFBackend", "SGLangBackend"]
