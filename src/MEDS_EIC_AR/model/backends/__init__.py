"""Pluggable generation backends — see :mod:`MEDS_EIC_AR.model.backends.base`."""

from .base import GenerationBackend
from .hf import HFBackend

__all__ = ["GenerationBackend", "HFBackend"]
