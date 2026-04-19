"""Pluggable generation backends — see :mod:`MEDS_EIC_AR.model.backends.base`."""

from .base import GenerationBackend
from .build import build_hf_backend, build_sglang_backend
from .hf import HFBackend
from .sglang import SGLangBackend

__all__ = [
    "GenerationBackend",
    "HFBackend",
    "SGLangBackend",
    "build_hf_backend",
    "build_sglang_backend",
]
