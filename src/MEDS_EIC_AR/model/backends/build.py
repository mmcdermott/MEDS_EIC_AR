"""Hydra-instantiable builder functions for :class:`GenerationBackend` implementations.

Each ``configs/backend/*.yaml`` names one of these via ``_target_`` and Hydra
``instantiate(cfg.backend, module=..., model_init_dir=...)`` returns either ``None`` (meaning
"keep the model's default HFBackend") or a concrete backend instance to pass to
:meth:`Model.set_backend`. Keeps the CLI free of any per-backend dispatch logic — adding a
new backend is one builder + one yaml, no code change in ``__main__.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...training.module import MEICARModule
    from .base import GenerationBackend

logger = logging.getLogger(__name__)


def build_hf_backend(*, module: MEICARModule, model_init_dir: str | Path) -> None:
    """No-op builder: the model already constructs an :class:`HFBackend` in ``__init__``.

    Returning ``None`` signals to the caller (``__main__.generate_trajectories``) to leave
    the model's default backend in place. Exists so ``configs/backend/hf.yaml`` can use the
    same ``_target_``-driven instantiation shape as other backends — no special-case branch
    in the CLI.
    """
    del module, model_init_dir  # unused; accepting them keeps the instantiate signature uniform
    return None


def build_sglang_backend(
    *,
    module: MEICARModule,
    model_init_dir: str | Path,
    engine_kwargs: dict[str, Any] | None = None,
) -> GenerationBackend:
    """Materialize the Lightning checkpoint as an HF directory and wrap in an :class:`SGLangBackend`.

    ``module.model.HF_model`` is saved (idempotently; see
    :func:`~MEDS_EIC_AR.model.backends.export.export_lightning_to_hf_dir`) under
    ``<model_init_dir>/hf_model/``, then handed to ``SGLangBackend`` together with any
    ``engine_kwargs`` from the backend config. Lazy imports ``sglang`` inside the adapter so
    the optional extra isn't required unless this builder actually runs.
    """
    from .export import export_lightning_to_hf_dir
    from .sglang import SGLangBackend

    hf_dir = export_lightning_to_hf_dir(module, Path(model_init_dir) / "hf_model")
    backend = SGLangBackend(hf_dir, engine_kwargs=dict(engine_kwargs or {}))
    logger.info(f"Generation backend switched to SGLang (engine at {hf_dir}).")
    return backend
