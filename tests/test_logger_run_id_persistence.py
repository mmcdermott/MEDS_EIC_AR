"""Failing regression tests for the logger run-id save / restore flow.

Two distinct bugs are covered, intentionally co-located so the eventual fix can be
reviewed against both at once:

- **Issue #152 — save fires too late**: ``save_logger_run_ids`` is invoked at
  ``__main__.pretrain`` *after* ``trainer.fit(...)`` returns. For a clean-completion
  run that's fine; for the case the helper actually exists to support
  (interrupted-and-resumed runs from OOM / SIGINT / OS reboot) ``trainer.fit`` never
  returns, the save line never executes, no ``wandb_run_id.txt`` is written, and the
  next ``MEICAR_pretrain do_resume=True`` invocation finds nothing to restore — wandb
  spawns a fresh run and continuity is lost. The fix is to persist run ids on
  ``on_train_start`` (a Lightning Callback) rather than only after fit returns.

- **Issue #131 — generate save/restore paths don't connect**: ``generate_trajectories``
  *restores* run ids from ``cfg.model_initialization_dir`` but *saves* them to
  ``cfg.output_dir``. Nothing reads ``output_dir`` back on the next generation resume
  — it always re-restores from the training dir. The save is an orphan write. The fix
  is layered lookup: read ``output_dir`` first, fall back to ``model_initialization_dir``.

Both issues are bundled here because the fix touches the same two functions
(``apply_saved_logger_run_ids`` and ``save_logger_run_ids``) plus their callers in
``__main__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from omegaconf import DictConfig

import MEDS_EIC_AR.utils as utils

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Issue #152 — wandb run id must be on disk before trainer.fit returns
# ---------------------------------------------------------------------------


def test_pretrain_persists_wandb_run_id_when_fit_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preprocessed_dataset: Path,
):
    """Bug #152: the wandb run id must be persisted at training START, not at fit RETURN.

    Today ``__main__.pretrain`` calls ``save_logger_run_ids`` *after* ``trainer.fit(...)``
    returns. That means an interrupted run (the case the helper exists to support) never
    persists its id, and a subsequent ``MEICAR_pretrain do_resume=True`` finds no saved
    id and lets wandb spawn a fresh run — silently breaking continuity.

    Strategy: build the demo pretrain config with an offline wandb logger, patch
    ``Trainer.fit`` to fire ``on_train_start`` callbacks then raise (mimicking a crash
    that returns control to Python via an exception, not via clean fit completion), and
    assert ``loggers/wandb_run_id.txt`` already exists on disk. Under the current code,
    no callback writes the file and the trailing ``save_logger_run_ids`` never runs, so
    the file is absent — test fails. Under the fix, an ``on_train_start`` callback writes
    the file before the simulated crash, so the file is present — test passes.
    """
    pytest.importorskip("wandb")

    # Keep wandb fully offline so the test is hermetic — no remote contact, no auth, no
    # network flake. ``WANDB_DIR`` redirects state into the test's tmp_path so we don't
    # leave a ``./wandb/`` directory behind in the repo root.
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb_state"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb_cfg"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb_cache"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DISABLE_GIT", "true")

    output_dir = tmp_path / "interrupted_run"
    output_dir.mkdir()

    import lightning.pytorch as L

    def fault_inject_fit(self, model=None, *args, **kwargs):
        """Stand-in for ``Trainer.fit`` that mimics a crash mid-training.

        Materializes loggers (so any lazily-initialized backend like wandb has its
        ``.experiment.id`` populated), fires ``on_train_start`` on every attached
        callback (so any callback-based persistence runs), and then raises — so the
        caller sees an in-flight failure rather than a clean fit return.
        """
        for lg in self.loggers:
            if hasattr(lg, "experiment"):
                _ = lg.experiment
        for cb in list(self.callbacks):
            if hasattr(cb, "on_train_start"):
                cb.on_train_start(self, model)
        raise KeyboardInterrupt("simulated mid-training crash")

    monkeypatch.setattr(L.Trainer, "fit", fault_inject_fit)

    from hydra import compose, initialize_config_module

    from MEDS_EIC_AR.__main__ import pretrain

    with initialize_config_module(config_module="MEDS_EIC_AR.configs", version_base=None):
        cfg = compose(
            config_name="_demo_pretrain",
            overrides=[
                f"output_dir={output_dir}",
                f"datamodule.config.tensorized_cohort_dir={preprocessed_dataset}",
                "trainer/logger=wandb",
                "trainer.logger.offline=true",
                f"trainer.logger.save_dir={tmp_path / 'wandb_save'}",
                # Avoid wandb-side warnings about unset ``project`` in offline mode.
                "trainer.logger.project=meds_eic_ar_test",
                # Drop the ``${hydra:runtime.choices...}`` interpolation in the wandb config
                # — under ``hydra.compose`` (no ``hydra.main``) ``HydraConfig`` is not set,
                # so the interpolation would fail to resolve. Tags are not relevant to this
                # test's invariant.
                "~trainer.logger.tags",
            ],
        )

    with pytest.raises(KeyboardInterrupt):
        pretrain.__wrapped__(cfg)

    saved_fp = output_dir / "loggers" / "wandb_run_id.txt"
    assert saved_fp.is_file(), (
        "wandb_run_id.txt was not written before fit was interrupted — save_logger_run_ids "
        "fires only after trainer.fit returns, so an interrupted run never persists its id "
        "(#152). The fix is to persist on on_train_start via a Lightning Callback."
    )
    saved_id = saved_fp.read_text().strip()
    assert saved_id, f"wandb_run_id.txt was written but is empty (contents: {saved_id!r})"


# ---------------------------------------------------------------------------
# Issue #131 — generate's saved run id must be readable on the next resume
# ---------------------------------------------------------------------------


def test_apply_saved_logger_run_ids_prefers_run_dir_over_fallback(tmp_path: Path):
    """Bug #131: ``apply_saved_logger_run_ids`` must read ``run_dir`` first, then fall back.

    ``generate_trajectories`` saves run ids to ``cfg.output_dir`` and restores them from
    ``cfg.model_initialization_dir``. With single-arg ``apply_saved_logger_run_ids``, the
    saved generation ids are never read back — they're an orphan write. The fix is a
    layered lookup: a saved id under ``run_dir`` wins over the same id under
    ``fallback_dir``.

    This test fails today because ``apply_saved_logger_run_ids`` does not accept a
    ``fallback_dir`` keyword.
    """
    primary = tmp_path / "generation_output"
    fallback = tmp_path / "training_output"

    (primary / "loggers").mkdir(parents=True)
    (fallback / "loggers").mkdir(parents=True)
    (primary / "loggers" / "wandb_run_id.txt").write_text("from-generation")
    (primary / "loggers" / "mlflow_run_id.txt").write_text("mlflow-from-generation")
    (fallback / "loggers" / "wandb_run_id.txt").write_text("from-training")
    (fallback / "loggers" / "mlflow_run_id.txt").write_text("mlflow-from-training")

    cfg = DictConfig(
        {
            "loggers": [
                {"_target_": "MLFlowLogger"},
                {"_target_": "WandbLogger"},
            ]
        }
    )

    utils.apply_saved_logger_run_ids(cfg, primary, fallback_dir=fallback)

    assert cfg.loggers[0]["run_id"] == "mlflow-from-generation"
    assert cfg.loggers[1]["id"] == "from-generation"


def test_apply_saved_logger_run_ids_falls_back_when_run_dir_empty(tmp_path: Path):
    """Bug #131 (companion): when ``run_dir`` has no saved id, apply falls back.

    First-ever generation resume: ``cfg.output_dir/loggers`` exists but is empty (no
    prior generation has run), so the helper must read from ``fallback_dir`` (the
    training dir) instead of failing closed. Under the fix this just works; under the
    current single-arg signature it raises ``TypeError`` because there's no
    ``fallback_dir`` parameter.
    """
    primary = tmp_path / "generation_output"
    fallback = tmp_path / "training_output"

    (primary / "loggers").mkdir(parents=True)
    (fallback / "loggers").mkdir(parents=True)
    (fallback / "loggers" / "wandb_run_id.txt").write_text("from-training")

    cfg = DictConfig({"loggers": [{"_target_": "WandbLogger"}]})

    utils.apply_saved_logger_run_ids(cfg, primary, fallback_dir=fallback)

    assert cfg.loggers[0]["id"] == "from-training"
    assert cfg.loggers[0]["resume"] == "allow"
