"""Integration tests for the logger-id save / apply helpers using real ``mlflow`` and ``wandb``.

These exercises run end-to-end against the actual Lightning logger classes — they're the
sanity check that our save/apply helpers remain compatible with real ``MLFlowLogger`` /
``WandbLogger`` attribute shapes across upstream versions, not just against the dummy
classes in ``tests/test_logger_helpers.py``.

**Offline by design**: both backends are configured to write to local files only (MLflow via
a ``file://`` tracking URI; WandB via ``WandbLogger(offline=True)`` plus ``WANDB_MODE=offline``
belt-and-braces). No remote servers are contacted. That keeps the tests runnable in CI
without credentials and without network flake.

**Gated by install**: ``pytest.importorskip`` on both modules means these tests are skipped
cleanly on a default install. To run them, install the extras:

```
uv sync --extra mlflow --extra wandb
```

Then run ``uv run pytest tests/test_logger_helpers_integration.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Guard the whole module behind availability of the optional deps. Done at module level so
# pytest collection doesn't even try to import Lightning logger classes when the extras
# aren't installed (avoids noisy import errors that look like test failures).
pytest.importorskip("mlflow")
pytest.importorskip("wandb")

from lightning.pytorch.loggers import MLFlowLogger, WandbLogger

import MEDS_EIC_AR.utils as utils

# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


def test_mlflow_save_and_apply_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real ``MLFlowLogger`` → save → apply round-trip, offline via a local ``file://`` URI.

    Creates a real MLflow run with a local tracking store, saves its id + uri via
    :func:`save_logger_run_ids`, then drives :func:`apply_saved_logger_run_ids` against a
    fresh ``OmegaConf`` config and asserts both ``run_id`` and ``tracking_uri`` are restored
    exactly — proving the helper is compatible with Lightning's real ``MLFlowLogger`` attribute
    layout (run_id, _tracking_uri).
    """
    # Quiet noisy mlflow/git integration attempts during test runs.
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    monkeypatch.setenv("GIT_PYTHON_REFRESH", "quiet")

    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    logger = MLFlowLogger(
        experiment_name="logger_helpers_integration",
        tracking_uri=tracking_uri,
    )
    # Force run creation so ``.run_id`` is populated (Lightning creates lazily on first log).
    logger.log_metrics({"sanity": 1.0}, step=0)
    assert logger.run_id, "MLFlowLogger.run_id should be populated after first log_metrics call"

    # Save under a dedicated pretrain-style dir.
    pretrain_dir = tmp_path / "pretrain_out"
    utils.save_logger_run_ids([logger], pretrain_dir)

    saved_run_id = (pretrain_dir / "loggers" / "mlflow_run_id.txt").read_text()
    saved_uri = (pretrain_dir / "loggers" / "mlflow_tracking_uri.txt").read_text()
    assert saved_run_id == str(logger.run_id)
    assert saved_uri == tracking_uri

    # Apply to a fresh config that mimics the default trainer/logger/mlflow.yaml shape —
    # populated tracking_uri pointing at a DIFFERENT (would-be-new) run's store. Helper must
    # override it with the saved uri because we're resuming.
    from omegaconf import DictConfig

    new_run_uri = f"file://{tmp_path / 'new_run_mlruns'}"
    cfg = DictConfig(
        {
            "loggers": [
                {
                    "_target_": "lightning.pytorch.loggers.MLFlowLogger",
                    "tracking_uri": new_run_uri,
                    "experiment_name": "logger_helpers_integration",
                }
            ]
        }
    )
    utils.apply_saved_logger_run_ids(cfg, pretrain_dir)

    assert cfg.loggers[0]["run_id"] == str(logger.run_id)
    assert cfg.loggers[0]["tracking_uri"] == tracking_uri


# ---------------------------------------------------------------------------
# WandB integration
# ---------------------------------------------------------------------------


def test_wandb_save_and_apply_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real ``WandbLogger`` → save → apply round-trip, fully offline.

    ``WandbLogger(offline=True)`` plus ``WANDB_MODE=offline`` / ``WANDB_DIR`` env overrides
    keep all state on local disk with no server contact. ``WandbLogger.experiment.id`` is
    what our save path reads, so this test is the real attribute-shape check.
    """
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb_state"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb_cfg"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb_cache"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    # WandB disables its own offline warning banner if this is set.
    monkeypatch.setenv("WANDB_DISABLE_GIT", "true")

    logger = WandbLogger(
        project="logger_helpers_integration",
        save_dir=str(tmp_path / "wandb_save"),
        offline=True,
    )
    # Touch .experiment to actually initialize the run (WandbLogger is lazy).
    _ = logger.experiment
    assert logger.experiment.id, "WandbLogger.experiment.id should be populated after init"

    pretrain_dir = tmp_path / "pretrain_out"
    utils.save_logger_run_ids([logger], pretrain_dir)

    saved_id = (pretrain_dir / "loggers" / "wandb_run_id.txt").read_text()
    assert saved_id == str(logger.experiment.id)

    from omegaconf import DictConfig

    cfg = DictConfig({"loggers": [{"_target_": "lightning.pytorch.loggers.WandbLogger", "offline": True}]})
    utils.apply_saved_logger_run_ids(cfg, pretrain_dir)

    assert cfg.loggers[0]["id"] == str(logger.experiment.id)
    assert cfg.loggers[0]["resume"] == "allow"

    # Explicit teardown — WandB buffers files on exit; with offline mode there's no upload,
    # but we call finish() anyway so pytest doesn't print noisy shutdown-hook warnings.
    logger.experiment.finish()


# ---------------------------------------------------------------------------
# Detector regression: is_mlflow_logger / is_wandb_logger against real classes
# ---------------------------------------------------------------------------


def test_is_logger_detectors_match_real_classes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``is_mlflow_logger`` / ``is_wandb_logger`` must return True for the real classes.

    These are the detectors used in ``save_logger_run_ids`` to decide which file to write.
    The docstring-level doctests use dummy classes; this test closes the gap against the
    real Lightning logger implementations (useful when e.g. a Lightning upgrade renames the
    logger class or changes its module path).
    """
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb_state"))
    monkeypatch.setenv("WANDB_SILENT", "true")

    mlflow_logger = MLFlowLogger(tracking_uri=f"file://{tmp_path / 'mlruns'}")
    wandb_logger = WandbLogger(
        project="detector_test",
        save_dir=str(tmp_path / "wandb_save"),
        offline=True,
    )

    assert utils.is_mlflow_logger(mlflow_logger)
    assert not utils.is_mlflow_logger(wandb_logger)
    assert utils.is_wandb_logger(wandb_logger)
    assert not utils.is_wandb_logger(mlflow_logger)


# ---------------------------------------------------------------------------
# Reliability: corrupted saved-id files must not crash the restore
# ---------------------------------------------------------------------------


def test_corrupted_saved_id_files_do_not_crash(tmp_path: Path):
    """Empty / whitespace-only saved-id files must be treated as "no saved id", not as empty strings.

    Reliability guard requested in PR #73 review: a zero-byte ``mlflow_run_id.txt`` left over
    from a failed pretrain must not poison the next run by restoring ``run_id=""`` (which MLflow
    would reject). The helper reads via ``_read_saved_id`` which returns ``None`` for
    empty/whitespace content.
    """
    del tmp_path  # unused
    assert importlib.util.find_spec("omegaconf") is not None  # sanity
    from tempfile import TemporaryDirectory

    from omegaconf import DictConfig

    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        log_dir = run_dir / "loggers"
        log_dir.mkdir()
        (log_dir / "mlflow_run_id.txt").write_text("")
        (log_dir / "wandb_run_id.txt").write_text("   \n\t  \n")

        cfg = DictConfig(
            {
                "loggers": [
                    {"_target_": "MLFlowLogger"},
                    {"_target_": "WandbLogger"},
                ]
            }
        )
        utils.apply_saved_logger_run_ids(cfg, run_dir)

        # Neither logger should have been touched, because the saved files were blank.
        assert "run_id" not in cfg.loggers[0]
        assert "id" not in cfg.loggers[1]
        assert "resume" not in cfg.loggers[1]
