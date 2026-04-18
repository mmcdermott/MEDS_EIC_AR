import builtins

from omegaconf import DictConfig

import MEDS_EIC_AR.utils as utils


class DummyMLFlowLogger:
    def __init__(self, run_id="mlflow_run"):
        self.run_id = run_id


class DummyWandBExp:
    def __init__(self, id="wandb_run"):
        self.id = id


class DummyWandbLogger:
    def __init__(self, exp_id="wandb_run"):
        self.experiment = DummyWandBExp(exp_id)


def test_apply_saved_logger_run_ids(tmp_path):
    cfg = DictConfig(
        {
            "loggers": [
                {"_target_": "MLFlowLogger"},
                {"_target_": "WandbLogger"},
            ]
        }
    )

    log_dir = tmp_path / "loggers"
    log_dir.mkdir()
    (log_dir / "mlflow_run_id.txt").write_text("abc")
    (log_dir / "wandb_run_id.txt").write_text("xyz")

    utils.apply_saved_logger_run_ids(cfg, tmp_path)

    assert cfg.loggers[0]["run_id"] == "abc"
    assert cfg.loggers[1]["id"] == "xyz"
    assert cfg.loggers[1]["resume"] == "allow"


def test_save_logger_run_ids(tmp_path, monkeypatch):
    # Patch lightning logger classes so helpers recognise the dummy objects
    import importlib

    loggers_mod = importlib.import_module("lightning.pytorch.loggers")
    monkeypatch.setattr(loggers_mod, "MLFlowLogger", DummyMLFlowLogger, raising=False)
    monkeypatch.setattr(loggers_mod, "WandbLogger", DummyWandbLogger, raising=False)

    loggers = [DummyMLFlowLogger("mlflow-id"), DummyWandbLogger("wandb-id")]
    utils.save_logger_run_ids(loggers, tmp_path)

    assert (tmp_path / "loggers" / "mlflow_run_id.txt").read_text() == "mlflow-id"
    assert (tmp_path / "loggers" / "wandb_run_id.txt").read_text() == "wandb-id"


def test_generation_speed_logger_logs_once_on_rank_zero():
    """``GenerationSpeedLogger`` writes the avg-epoch-time metric exactly once per predict run.

    Also covers the rank-zero gate added in response to a Copilot review on PR #73: in distributed predict,
    non-rank-zero trainers must not log.
    """
    from MEDS_EIC_AR.training.callbacks import GenerationSpeedLogger

    class RecordingLogger:
        def __init__(self):
            self.calls: list[tuple[dict, int | None]] = []

        def log_metrics(self, metrics, step=None):
            self.calls.append((dict(metrics), step))

    class FakeTrainer:
        def __init__(self, loggers, is_global_zero=True, global_step=42):
            self.loggers = loggers
            self.is_global_zero = is_global_zero
            self.global_step = global_step

    # Rank-zero happy path: one epoch, one metric written.
    rec = RecordingLogger()
    cb = GenerationSpeedLogger()
    trainer = FakeTrainer(loggers=[rec])
    cb.on_predict_start(trainer, pl_module=None)
    cb.on_predict_epoch_start(trainer, pl_module=None)
    cb.on_predict_epoch_end(trainer, pl_module=None)
    cb.on_predict_end(trainer, pl_module=None)

    assert len(rec.calls) == 1
    metrics, step = rec.calls[0]
    assert set(metrics) == {"predict/avg_epoch_time_sec"}
    assert metrics["predict/avg_epoch_time_sec"] >= 0.0
    assert step == 42

    # Non-rank-zero path: nothing logged even though epoch times were recorded.
    rec2 = RecordingLogger()
    cb2 = GenerationSpeedLogger()
    trainer2 = FakeTrainer(loggers=[rec2], is_global_zero=False)
    cb2.on_predict_start(trainer2, pl_module=None)
    cb2.on_predict_epoch_start(trainer2, pl_module=None)
    cb2.on_predict_epoch_end(trainer2, pl_module=None)
    cb2.on_predict_end(trainer2, pl_module=None)

    assert rec2.calls == []


def test_is_wandb_logger_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lightning.pytorch.loggers" and "WandbLogger" in fromlist:
            raise ImportError
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert utils.is_wandb_logger(DummyWandbLogger()) is False
