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


def test_apply_saved_logger_run_ids(tmp_path, monkeypatch):
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


def test_is_wandb_logger_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lightning.pytorch.loggers" and "WandbLogger" in fromlist:
            raise ImportError
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert utils.is_wandb_logger(DummyWandbLogger()) is False
