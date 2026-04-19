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


def test_apply_saved_logger_run_ids_overrides_default_mlflow_tracking_uri(tmp_path):
    """Saved tracking_uri must override a repo-default tracking_uri when a saved run_id is applied.

    Regression guard for the case Copilot flagged on PR #73: the default
    ``configs/trainer/logger/mlflow.yaml`` sets ``tracking_uri: ${log_dir}/mlflow/mlruns``,
    which post-interpolation is a truthy non-empty string. A naive "only restore if absent"
    rule would then never fire, and a resumed ``run_id`` would point at the *new* run's
    tracking store — 404 on resume, or quiet log-to-wrong-store. The current rule is:
    when a saved ``run_id`` is applied, the saved ``tracking_uri`` overrides whatever
    the current config contained. This test locks that rule in against both the default
    config's tracking_uri and a caller-set tracking_uri.
    """
    cfg = DictConfig(
        {
            "loggers": [
                {
                    "_target_": "MLFlowLogger",
                    "tracking_uri": str(tmp_path / "current-output" / "mlruns"),
                },
                {"_target_": "WandbLogger"},
            ]
        }
    )

    log_dir = tmp_path / "loggers"
    log_dir.mkdir()
    (log_dir / "mlflow_run_id.txt").write_text("abc")
    (log_dir / "mlflow_tracking_uri.txt").write_text(str(tmp_path / "original" / "mlruns"))
    (log_dir / "wandb_run_id.txt").write_text("xyz")

    utils.apply_saved_logger_run_ids(cfg, tmp_path)

    assert cfg.loggers[0]["run_id"] == "abc"
    assert cfg.loggers[0]["tracking_uri"] == str(tmp_path / "original" / "mlruns")
    assert cfg.loggers[1]["id"] == "xyz"
    assert cfg.loggers[1]["resume"] == "allow"


def test_apply_saved_logger_run_ids_handles_disabled_logger(tmp_path):
    """``trainer.logger=false`` / ``trainer.logger=null`` must not crash.

    Lightning accepts ``logger: bool | Logger | Iterable[Logger]``. A Hydra user who disables
    logging via ``trainer.logger=false`` would have previously tripped an ``AttributeError``
    inside this helper when it tried ``logger_cfg.get("_target_", "")`` on a bool. The
    normalization step now skips non-mapping entries silently.
    """
    log_dir = tmp_path / "loggers"
    log_dir.mkdir()
    (log_dir / "mlflow_run_id.txt").write_text("abc")

    # Case: single logger disabled via bool.
    cfg_bool = DictConfig({"logger": False})
    utils.apply_saved_logger_run_ids(cfg_bool, tmp_path)  # does not raise

    # Case: single logger disabled via null.
    cfg_null = DictConfig({"logger": None})
    utils.apply_saved_logger_run_ids(cfg_null, tmp_path)

    # Case: ``loggers`` key holds a bool rather than a list.
    cfg_list_bool = DictConfig({"loggers": False})
    utils.apply_saved_logger_run_ids(cfg_list_bool, tmp_path)

    # Case: mixed — real dict entry alongside a null entry in the list; real entry still patched.
    cfg_mixed = DictConfig({"loggers": [None, {"_target_": "MLFlowLogger"}]})
    utils.apply_saved_logger_run_ids(cfg_mixed, tmp_path)
    assert cfg_mixed.loggers[1]["run_id"] == "abc"


def test_apply_saved_logger_run_ids_preserves_explicit_run_id(tmp_path):
    """If the caller explicitly set ``run_id``, neither ``run_id`` nor ``tracking_uri`` is restored.

    This is the escape hatch for "log a fresh run into a different MLflow store." Without it,
    a caller who wants a new run in a new store would always have their ``tracking_uri`` silently
    overwritten by the saved one when a ``mlflow_tracking_uri.txt`` sits in the run_dir.
    """
    cfg = DictConfig(
        {
            "loggers": [
                {
                    "_target_": "MLFlowLogger",
                    "run_id": "fresh",
                    "tracking_uri": str(tmp_path / "new-store" / "mlruns"),
                },
            ]
        }
    )

    log_dir = tmp_path / "loggers"
    log_dir.mkdir()
    (log_dir / "mlflow_run_id.txt").write_text("abc")
    (log_dir / "mlflow_tracking_uri.txt").write_text(str(tmp_path / "original" / "mlruns"))

    utils.apply_saved_logger_run_ids(cfg, tmp_path)

    assert cfg.loggers[0]["run_id"] == "fresh"
    assert cfg.loggers[0]["tracking_uri"] == str(tmp_path / "new-store" / "mlruns")


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

    # Rank-zero happy path: one epoch with two batches → all mean/min/max/std metrics written.
    rec = RecordingLogger()
    cb = GenerationSpeedLogger()
    trainer = FakeTrainer(loggers=[rec])
    cb.on_predict_start(trainer, pl_module=None)
    cb.on_predict_epoch_start(trainer, pl_module=None)
    cb.on_predict_batch_start(trainer, pl_module=None, batch=None, batch_idx=0)
    cb.on_predict_batch_end(trainer, pl_module=None, outputs=None, batch=None, batch_idx=0)
    cb.on_predict_batch_start(trainer, pl_module=None, batch=None, batch_idx=1)
    cb.on_predict_batch_end(trainer, pl_module=None, outputs=None, batch=None, batch_idx=1)
    cb.on_predict_epoch_end(trainer, pl_module=None)
    cb.on_predict_end(trainer, pl_module=None)

    assert len(rec.calls) == 1
    metrics, step = rec.calls[0]
    expected_keys = {
        "predict/total_time_sec",
        "predict/num_batches",
        "predict/epoch_time_sec_mean",
        "predict/epoch_time_sec_min",
        "predict/epoch_time_sec_max",
        "predict/epoch_time_sec_std",
        "predict/batch_time_sec_mean",
        "predict/batch_time_sec_min",
        "predict/batch_time_sec_max",
        "predict/batch_time_sec_std",
    }
    assert set(metrics) == expected_keys
    assert metrics["predict/num_batches"] == 2
    # All time-like values are non-negative; min <= mean <= max for both granularities.
    for prefix in ("predict/epoch_time_sec", "predict/batch_time_sec"):
        assert metrics[f"{prefix}_min"] <= metrics[f"{prefix}_mean"] <= metrics[f"{prefix}_max"]
        assert metrics[f"{prefix}_std"] >= 0.0
    # Single-epoch case collapses epoch std to 0.0 (population stdev of one sample is 0).
    assert metrics["predict/epoch_time_sec_std"] == 0.0
    assert step == 42

    # Non-rank-zero path: nothing logged even though epoch/batch times were recorded.
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
