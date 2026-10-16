"""Tests for Anytime pretraining: the growing-horizon EMA callback and horizon-free resume.

These use a tiny deterministic Lightning module rather than the MEICAR model so that runs are bit-reproducible
and the EMA recurrence can be checked against a hand-computed reference; the CLI path is covered at the end.
"""

import shutil
import subprocess
from pathlib import Path

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_constant_schedule_with_warmup

from MEDS_EIC_AR.training import AnytimeEMA, MEICARModule
from MEDS_EIC_AR.training.anytime import ANYTIME_EMA_STATE_KEY, ema_beta

N_BATCHES = 4
BATCH_SIZE = 2
PEAK_LR = 0.1
WARMUP = 3


class TinyModule(L.LightningModule):
    def __init__(self, fail_validation: bool = False):
        super().__init__()
        torch.manual_seed(0)
        self.layer = torch.nn.Linear(3, 1)
        self.fail_validation = fail_validation

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        if self.fail_validation:
            raise RuntimeError("validation failed")
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.layer(x), y)
        self.log("tuning/loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=PEAK_LR)
        sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=WARMUP)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def _loader() -> DataLoader:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(N_BATCHES * BATCH_SIZE, 3, generator=g)
    y = torch.randn(N_BATCHES * BATCH_SIZE, 1, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=False)


def _trainer(tmp_path: Path, max_steps: int, callbacks: list[Callback], **kwargs) -> L.Trainer:
    return L.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        callbacks=callbacks,
        **kwargs,
    )


def _checkpointer(tmp_path: Path) -> ModelCheckpoint:
    return ModelCheckpoint(dirpath=tmp_path / "checkpoints", save_last=True, monitor="tuning/loss")


class RecordRawParams(Callback):
    """Records the raw parameters after every optimizer step, for a hand-computed EMA reference."""

    def __init__(self):
        self.snapshots: list[dict[str, torch.Tensor]] = []
        self._last_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step != self._last_step:
            self._last_step = trainer.global_step
            self.snapshots.append({n: p.detach().clone() for n, p in pl_module.named_parameters()})


def test_ema_follows_recurrence_once_per_optimizer_step(tmp_path: Path):
    """With gradient accumulation, the EMA advances once per optimizer step and matches the recurrence."""
    ema_cb = AnytimeEMA(k=2.0)
    recorder = RecordRawParams()
    module = TinyModule()
    initial = {n: p.detach().clone() for n, p in module.named_parameters()}

    trainer = _trainer(tmp_path, max_steps=6, callbacks=[recorder, ema_cb], accumulate_grad_batches=2)
    trainer.fit(module, train_dataloaders=_loader())

    assert trainer.global_step == 6
    assert ema_cb.age == 6
    assert len(recorder.snapshots) == 6

    expected = {n: t.float() for n, t in initial.items()}
    for t, snap in enumerate(recorder.snapshots, start=1):
        beta = ema_beta(t, 2.0)
        expected = {n: beta * expected[n] + (1 - beta) * snap[n] for n in expected}
    for n, e in expected.items():
        torch.testing.assert_close(ema_cb.ema[n], e)


def test_half_life_grows_with_age():
    half_lives = [t / 25.0 for t in (10, 100, 1000)]
    assert half_lives == sorted(half_lives)
    for t, h in zip((10, 100, 1000), half_lives, strict=True):
        assert ema_beta(t, 25.0) ** h == pytest.approx(0.5)


class CheckValidationWeights(Callback):
    """Asserts that validation sees the EMA weights and that training sees the raw weights afterwards."""

    def __init__(self, ema_cb: AnytimeEMA):
        self.ema_cb = ema_cb
        self.raw_before: dict[str, torch.Tensor] | None = None
        self.n_checked = 0

    def on_validation_start(self, trainer, pl_module):
        # Registered before AnytimeEMA, so this runs before the swap.
        self.raw_before = {n: p.detach().clone() for n, p in pl_module.named_parameters()}

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        for n, p in pl_module.named_parameters():
            torch.testing.assert_close(p.detach(), self.ema_cb.ema[n], rtol=0, atol=0)

    def on_validation_end(self, trainer, pl_module):
        # Registered before AnytimeEMA, so the raw weights are restored only after this runs; check them on
        # the next training batch instead.
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.raw_before is None:
            return
        for n, p in pl_module.named_parameters():
            assert torch.equal(p.detach(), self.raw_before[n]), f"raw weight {n} was not restored exactly"
        self.raw_before = None
        self.n_checked += 1


def test_validation_uses_ema_and_restores_raw_weights_exactly(tmp_path: Path):
    ema_cb = AnytimeEMA(k=2.0)
    checker = CheckValidationWeights(ema_cb)
    trainer = _trainer(tmp_path, max_steps=8, callbacks=[checker, ema_cb], val_check_interval=2)
    trainer.fit(TinyModule(), train_dataloaders=_loader(), val_dataloaders=_loader())

    # Sanity check plus validation after steps 2 and 6 are followed by a training batch.
    assert checker.n_checked >= 2


def test_failed_validation_restores_raw_weights(tmp_path: Path):
    ema_cb = AnytimeEMA(k=2.0)
    module = TinyModule(fail_validation=True)
    raw = {n: p.detach().clone() for n, p in module.named_parameters()}
    # Make the EMA visibly different from the raw weights so a missed restore would show.
    ema_cb.ema = {n: t.float() + 1.0 for n, t in raw.items()}
    trainer = _trainer(tmp_path, max_steps=2, callbacks=[ema_cb])

    with pytest.raises(RuntimeError, match="validation failed"):
        trainer.fit(module, train_dataloaders=_loader(), val_dataloaders=_loader())

    for n, p in module.named_parameters():
        assert torch.equal(p.detach(), raw[n])


def test_ema_weights_context_manager_restores_on_error():
    ema_cb = AnytimeEMA(k=2.0)
    module = TinyModule()
    raw = {n: p.detach().clone() for n, p in module.named_parameters()}
    ema_cb.ema = {n: t.float() + 1.0 for n, t in raw.items()}

    with pytest.raises(KeyError), ema_cb.ema_weights(module):
        assert torch.equal(module.layer.weight.detach(), ema_cb.ema["layer.weight"])
        raise KeyError("boom")
    for n, p in module.named_parameters():
        assert torch.equal(p.detach(), raw[n])


def test_state_dict_round_trip():
    src = AnytimeEMA(k=5.0)
    src.age = 17
    src.ema = {"w": torch.randn(2, 2)}
    dst = AnytimeEMA(k=5.0)
    dst.load_state_dict(src.state_dict())
    assert dst.age == 17
    assert torch.equal(dst.ema["w"], src.ema["w"])

    with pytest.raises(ValueError, match=r"k=5\.0, but this run uses k=25\.0"):
        AnytimeEMA(k=25.0).load_state_dict(src.state_dict())


def _train_and_save(tmp_path: Path, max_steps: int, k: float = 2.0) -> Path:
    trainer = _trainer(
        tmp_path,
        max_steps=max_steps,
        callbacks=[AnytimeEMA(k=k), _checkpointer(tmp_path)],
        val_check_interval=3,
    )
    trainer.fit(TinyModule(), train_dataloaders=_loader(), val_dataloaders=_loader())
    return tmp_path / "checkpoints" / "last.ckpt"


def test_finished_run_saves_last_checkpoint_at_final_step(tmp_path: Path):
    # Validation runs every 3 steps, so without the train-end save last.ckpt would stop at step 3.
    ckpt = torch.load(_train_and_save(tmp_path, max_steps=5), weights_only=False)
    assert ckpt["global_step"] == 5
    assert ckpt["callbacks"][ANYTIME_EMA_STATE_KEY]["age"] == 5


def test_resume_rejects_mismatched_ema_age(tmp_path: Path):
    ckpt_fp = _train_and_save(tmp_path, max_steps=4)
    ckpt = torch.load(ckpt_fp, weights_only=False)
    ckpt["callbacks"][ANYTIME_EMA_STATE_KEY]["age"] = 0
    torch.save(ckpt, ckpt_fp)

    trainer = _trainer(tmp_path, max_steps=8, callbacks=[AnytimeEMA(k=2.0)])
    with pytest.raises(ValueError, match="absorbed 0 optimizer steps but training is resuming at step 4"):
        trainer.fit(TinyModule(), train_dataloaders=_loader(), ckpt_path=ckpt_fp)


def test_resume_rejects_checkpoint_without_ema(tmp_path: Path):
    ckpt_fp = _train_and_save(tmp_path, max_steps=4)
    ckpt = torch.load(ckpt_fp, weights_only=False)
    del ckpt["callbacks"][ANYTIME_EMA_STATE_KEY]
    torch.save(ckpt, ckpt_fp)

    trainer = _trainer(tmp_path, max_steps=8, callbacks=[AnytimeEMA(k=2.0)])
    with pytest.raises(ValueError, match="training is resuming at step 4"):
        trainer.fit(TinyModule(), train_dataloaders=_loader(), ckpt_path=ckpt_fp)


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (lambda ema: ema.pop("layer.bias"), "missing \\['layer.bias'\\]"),
        (lambda ema: ema.update({"layer.bias": torch.zeros(2)}), "has shape \\(2,\\)"),
        (lambda ema: ema["layer.bias"].fill_(float("inf")), "non-finite"),
    ],
)
def test_resume_rejects_bad_ema_tensors(tmp_path: Path, corrupt, match: str):
    ckpt_fp = _train_and_save(tmp_path, max_steps=4)
    ckpt = torch.load(ckpt_fp, weights_only=False)
    corrupt(ckpt["callbacks"][ANYTIME_EMA_STATE_KEY]["ema"])
    torch.save(ckpt, ckpt_fp)

    trainer = _trainer(tmp_path, max_steps=8, callbacks=[AnytimeEMA(k=2.0)])
    with pytest.raises(ValueError, match=match):
        trainer.fit(TinyModule(), train_dataloaders=_loader(), ckpt_path=ckpt_fp)


def test_rejects_sharded_strategy():
    # FSDP cannot run on a CPU-only trainer, so call the setup hook directly with a stub trainer.
    from types import SimpleNamespace

    from lightning.pytorch.strategies import FSDPStrategy

    with pytest.raises(ValueError, match="single-device and DDP training only; got FSDPStrategy"):
        AnytimeEMA().setup(SimpleNamespace(strategy=FSDPStrategy()), TinyModule(), "fit")


def test_finish_then_extend_continues_the_same_state(tmp_path: Path):
    """Training to N, finishing, and resuming to N + M continues every piece of Anytime state exactly.

    This does not compare against an uninterrupted N + M run: on resume Lightning (2.5) replays one batch
    from a fresh iterator under the finished epoch's progress counters, so the split run sees a different
    data stream from the uninterrupted one. That affects any Lightning resume, Anytime or not. What the
    Anytime workflow needs, and what is asserted bit-exactly here, is that nothing is reset or re-derived:
    raw weights, EMA weights and age, optimizer moments, scheduler position and step all carry over, warmup
    does not restart, and the EMA keeps following ``beta_t = 2^(-k / t)`` from ``t = N + 1``.
    """
    n, m, k = N_BATCHES, 2 * N_BATCHES, 2.0

    ema_1 = AnytimeEMA(k=k)
    module_1 = TinyModule()
    trainer_1 = _trainer(
        tmp_path, max_steps=n, callbacks=[ema_1, _checkpointer(tmp_path)], val_check_interval=3
    )
    trainer_1.fit(module_1, train_dataloaders=_loader(), val_dataloaders=_loader())

    last_ckpt = tmp_path / "checkpoints" / "last.ckpt"
    saved = torch.load(last_ckpt, weights_only=False)
    assert saved["global_step"] == n
    assert saved["callbacks"][ANYTIME_EMA_STATE_KEY]["age"] == n
    for name, p in module_1.named_parameters():
        assert torch.equal(saved["state_dict"][name], p.detach())
        assert torch.equal(saved["callbacks"][ANYTIME_EMA_STATE_KEY]["ema"][name], ema_1.ema[name])
    end_of_phase_1 = trainer_1.optimizers[0].state_dict()
    assert saved["optimizer_states"][0]["param_groups"] == end_of_phase_1["param_groups"]
    for idx, state in end_of_phase_1["state"].items():
        for key, val in state.items():
            assert torch.equal(saved["optimizer_states"][0]["state"][idx][key], val)
    assert saved["lr_schedulers"][0] == trainer_1.lr_scheduler_configs[0].scheduler.state_dict()

    class CheckRestoredAndRecord(Callback):
        """Checks the restored state before the first resumed step, then records LRs and raw weights."""

        def __init__(self):
            self.lrs: list[float] = []
            self.snapshots: list[dict[str, torch.Tensor]] = []
            self._last_step = n

        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == n
            for name, p in pl_module.named_parameters():
                assert torch.equal(p.detach(), saved["state_dict"][name]), f"raw {name} not restored"
            restored = trainer.optimizers[0].state_dict()
            for idx, state in end_of_phase_1["state"].items():
                for key, val in state.items():
                    assert torch.equal(restored["state"][idx][key], val), f"optimizer {key} not restored"

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.lrs.append(trainer.optimizers[0].param_groups[0]["lr"])

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step != self._last_step:
                self._last_step = trainer.global_step
                self.snapshots.append({nm: p.detach().clone() for nm, p in pl_module.named_parameters()})

    ema_2 = AnytimeEMA(k=k)
    recorder = CheckRestoredAndRecord()
    trainer_2 = _trainer(
        tmp_path, max_steps=n + m, callbacks=[recorder, ema_2, _checkpointer(tmp_path)], val_check_interval=3
    )
    trainer_2.fit(TinyModule(), train_dataloaders=_loader(), val_dataloaders=_loader(), ckpt_path=last_ckpt)

    # Warmup does not restart: every resumed step runs at the peak LR.
    assert recorder.lrs == [pytest.approx(PEAK_LR)] * m
    assert trainer_2.global_step == n + m
    assert ema_2.age == n + m
    assert trainer_2.optimizers[0].state_dict()["state"][0]["step"] == n + m

    expected = dict(ema_1.ema)
    for t, snap in enumerate(recorder.snapshots, start=n + 1):
        beta = ema_beta(t, k)
        expected = {nm: beta * expected[nm] + (1 - beta) * snap[nm] for nm in expected}
    assert len(recorder.snapshots) == m
    for nm, e in expected.items():
        torch.testing.assert_close(ema_2.ema[nm], e)


def _run(cmd: list[str]) -> None:
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        raise AssertionError(f"{cmd} failed:\n{out.stdout.decode()}\n{out.stderr.decode()}")


def _pretrain_cmd(output_dir: Path, dataset_dir: Path, *overrides: str) -> list[str]:
    return [
        "MEICAR_pretrain",
        "--config-name=_demo_pretrain",
        f"output_dir={output_dir!s}",
        f"datamodule.config.tensorized_cohort_dir={dataset_dir!s}",
        "training_schedule=anytime_constant",
        "lightning_module.LR_scheduler.num_warmup_steps=2",
        "trainer.max_epochs=null",
        *overrides,
    ]


def test_cli_anytime_finish_then_extend(preprocessed_dataset: Path, tmp_path: Path):
    output_dir = tmp_path / "anytime_run"

    _run(_pretrain_cmd(output_dir, preprocessed_dataset, "trainer.max_steps=3"))
    first = torch.load(output_dir / "checkpoints" / "last.ckpt", weights_only=False)
    assert first["global_step"] == 3
    assert first["callbacks"][ANYTIME_EMA_STATE_KEY]["age"] == 3

    _run(_pretrain_cmd(output_dir, preprocessed_dataset, "trainer.max_steps=6", "do_resume=True"))
    second = torch.load(output_dir / "checkpoints" / "last.ckpt", weights_only=False)
    assert second["global_step"] == 6
    assert second["callbacks"][ANYTIME_EMA_STATE_KEY]["age"] == 6
    sched = second["lr_schedulers"][0]
    assert sched["last_epoch"] == 6
    peak_lr = second["optimizer_states"][0]["param_groups"][0]["initial_lr"]
    assert sched["_last_lr"] == [peak_lr, peak_lr], "LR should stay at its peak after the resumed run"

    # Inference/export loads the EMA weights by default and the raw weights on request.
    ema_module = MEICARModule.load_from_checkpoint(output_dir / "checkpoints" / "last.ckpt")
    raw_module = MEICARModule.load_from_checkpoint(output_dir / "checkpoints" / "last.ckpt", weights="raw")
    ema_state = second["callbacks"][ANYTIME_EMA_STATE_KEY]["ema"]
    raw_differs = False
    for name, p in ema_module.named_parameters():
        assert torch.equal(p.detach(), ema_state[name].to(p.dtype))
        raw_differs |= not torch.equal(p.detach(), raw_module.get_parameter(name).detach())
    assert raw_differs, "EMA and raw weights should differ after training"


def test_cli_anytime_resume_rejects_schedule_change(preprocessed_dataset: Path, tmp_path: Path):
    output_dir = tmp_path / "anytime_run"
    _run(_pretrain_cmd(output_dir, preprocessed_dataset, "trainer.max_steps=2"))
    shutil.rmtree(output_dir / ".logs")

    switch_to_cosine = [
        a
        for a in _pretrain_cmd(output_dir, preprocessed_dataset, "trainer.max_steps=4", "do_resume=True")
        if not a.startswith(("training_schedule=", "lightning_module.LR_scheduler"))
    ]
    with pytest.raises(AssertionError, match="does not match the input"):
        _run(switch_to_cosine)
