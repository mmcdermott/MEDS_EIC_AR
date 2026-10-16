"""Growing-horizon weight averaging for Anytime pretraining.

Anytime pretraining (Meterez et al., 2026, https://arxiv.org/abs/2602.03702) pairs a horizon-free learning
rate schedule (warmup, then constant) with an exponential moving average of the weights whose horizon grows
with training age. At optimizer step ``t`` the average is updated as

    beta_t = 2 ** (-k / t)
    ema_t = beta_t * ema_{t-1} + (1 - beta_t) * theta_t

where ``theta_t`` is the raw parameter vector after the ``t``-th optimizer step. The instantaneous half-life
of the average is ``t / k`` steps, so the averaging window scales with the run instead of being fixed in
advance. Nothing here depends on the final step budget, which is what allows a finished run to be resumed
with a larger ``trainer.max_steps``.

Two sets of weights are kept explicitly distinct:

- **Raw weights** are the module's parameters: the ones the optimizer updates, and the ones a resume
  continues from.
- **EMA weights** live in :class:`AnytimeEMA`'s callback state. They are swapped into the module for
  validation (and so drive checkpoint selection) and are what :meth:`MEICARModule.load_from_checkpoint`
  selects by default for inference and export when a checkpoint carries them.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, SingleDeviceStrategy

logger = logging.getLogger(__name__)

ANYTIME_EMA_STATE_KEY = "AnytimeEMA"


def ema_beta(t: int, k: float) -> float:
    """Returns the EMA decay ``2 ** (-k / t)`` applied at optimizer step ``t``.

    Args:
        t: The 1-indexed optimizer step being averaged in.
        k: The EMA-k constant. Larger values average over a shorter window.

    Returns:
        The weight kept on the previous average.

    Raises:
        ValueError: If ``t`` is not positive.

    Examples:
        >>> ema_beta(1, 25.0)
        2.9802322387695312e-08
        >>> ema_beta(25, 25.0)
        0.5
        >>> round(ema_beta(1000, 25.0), 6)
        0.982821

    The decay at step ``t`` corresponds to a half-life of ``t / k`` steps:

        >>> round(ema_beta(1000, 25.0) ** (1000 / 25.0), 12)
        0.5
        >>> ema_beta(0, 25.0)
        Traceback (most recent call last):
            ...
        ValueError: The EMA step t must be a positive optimizer step; got 0.
    """
    if t < 1:
        raise ValueError(f"The EMA step t must be a positive optimizer step; got {t}.")
    return 2.0 ** (-k / t)


def _check_ema_matches(ema: dict[str, torch.Tensor], params: dict[str, torch.nn.Parameter]) -> None:
    """Raises if ``ema`` does not cover exactly ``params`` with matching shapes and finite values.

    Examples:
        >>> params = {"w": torch.nn.Parameter(torch.zeros(2, 3))}
        >>> _check_ema_matches({"w": torch.ones(2, 3)}, params)
        >>> _check_ema_matches({"w": torch.ones(3, 2)}, params)
        Traceback (most recent call last):
            ...
        ValueError: EMA tensor 'w' has shape (3, 2) but the model parameter has shape (2, 3).
        >>> _check_ema_matches({"v": torch.ones(2, 3)}, params)
        Traceback (most recent call last):
            ...
        ValueError: EMA parameters do not match the model: missing ['w'], unexpected ['v'].
        >>> _check_ema_matches({"w": torch.full((2, 3), float("nan"))}, params)
        Traceback (most recent call last):
            ...
        ValueError: EMA tensor 'w' contains non-finite values.
    """
    if ema.keys() != params.keys():
        missing = sorted(params.keys() - ema.keys())
        unexpected = sorted(ema.keys() - params.keys())
        raise ValueError(
            f"EMA parameters do not match the model: missing {missing}, unexpected {unexpected}."
        )
    for name, p in params.items():
        e = ema[name]
        if e.shape != p.shape:
            raise ValueError(
                f"EMA tensor '{name}' has shape {tuple(e.shape)} but the model parameter has shape "
                f"{tuple(p.shape)}."
            )
        if not bool(torch.isfinite(e).all()):
            raise ValueError(f"EMA tensor '{name}' contains non-finite values.")


def load_ema_weights(module: torch.nn.Module, ema_state: dict[str, Any]) -> None:
    """Copies the EMA weights stored in an :class:`AnytimeEMA` callback state into ``module``.

    Args:
        module: The module whose parameters are overwritten in place.
        ema_state: The callback state, as saved under ``checkpoint["callbacks"][ANYTIME_EMA_STATE_KEY]``.

    Raises:
        ValueError: If the EMA tensors do not match the module's parameters.

    Examples:
        >>> module = torch.nn.Linear(2, 1)
        >>> state = {"k": 25.0, "age": 3, "ema": {"weight": torch.ones(1, 2), "bias": torch.zeros(1)}}
        >>> load_ema_weights(module, state)
        >>> module.weight
        Parameter containing:
        tensor([[1., 1.]], requires_grad=True)
    """
    params = dict(module.named_parameters())
    _check_ema_matches(ema_state["ema"], params)
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(ema_state["ema"][name])


class AnytimeEMA(Callback):
    """Maintains a growing-horizon EMA of the module's weights (EMA-``k``) for Anytime pretraining.

    The average is updated once per optimizer step, after the step, using ``t = trainer.global_step``; under
    gradient accumulation the batches that do not step the optimizer leave it untouched. It is kept in fp32
    regardless of the parameters' dtype. During validation the EMA weights are swapped into the module and
    the raw weights are restored bit-exactly afterwards (also when validation raises), so validation metrics
    and checkpoint selection reflect the averaged model while checkpoints and training continue from the raw
    weights.

    The EMA tensors and their age are saved as callback state in every checkpoint. On resume the callback
    refuses to continue if that state is missing or out of step with the restored ``global_step``, if the
    tensors no longer match the model, or if ``k`` changed, rather than silently starting a fresh average.

    Because a run that reaches ``trainer.max_steps`` between validation checks would otherwise leave
    ``last.ckpt`` behind the final step, the callback also re-saves ``last.ckpt`` when training ends, so a
    finished run can be extended from exactly where it stopped.

    Only single-device and DDP strategies are supported: both keep full, replicated parameters on every
    rank. Sharded strategies (FSDP, DeepSpeed, model parallel) are rejected at setup.

    Args:
        k: The EMA-k constant; the averaging half-life at step ``t`` is ``t / k`` steps.

    Raises:
        ValueError: If ``k`` is not positive.

    Examples:
        >>> AnytimeEMA(k=25).state_dict()
        {'k': 25.0, 'age': 0, 'ema': None}
        >>> AnytimeEMA(k=0)
        Traceback (most recent call last):
            ...
        ValueError: AnytimeEMA requires k > 0; got 0.
    """

    def __init__(self, k: float = 25.0):
        if not k > 0:
            raise ValueError(f"AnytimeEMA requires k > 0; got {k}.")
        self.k = float(k)
        self.age = 0
        self.ema: dict[str, torch.Tensor] | None = None
        self._raw_backup: dict[str, torch.Tensor] | None = None

    @property
    def state_key(self) -> str:
        return ANYTIME_EMA_STATE_KEY

    def state_dict(self) -> dict[str, Any]:
        return {"k": self.k, "age": self.age, "ema": self.ema}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["k"] != self.k:
            raise ValueError(
                f"The checkpoint's EMA was built with k={state_dict['k']}, but this run uses k={self.k}. "
                "Changing k mid-run starts a different averaging trajectory; refusing to resume."
            )
        self.age = int(state_dict["age"])
        self.ema = state_dict["ema"]

    @staticmethod
    def _params(pl_module: L.LightningModule) -> dict[str, torch.nn.Parameter]:
        return dict(pl_module.named_parameters())

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        strategy = trainer.strategy
        supported = isinstance(strategy, SingleDeviceStrategy) or (
            isinstance(strategy, DDPStrategy) and not isinstance(strategy, DeepSpeedStrategy)
        )
        if not supported:
            raise ValueError(
                f"AnytimeEMA supports single-device and DDP training only; got {type(strategy).__name__}. "
                "Sharded strategies hold only part of each parameter per rank, which this callback does not "
                "average or checkpoint correctly."
            )

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        params = self._params(pl_module)
        if self.ema is None:
            # Either a fresh run or a checkpoint without EMA state; the latter is caught in on_train_start
            # once the trainer's step counter has been restored.
            self.ema = {n: p.detach().to(torch.float32, copy=True) for n, p in params.items()}
            return

        _check_ema_matches(self.ema, params)
        self.ema = {n: self.ema[n].to(device=p.device, dtype=torch.float32) for n, p in params.items()}

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.age != trainer.global_step:
            raise ValueError(
                f"The EMA has absorbed {self.age} optimizer steps but training is resuming at step "
                f"{trainer.global_step}. The averaged weights would be out of step with the raw weights "
                "(for example a checkpoint saved without AnytimeEMA, or an EMA that was reset); refusing to "
                "continue."
            )

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        t = trainer.global_step
        if t == self.age:
            return  # gradient accumulation: no optimizer step on this batch
        if t != self.age + 1:
            raise RuntimeError(
                f"AnytimeEMA expected optimizer step {self.age + 1} but the trainer is at step {t}; an "
                "optimizer step was not averaged."
            )

        params = self._params(pl_module)
        ema = [self.ema[n] for n in params]
        raw = [p.detach().to(torch.float32) for p in params.values()]
        with torch.no_grad():
            torch._foreach_lerp_(ema, raw, 1.0 - ema_beta(t, self.k))
        self.age = t

        pl_module.log("anytime/ema_age", float(t), on_step=True, on_epoch=False)
        pl_module.log("anytime/ema_half_life", t / self.k, on_step=True, on_epoch=False)

    def _swap_in_ema(self, pl_module: L.LightningModule) -> None:
        if self.ema is None or self._raw_backup is not None:
            return
        params = self._params(pl_module)
        self._raw_backup = {n: p.detach().clone() for n, p in params.items()}
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(self.ema[n])

    def _restore_raw(self, pl_module: L.LightningModule) -> None:
        if self._raw_backup is None:
            return
        with torch.no_grad():
            for n, p in self._params(pl_module).items():
                p.copy_(self._raw_backup[n])
        self._raw_backup = None

    @contextmanager
    def ema_weights(self, pl_module: L.LightningModule) -> Iterator[None]:
        """Temporarily loads the EMA weights into ``pl_module``, restoring the raw weights on exit."""
        self._swap_in_ema(pl_module)
        try:
            yield
        finally:
            self._restore_raw(pl_module)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_in_ema(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Lightning runs ModelCheckpoint callbacks after all others, so the raw weights are back in place
        # before a checkpoint triggered by this validation pass is written.
        self._restore_raw(pl_module)

    def on_exception(
        self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException
    ) -> None:
        self._restore_raw(pl_module)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        ckpt_cb = trainer.checkpoint_callback
        if not isinstance(ckpt_cb, ModelCheckpoint) or not ckpt_cb.save_last or ckpt_cb.dirpath is None:
            return
        last_fp = Path(ckpt_cb.dirpath) / f"{ckpt_cb.CHECKPOINT_NAME_LAST}{ckpt_cb.FILE_EXTENSION}"
        trainer.save_checkpoint(last_fp)
        logger.info(f"Saved {last_fp} at the final optimizer step {trainer.global_step}.")
