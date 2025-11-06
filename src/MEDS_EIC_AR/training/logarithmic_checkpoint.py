import time

from lightning.pytorch.callbacks import ModelCheckpoint


class LogarithmicModelCheckpoint(ModelCheckpoint):
    """Checkpoint callback with an optional logarithmic saving schedule."""

    def __init__(
        self,
        base: float = 2.0,
        start: int = 2,
        enable_logarithmic: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if base <= 1:
            raise ValueError(f"base must be > 1, got {base}")
        if start < 1:
            raise ValueError(f"start must be >= 1, got {start}")
        self.base = base
        self._next_step = start
        self.enable_logarithmic = enable_logarithmic

    def _update_next_step(self) -> None:
        self._next_step = max(int(self._next_step * self.base), self._next_step + 1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore[override]
        if not self.enable_logarithmic:
            return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if self._should_skip_saving_checkpoint(trainer):
            return

        # ``skip_batch`` determines whether we have reached the next scheduled
        # checkpointing step.
        skip_batch = trainer.global_step < self._next_step

        # ``skip_time`` replicates the behaviour of the parent class for the
        # ``train_time_interval`` argument.  If this interval is set, checkpoints
        # are written no more frequently than the given duration regardless of
        # step count.
        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = (
                prev_time_check is None or (now - prev_time_check) < train_time_interval.total_seconds()
            )
            skip_time = trainer.strategy.broadcast(skip_time)

        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

        if not skip_batch:
            # Advance to the next checkpointing step only when we actually saved
            # one due to reaching the scheduled step count.
            self._update_next_step()
