import time
from collections.abc import Sequence

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger


class GenerationSpeedLogger(Callback):
    """Logs generation speed statistics during prediction."""

    def on_predict_start(self, trainer, pl_module) -> None:
        self._epoch_times: list[float] = []
        self._epoch_start: float | None = None

    def on_predict_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.perf_counter()

    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        if self._epoch_start is None:
            return
        self._epoch_times.append(time.perf_counter() - self._epoch_start)

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self._epoch_times:
            return
        avg_epoch_time = sum(self._epoch_times) / len(self._epoch_times)
        metrics = {"predict/avg_epoch_time_sec": avg_epoch_time}
        for logger in _ensure_logger_sequence(trainer.loggers):
            logger.log_metrics(metrics, step=trainer.global_step)


def _ensure_logger_sequence(loggers: Logger | Sequence[Logger] | None) -> Sequence[Logger]:
    if loggers is None:
        return []
    if isinstance(loggers, Logger):
        return [loggers]
    return list(loggers)
