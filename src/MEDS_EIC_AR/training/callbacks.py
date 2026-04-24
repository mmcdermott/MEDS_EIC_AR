import statistics
import time
from collections.abc import Sequence

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger


class GenerationSpeedLogger(Callback):
    """Logs generation speed statistics (mean + distribution) during prediction.

    Tracks two granularities:

    - **Per-epoch time**, emitted as ``predict/epoch_time_sec_{mean,min,max,std}``. Useful
      when a single ``trainer.predict`` call iterates multiple dataloaders (e.g., multiple
      evaluation splits); with only one epoch the min/max/std collapse to the mean / 0 and
      the signal is in the mean itself.
    - **Per-batch time**, emitted as ``predict/batch_time_sec_{mean,min,max,std}``. This is
      where the variance story actually shows up — batch timings fluctuate with sequence
      length (under rolling generation), GPU warm-up, and device contention. Reporting the
      distribution (not just the mean) makes it possible to tell "consistently slow" from
      "occasionally slow" without scraping per-batch logs.

    Additionally emits ``predict/total_time_sec`` (wall-clock across all epochs/batches)
    and ``predict/num_batches`` as raw counters for downstream computation.

    All metrics are written once per ``trainer.predict`` call via ``on_predict_end`` and
    rank-gated to ``trainer.is_global_zero`` to avoid duplicated writes in distributed
    prediction.
    """

    def on_predict_start(self, trainer, pl_module) -> None:
        self._epoch_times: list[float] = []
        self._batch_times: list[float] = []
        self._epoch_start: float | None = None
        self._batch_start: float | None = None
        self._predict_start: float = time.perf_counter()

    def on_predict_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.perf_counter()

    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        if self._epoch_start is None:
            return
        self._epoch_times.append(time.perf_counter() - self._epoch_start)
        self._epoch_start = None

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        self._batch_start = time.perf_counter()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self._batch_start is None:
            return
        self._batch_times.append(time.perf_counter() - self._batch_start)
        self._batch_start = None

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self._epoch_times and not self._batch_times:
            return
        # Rank-gate: in distributed predict, every rank hits ``on_predict_end``. Writing
        # ``log_metrics`` from every rank produces duplicated metric writes and races some
        # backends (e.g., MLflow), so only rank-zero logs. Measurement itself is per-rank
        # and independent — we just restrict the *write* side.
        if not trainer.is_global_zero:
            return

        metrics: dict[str, float | int] = {
            "predict/total_time_sec": time.perf_counter() - self._predict_start,
            "predict/num_batches": len(self._batch_times),
        }
        metrics.update(_summary_stats("predict/epoch_time_sec", self._epoch_times))
        metrics.update(_summary_stats("predict/batch_time_sec", self._batch_times))

        for logger in _ensure_logger_sequence(trainer.loggers):
            logger.log_metrics(metrics, step=trainer.global_step)


def _summary_stats(prefix: str, values: Sequence[float]) -> dict[str, float]:
    """Return ``{prefix}_{mean,min,max,std}`` for a sequence, or ``{}`` if empty.

    ``std`` is the population standard deviation (``statistics.pstdev``), which is defined
    even for a single data point (yields 0.0) — we use it rather than ``stdev`` so callers
    don't need to special-case the 1-sample epoch case. With an empty list we emit nothing
    so the downstream ``log_metrics`` call doesn't write garbage zeros.
    """
    if not values:
        return {}
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_std": statistics.pstdev(values),
    }


def _ensure_logger_sequence(loggers: Logger | Sequence[Logger] | None) -> Sequence[Logger]:
    if loggers is None:
        return []
    if isinstance(loggers, Logger):
        return [loggers]
    return list(loggers)
