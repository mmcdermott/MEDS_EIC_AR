from collections.abc import Callable, Iterator

import lightning as L
import torch
from meds_torchdata import MEDSTorchBatch
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..model import Model
from .metrics import NextCodeMetrics


class MEICARModule(L.LightningModule):
    def __init__(
        self,
        model: Model,
        metrics: NextCodeMetrics,
        optimizer: Callable[[Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer],
        LR_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler] | None = None,
    ):
        super().__init__()
        self.model = model
        self.metrics = metrics
        self.optimizer_factory = optimizer
        self.LR_scheduler_factory = LR_scheduler

        self.save_hyperparameters()

    def _log_metrics(
        self, loss: torch.Tensor, outputs: CausalLMOutputWithPast, batch: MEDSTorchBatch, stage: str
    ):
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f"{stage}/{k}": v for k, v in self.metrics(outputs.logits, batch).items()})

    def training_step(self, batch: MEDSTorchBatch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, "train")

        return loss

    def validation_step(self, batch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, "tuning")

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())

        if self.LR_scheduler_factory is not None:
            scheduler = self.LR_scheduler_factory(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "tuning/loss"},
            }
        else:
            return optimizer

    def predict_step(self, batch: MEDSTorchBatch):
        """Produces generated trajectories for a given batch of data."""
        return self.model.generate(batch)
