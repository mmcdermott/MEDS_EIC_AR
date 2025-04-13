import lightning as L
import torch
from meds_torchdata import MEDSTorchBatch
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..model import Model
from .metrcis import NextCodeMetrics


class MEICARModule(L.LightningModule):
    def __init__(
        self,
        model: Model,
        metrics: NextCodeMetrics,
    ):
        super().__init__()
        self.model = model
        self.metrics = metrics

    def _log_metrics(
        self, loss: torch.Tensor, outputs: CausalLMOutputWithPast, batch: MEDSTorchBatch, stage: str
    ):
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/metrics", self.metrics(outputs.logits, batch))

    def training_step(self, batch: MEDSTorchBatch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, "train")

        return loss

    def validation_step(self, batch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, "val")

        return loss

    def configure_optimizers(self):
        raise NotImplementedError("Optimizer not yet implemented")
