import lightning as L
from meds_torchdata import MEDSTorchBatch
from omegaconf import DictConfig

from ..model import Model


class MEICARModule(L.LightningModule):
    def __init__(self, gpt_kwargs: dict | DictConfig):
        super().__init__()
        self.model = Model(gpt_kwargs)

    def training_step(self, batch: MEDSTorchBatch):
        loss, outputs = self.model(batch)

        return loss

    def validation_step(self, batch):
        loss, outputs = self.model(batch)

        return loss

    def configure_optimizers(self):
        raise NotImplementedError("Optimizer not yet implemented")
