from collections.abc import Callable, Iterator
from pathlib import Path

import lightning as L
import torch
from meds_torchdata import MEDSTorchBatch
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..model import Model
from .metrics import NextCodeMetrics


class MEICARModule(L.LightningModule):
    """A LightningModule for training and evaluating the MEICAR model."""

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: Path | None = None) -> "MEICARModule":
        """Loads the full lightning module from a checkpoint.

        Args:
            ckpt_path: Path to the checkpoint file.

        Returns:
            The loaded MEICARModule instance.

        Raises:
            KeyError: If the checkpoint does not contain the expected hyperparameters.

        Examples:
            >>> model = Model({
            ...     "num_hidden_layers": 2,
            ...     "num_attention_heads": 2,
            ...     "hidden_size": 4,
            ...     "max_position_embeddings": 3,
            ...     "vocab_size": 10,
            ... })
            >>> metrics = NextCodeMetrics(top_k=[1, 2, 3], vocab_size=4)
            >>> module = MEICARModule(model=model, metrics=metrics, optimizer=None)

        In pytorch lightning, saving and loading checkpoints is done using the `Trainer` class. We'll make one
        and attach the module to it for testing purposes:

            >>> trainer = L.Trainer(logger=False)
            >>> trainer.strategy.connect(module)

        We'll grab the models current parameters so we can compare after loading

            >>> import copy
            >>> model_params = copy.deepcopy(module.state_dict())

        Now, we can save the checkpoint to a temporary file and load it back in:

            >>> with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
            ...     trainer.save_checkpoint(f.name)
            ...     loaded_module = MEICARModule.load_from_checkpoint(f.name)

        We can check that the loaded module has the same parameters as the original:

            >>> if loaded_module.state_dict().keys() != model_params.keys():
            ...     print("Loaded module has different parameter names than the original!")
            ... else:
            ...     print("Loaded module has the same parameter names as the original!")
            Loaded module has the same parameter names as the original!
            >>> for k, v in model_params.items():
            ...     assert torch.equal(v, loaded_module.state_dict()[k]), f"Parameter {k} does not match"
        """

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        hparams = checkpoint.get("hyper_parameters", {})

        for k in ["model", "metrics", "optimizer", "LR_scheduler"]:
            if k not in hparams:
                raise KeyError(f"Checkpoint does not contain {k} hyperparameters. Got {list(hparams.keys())}")

        model = Model(**hparams["model"])
        metrics = NextCodeMetrics(**hparams["metrics"])
        optimizer = hparams["optimizer"]
        LR_scheduler = hparams["LR_scheduler"]

        return super().load_from_checkpoint(
            ckpt_path, model=model, metrics=metrics, optimizer=optimizer, LR_scheduler=LR_scheduler
        )

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

        self.save_hyperparameters(
            {
                "model": model.hparams,
                "metrics": metrics.hparams,
                "optimizer": optimizer,
                "LR_scheduler": LR_scheduler,
            }
        )

    def _log_metrics(
        self, loss: torch.Tensor, outputs: CausalLMOutputWithPast, batch: MEDSTorchBatch, stage: str
    ):
        batch_size = batch.batch_size
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(
            {f"{stage}/{k}": v for k, v in self.metrics(outputs.logits, batch).items()},
            batch_size=batch_size,
        )

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

        if self.LR_scheduler_factory is None:
            return optimizer

        scheduler = self.LR_scheduler_factory(optimizer)

        LR_config = {
            "scheduler": scheduler,
            "frequency": 1,
        }

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau requires observing stable trends to make a conclusion about LR decay, so an
            # epcoh level interval is more appropriate.

            LR_config["monitor"] = "tuning/loss_epoch"
            LR_config["strict"] = True
            LR_config["interval"] = "epoch"
        else:
            # All other schedulers operate at a step level as they do not monitor the loss to make a
            # conclusion about LR decay.

            LR_config["interval"] = "step"

        return {"optimizer": optimizer, "lr_scheduler": LR_config}

    def predict_step(self, batch: MEDSTorchBatch):
        """Produces generated trajectories for a given batch of data."""
        return self.model.generate(batch)
