from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Literal

import hydra
import lightning as L
import torch
from meds import held_out_split, train_split, tuning_split
from meds_torchdata import MEDSTorchBatch
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..model import Model
from .metrics import NextCodeMetrics


def _factory_to_dict(factory: partial | None) -> dict[str, Any] | None:
    """Extracts a sufficient dictionary for reconstructing the optimizer or LR scheduler.

    Args:
        factory: A partial function that creates an optimizer or LR scheduler. This comes from the Hydra
            instantiation's "partial" functionality, which is used to partially initialize optimizers or LR
            schedulers without the need to pass in the parameters or optimizers, respectively.

    Returns:
        A dictionary suitable for storing, logging, and sufficient to reconstruct the given factory function.
        The dictionary will contain the special key "_target_" which contains the full path to the target
        function or module to be called. The rest of the dictionary will contain the keyword arguments passed
        in the partial. If the factory is None, returns None.

    Raises:
        TypeError: If the factory is not a partial function.
        ValueError: If the factory partial has any positional arguments or if it uses the reserved key
            "_target_" as a keyword argument.

    Examples:
        >>> print(_factory_to_dict(None))
        None
        >>> _factory_to_dict(partial(torch.optim.Adam, lr=0.001))
        {'_target_': 'torch.optim.adam.Adam', 'lr': 0.001}
        >>> from transformers import get_cosine_schedule_with_warmup
        >>> _factory_to_dict(partial(get_cosine_schedule_with_warmup, num_warmup_steps=10))
        {'_target_': 'transformers.optimization.get_cosine_schedule_with_warmup', 'num_warmup_steps': 10}

    Errors include type checking and some value checks:

        >>> _factory_to_dict(43)
        Traceback (most recent call last):
            ...
        TypeError: Expected a partial function, got <class 'int'>
        >>> _factory_to_dict(partial(torch.optim.Adam, 0.001))
        Traceback (most recent call last):
            ...
        ValueError: Expected a partial function with no positional arguments. Got (0.001,)
        >>> _factory_to_dict(partial(torch.optim.Adam, lr=0.001, _target_="foo"))
        Traceback (most recent call last):
            ...
        ValueError: Expected a partial function with no _target_ keyword argument. Got _target_=foo
    """
    if factory is None:
        return None

    if not isinstance(factory, partial):
        raise TypeError(f"Expected a partial function, got {type(factory)}")

    if factory.args:
        raise ValueError(f"Expected a partial function with no positional arguments. Got {factory.args}")

    kwargs = factory.keywords.copy()

    if "_target_" in kwargs:
        raise ValueError(
            "Expected a partial function with no _target_ keyword argument. "
            f"Got _target_={kwargs['_target_']}"
        )

    target = f"{factory.func.__module__}.{factory.func.__qualname__}"

    return {"_target_": target, **kwargs}


def _dict_to_factory(d: dict[str, Any] | None) -> partial:
    """Reconstructs a partial function from a dictionary.

    This is actually just a wrapper around `hydra.utils.instantiate` that sets the `_partial_` flag to True,
    so that it is clear we can use `_factory_to_dict` and `_dict_to_factory` to round-trip encode-decode the
    partial objects instantiated by `hydra.utils.instantiate`.

    Args:
        d: A dictionary containing the target function or module to be called under the key "_target_". The
            rest of the dictionary should contain the keyword arguments to be passed to the function.

    Returns:
        A partial function that creates an optimizer or LR scheduler.

    Examples:
        >>> d = {'_target_': 'torch.optim.adam.Adam', 'lr': 0.001}
        >>> factory = _dict_to_factory(d)
        >>> print(factory.func)
        <class 'torch.optim.adam.Adam'>
        >>> print(factory.keywords)
        {'lr': 0.001}
        >>> print(_factory_to_dict(factory))
        {'_target_': 'torch.optim.adam.Adam', 'lr': 0.001}
        >>> print(_dict_to_factory(None))
        None
    """

    return None if d is None else hydra.utils.instantiate(d, _partial_=True)


class MEICARModule(L.LightningModule):
    """A LightningModule for training and evaluating the MEICAR model."""

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: Path | None = None) -> "MEICARModule":
        """Loads the full lightning module from a checkpoint.

        Args:
            ckpt_path: Path to the checkpoint file.

        Returns:
            The loaded MEICARModule instance, with all hyperparameters matching _except_ for the optimizer
            factory and LR scheduler which are discarded, as we can't tell from the saved data alone what
            classes they should be.

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
        optimizer = _dict_to_factory(hparams["optimizer"])
        LR_scheduler = _dict_to_factory(hparams["LR_scheduler"])

        return super().load_from_checkpoint(
            ckpt_path, model=model, metrics=metrics, optimizer=optimizer, LR_scheduler=LR_scheduler
        )

    def __init__(
        self,
        model: Model,
        metrics: NextCodeMetrics,
        optimizer: Callable[[Iterator[torch.nn.parameter.Parameter]], torch.optim.Optimizer] | None = None,
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
                "optimizer": _factory_to_dict(self.optimizer_factory),
                "LR_scheduler": _factory_to_dict(self.LR_scheduler_factory),
            }
        )

    def _log_metrics(
        self,
        loss: torch.Tensor,
        outputs: CausalLMOutputWithPast,
        batch: MEDSTorchBatch,
        stage: Literal[train_split, tuning_split, held_out_split],
    ):
        batch_size = batch.batch_size

        is_train = stage == train_split

        self.log(f"{stage}/loss", loss, on_step=is_train, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(
            {f"{stage}/{k}": v for k, v in self.metrics(outputs.logits, batch).items()},
            batch_size=batch_size,
            on_step=is_train,
        )

    def training_step(self, batch: MEDSTorchBatch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, train_split)

        return loss

    def validation_step(self, batch):
        loss, outputs = self.model(batch)
        self._log_metrics(loss, outputs, batch, tuning_split)

        return loss

    def configure_optimizers(self):
        if self.optimizer_factory is None:
            raise ValueError("Optimizer factory is not set. Cannot configure optimizers.")

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

            LR_config["monitor"] = "tuning/loss"
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
