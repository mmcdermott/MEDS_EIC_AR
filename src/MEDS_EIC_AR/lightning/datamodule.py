from functools import cached_property

import lightning as L
from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig
from torch.utils.data import DataLoader


class Datamodule(L.LightningDataModule):
    """A lightning datamodule for a MEDSPytorchDataset.

    Args:
        config: The configuration for the dataset.
        batch_size: The batch size for the dataloaders. Defaults to 32.
        num_workers: The number of workers for the dataloaders. Defaults to 0.

    Examples:
        >>> D = Datamodule(config=dataset_config, batch_size=2)

    After construction, we can access dataloaders for training, validation, and testing:

        >>> val_dataloader = D.val_dataloader()
        >>> val_batch = next(iter(val_dataloader))
        >>> val_batch
        MEDSTorchBatch(code=tensor([[37,  5, 32,  3, 14, 28, 36,  4]]),
                       numeric_value=tensor([[ 0.0000,  0.0000, -6.7254,  0.0000,
                                               0.0000, -1.1314,  0.2792,  0.0000]]),
                       numeric_value_mask=tensor([[False, False,  True, False,
                                                   False,  True,  True, False]]),
                       time_delta_days=tensor([[0.0000e+00, 0.0000e+00, 7.6853e+03, 0.0000e+00,
                                                0.0000e+00, 0.0000e+00, 7.9329e-02, 0.0000e+00]]),
                       event_mask=None,
                       static_code=tensor([[ 7, 12]]),
                       static_numeric_value=tensor([[ 0.0000e+00, -3.4028e+38]]),
                       static_numeric_value_mask=tensor([[False,  True]]),
                       boolean_value=None)

    By default, the train dataloader shuffles, which is suitable for training but less so for doctest outputs.
    We can account for this by seeding, however:

        >>> _ = L.seed_everything(0)
        >>> train_dataloader = D.train_dataloader()
        >>> train_batch = next(iter(train_dataloader))
        >>> train_batch
        MEDSTorchBatch(code=tensor([[37,  5, 32,  1, 13, 25, 33, 15, 26, 35],
                                    [37,  5, 32,  2, 14, 30, 36,  4,  0,  0]]),
                       numeric_value=tensor([[ 0.0000,  0.0000, -1.5495,  0.0000,  1.0000,
                                               0.0000, -1.0000,  0.0000, 0.0000,  1.0000],
                                             [ 0.0000,  0.0000,  1.2423,  0.0000,  0.0000,
                                               0.0000, -1.0000,  0.0000, 0.0000,  0.0000]]),
                       numeric_value_mask=tensor([[False, False,  True, False,  True,
                                                   False,  True, False, False,  True],
                                                  [False, False,  True, False, False,
                                                   False,  True, False,  True,  True]]),
                       time_delta_days=tensor([[0.0000e+00, 0.0000e+00, 1.0727e+04, 0.0000e+00, 0.0000e+00,
                                                0.0000e+00, 4.8264e-03, 0.0000e+00, 0.0000e+00, 2.5544e-02],
                                               [0.0000e+00, 0.0000e+00, 1.2367e+04, 0.0000e+00, 0.0000e+00,
                                                0.0000e+00, 4.6424e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00]]),
                       event_mask=None,
                       static_code=tensor([[ 7, 12],
                                           [ 8,  9]]),
                       static_numeric_value=tensor([[0., 0.],
                                                    [0., 0.]]),
                       static_numeric_value_mask=tensor([[False, False],
                                                         [False, False]]),
                       boolean_value=None)
    """

    config: MEDSTorchDataConfig

    def __init__(
        self,
        config: MEDSTorchDataConfig,
        batch_size: int = 32,
        num_workers: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def shared_dataloader_kwargs(self) -> dict:
        out = {"batch_size": self.batch_size}
        if self.num_workers is not None:
            out["num_workers"] = self.num_workers
        return out

    @cached_property
    def train_dataset(self) -> MEDSPytorchDataset:
        return MEDSPytorchDataset(self.config, split="train")

    @cached_property
    def val_dataset(self) -> MEDSPytorchDataset:
        return MEDSPytorchDataset(self.config, split="tuning")

    @cached_property
    def test_dataset(self) -> MEDSPytorchDataset:
        return MEDSPytorchDataset(self.config, split="held_out")

    def __dataloader(self, dataset: MEDSPytorchDataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, collate_fn=dataset.collate, **self.shared_dataloader_kwargs, **kwargs)

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False)
