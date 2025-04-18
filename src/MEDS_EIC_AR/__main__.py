import logging
from importlib.resources import files

import hydra
from hydra.utils import instantiate
from meds_torchdata import MEDSTorchDataConfig
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_EIC_AR") / "configs"

MEDSTorchDataConfig.add_to_config_store("datamodule/config")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_pretrain")
def pretrain(cfg: DictConfig):
    D = instantiate(cfg.datamodule)

    M = instantiate(
        cfg.lightning_module,
        model={"gpt_kwargs": {"vocab_size": D.config.vocab_size}},
        metrics={"vocab_size": D.config.vocab_size},
    )

    trainer = instantiate(cfg.trainer)

    trainer.fit(M, D)
