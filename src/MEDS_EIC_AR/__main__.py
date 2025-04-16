import logging
from importlib.resources import files

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from meds import code_metadata_filepath
from meds_torchdata import MEDSTorchDataConfig
from omegaconf import DictConfig, open_dict

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_EIC_AR") / "configs"

cs = ConfigStore.instance()
cs.store(group="datamodule/config", name="MEDSTorchDataConfig", node=MEDSTorchDataConfig)
cfg_node = cs.repo["datamodule"]["config"]["MEDSTorchDataConfig.yaml"].node
with open_dict(cfg_node):
    cfg_node["_target_"] = "meds_torchdata.MEDSTorchDataConfig"


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_pretrain")
def pretrain(cfg: DictConfig):
    D = instantiate(cfg.datamodule)

    # TODO: Pending https://github.com/mmcdermott/meds-torch-data/issues/22, we set this manually.
    metadata_df = pl.read_parquet(D.config.tensorized_cohort_dir / code_metadata_filepath, use_pyarrow=True)
    D.vocab_size = metadata_df.select(pl.col("code/vocab_index")).max().item() + 1

    M = instantiate(
        cfg.lightning_module,
        model={"gpt_kwargs": {"vocab_size": D.vocab_size}},
        metrics={"vocab_size": D.vocab_size},
    )

    trainer = instantiate(cfg.trainer)

    trainer.fit(M, D)
