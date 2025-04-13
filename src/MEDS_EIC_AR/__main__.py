import logging
from importlib.resources import files

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from meds import code_metadata_filepath
from meds_torchdata import MEDSTorchDataConfig
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_EIC_AR") / "configs"

cs = ConfigStore.instance()
cs.store(name="MEDSTorchDataConfig", node=MEDSTorchDataConfig)


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_train")
def train(cfg: DictConfig):
    D = instantiate(cfg.dataset)
    metadata_df = pl.read_parquet(D.cfg.tensorized_cohort_dir / code_metadata_filepath, use_pyarrow=True)
    D.vocab_size = metadata_df.select(pl.col("code/vocab_index")).max().item() + 1

    M = instantiate(cfg.lightning_module, vocab_size=D.vocab_size)

    print("WOO", M)

    # trainer = instantiate(cfg.trainer)

    # trainer.fit(M, D)
