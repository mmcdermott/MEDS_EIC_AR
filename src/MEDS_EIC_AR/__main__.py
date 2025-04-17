import logging
import shutil
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from meds import code_metadata_filepath
from meds_torchdata import MEDSTorchDataConfig
from MEDS_transforms.runner import load_yaml_file  # noqa: F401
from omegaconf import DictConfig, OmegaConf, open_dict

from .lightning import MEICARModule

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_EIC_AR") / "configs"

cs = ConfigStore.instance()
cs.store(group="datamodule/config", name="MEDSTorchDataConfig", node=MEDSTorchDataConfig)
cfg_node = cs.repo["datamodule"]["config"]["MEDSTorchDataConfig.yaml"].node
with open_dict(cfg_node):
    cfg_node["_target_"] = "meds_torchdata.MEDSTorchDataConfig"


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_pretrain")
def pretrain(cfg: DictConfig):
    st = datetime.now(tz=UTC)

    OmegaConf.save(cfg, Path(cfg.model_dir) / "config.yaml")

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

    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    if not best_ckpt_path.is_file():
        raise ValueError("No best checkpoint reported.")

    output_fp = Path(cfg.model_dir) / "best_model.ckpt"
    shutil.copyfile(best_ckpt_path, output_fp)

    best_score = trainer.checkpoint_callback.best_model_score

    logger.info(f"Best checkpoint (with score {best_score:.2f}) copied to {output_fp!s}.")
    logger.info(f"Training complete in {datetime.now(tz=UTC) - st}")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_generate_trajectories")
def generate_trajectories(cfg: DictConfig):
    st = datetime.now(tz=UTC)

    D = instantiate(cfg.datamodule)

    M = instantiate(cfg.lightning_module)
    M = MEICARModule.load_from_checkpoint(cfg.ckpt_path)
    M.eval()

    trainer = instantiate(cfg.trainer)

    trainer.predict(D)

    raise NotImplementedError("Trajectory generation is not implemented yet.")

    logger.info(f"Generation of trajectories complete in {datetime.now(tz=UTC) - st}")
