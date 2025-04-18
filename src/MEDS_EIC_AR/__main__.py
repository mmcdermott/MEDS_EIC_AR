import logging
import shutil
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import hydra
from hydra.utils import instantiate
from meds_torchdata import MEDSTorchDataConfig
from MEDS_transforms.runner import load_yaml_file  # noqa: F401
from omegaconf import DictConfig, OmegaConf

from .lightning import MEICARModule
from .utils import resolve_generation_context_size  # noqa: F401

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_EIC_AR") / "configs"

MEDSTorchDataConfig.add_to_config_store("datamodule/config")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_pretrain")
def pretrain(cfg: DictConfig):
    st = datetime.now(tz=UTC)

    OmegaConf.save(cfg, Path(cfg.output_dir) / "config.yaml")

    D = instantiate(cfg.datamodule)

    M = instantiate(
        cfg.lightning_module,
        model={"gpt_kwargs": {"vocab_size": D.config.vocab_size}},
        metrics={"vocab_size": D.config.vocab_size},
    )

    trainer = instantiate(cfg.trainer)

    trainer.fit(model=M, datamodule=D)

    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    if not best_ckpt_path.is_file():
        raise ValueError("No best checkpoint reported.")

    output_fp = Path(cfg.output_dir) / "best_model.ckpt"
    shutil.copyfile(best_ckpt_path, output_fp)

    best_score = trainer.checkpoint_callback.best_model_score

    logger.info(f"Best checkpoint (with score {best_score:.2f}) copied to {output_fp!s}.")
    logger.info(f"Training complete in {datetime.now(tz=UTC) - st}")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_generate_trajectories")
def generate_trajectories(cfg: DictConfig):
    st = datetime.now(tz=UTC)

    D = instantiate(cfg.datamodule)

    M = MEICARModule.load_from_checkpoint(cfg.ckpt_path)
    M.eval()

    trainer = instantiate(cfg.trainer)

    val_predictions = trainer.predict(model=M, dataloaders=D.val_dataloader())
    held_out_predictions = trainer.predict(model=M, dataloaders=D.test_dataloader())

    raise NotImplementedError("Trajectory generation is not implemented yet.")

    logger.info(f"Generation of trajectories complete in {datetime.now(tz=UTC) - st}")
