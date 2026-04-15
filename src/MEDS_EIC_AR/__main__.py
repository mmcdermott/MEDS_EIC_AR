import logging
import os
import shutil
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import hydra
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from meds import held_out_split, train_split, tuning_split
from meds_torchdata import MEDSTorchBatch, MEDSTorchDataConfig
from MEDS_trajectory_evaluation.schema import GeneratedTrajectorySchema
from MEDS_transforms.runner import load_yaml_file
from omegaconf import DictConfig, OmegaConf

from .generation import format_trajectories, get_timeline_end_token_idx
from .training import MEICARModule, find_checkpoint_path, validate_resume_directory

# Import OmegaConf Resolvers
from .utils import (
    gpus_available,
    hash_based_seed,
    int_prod,
    is_mlflow_logger,
    num_cores,
    num_gpus,
    oc_min,
    resolve_generation_context_size,
    save_resolved_config,
    sub,
)

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_EIC_AR") / "configs"

MEDSTorchDataConfig.add_to_config_store("datamodule/config")


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_pretrain")
def pretrain(cfg: DictConfig):
    st = datetime.now(tz=UTC)

    if cfg.do_overwrite and cfg.do_resume:
        logger.warning(
            "Both `do_overwrite` and `do_resume` are set to True. "
            "Only `do_overwrite` will be used, and the output directory will be cleared."
        )

    output_dir = Path(cfg.output_dir)

    if output_dir.is_file():
        raise NotADirectoryError(f"Output directory {output_dir} is a file, not a directory.")

    cfg_path = output_dir / "config.yaml"

    ckpt_path = None

    if cfg_path.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing output directory {output_dir}.")
            shutil.rmtree(output_dir, ignore_errors=True)
        elif cfg.do_resume:
            validate_resume_directory(output_dir, cfg)
            ckpt_path = find_checkpoint_path(output_dir)
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is populated. "
                "Use `do_overwrite` or `do_resume` to proceed."
            )
    else:
        OmegaConf.save(cfg, output_dir / "config.yaml")
        save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    D = instantiate(cfg.datamodule)

    gpt_kwargs = {"vocab_size": D.config.vocab_size, "eos_token_id": get_timeline_end_token_idx(D.config)}

    M = instantiate(
        cfg.lightning_module,
        model={"gpt_kwargs": gpt_kwargs},
        metrics={"vocab_size": D.config.vocab_size},
    )

    if M.model.do_demo or cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    trainer = instantiate(cfg.trainer)
    if any(is_mlflow_logger(logger) for logger in trainer.loggers):
        # We do the import only here to avoid importing mlflow if it isn't installed.
        import mlflow

        if "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING" not in os.environ:
            # The user can set this environment variable to enable or disable system metrics logging on their
            # own, but if they don't, it will by default be enabled.
            mlflow.enable_system_metrics_logging()

    trainer_kwargs = {"model": M, "datamodule": D}
    if ckpt_path:
        logger.info(f"Trying to resume training from checkpoint {ckpt_path}.")
        trainer_kwargs["ckpt_path"] = ckpt_path

    trainer.fit(**trainer_kwargs)

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

    logger.info("Setting torch float32 matmul precision to 'medium'.")
    torch.set_float32_matmul_precision("medium")

    D = instantiate(cfg.datamodule)

    # Validate rolling-generation config early — before loading the checkpoint and before running any
    # batches — so bad values (zero or negative budgets) fail fast with a clear message instead of
    # surfacing deep inside ``Model._rolling_generate`` after minutes of setup. The in-method checks in
    # ``_rolling_generate`` stay as a defensive backstop for direct library callers that bypass this CLI.
    rolling_cfg = cfg.get("rolling_generation", None)
    rolling_requested = False
    rolling_kwargs: dict[str, int] = {}
    if rolling_cfg is not None:
        for k in ("max_new_tokens", "rolling_context_size"):
            v = rolling_cfg.get(k, None)
            if v is None:
                continue
            if not isinstance(v, int) or v <= 0:
                raise ValueError(
                    f"rolling_generation.{k} must be a positive integer when set; got {v!r}. "
                    f"Leave it null to disable rolling generation for {k!r}."
                )
            rolling_kwargs[k] = v
        rolling_requested = "max_new_tokens" in rolling_kwargs
        # ``rolling_context_size`` only takes effect on the rolling path. ``Model.generate`` dispatches
        # to the rolling loop iff ``max_new_tokens is not None``; a ``rolling_context_size`` set in
        # isolation would flow through to the legacy single-chunk path where it is silently dropped.
        # Fail fast rather than accept a no-op config option.
        if "rolling_context_size" in rolling_kwargs and not rolling_requested:
            raise ValueError(
                "rolling_generation.rolling_context_size is set but "
                "rolling_generation.max_new_tokens is null. `rolling_context_size` only takes effect "
                "on the rolling path, which is enabled by setting `max_new_tokens`. Either set "
                "`max_new_tokens` to a positive integer to enable rolling generation, or leave "
                "`rolling_context_size` null."
            )

    M = MEICARModule.load_from_checkpoint(Path(cfg.ckpt_path))
    M.eval()

    # Auto-populate ``eos_token_id`` on the loaded checkpoint if it's unset. Models pretrained through
    # ``MEICAR_pretrain`` already have it populated from ``get_timeline_end_token_idx(D.config)``, but
    # older checkpoints (or models instantiated directly) may not. Doing this here means every
    # checkpoint driven through the generation CLI has a usable eos for both single-chunk and rolling
    # generation without requiring manual config intervention.
    if M.model.HF_model.config.eos_token_id is None:
        timeline_end_idx = get_timeline_end_token_idx(D.config)
        logger.info(
            f"Checkpoint {cfg.ckpt_path} has no eos_token_id set; defaulting to "
            f"get_timeline_end_token_idx(dataset_config) = {timeline_end_idx}."
        )
        M.model.HF_model.config.eos_token_id = timeline_end_idx

    # Fail fast if rolling was requested but eos is still unset (e.g. dataset config doesn't have a
    # TIMELINE//END token) or collides with PAD_INDEX. This is the same invariant ``_rolling_generate``
    # enforces internally, hoisted to config-load time so we don't spend any compute before diagnosing.
    if rolling_requested:
        eos_id = M.model.HF_model.config.eos_token_id
        if eos_id is None:
            raise ValueError(
                "rolling_generation.max_new_tokens is set but the model's eos_token_id could not be "
                "auto-populated from the dataset config. Rolling generation needs a valid eos token "
                "to handle cross-chunk stopping."
            )
        if eos_id == MEDSTorchBatch.PAD_INDEX:
            raise ValueError(
                f"Rolling generation requires eos_token_id ({eos_id}) to differ from "
                f"MEDSTorchBatch.PAD_INDEX ({MEDSTorchBatch.PAD_INDEX}). The finished-mask and "
                "post-EOS truncation would otherwise collapse onto padding."
            )

    M.generation_kwargs.update(rolling_kwargs)

    trainer = instantiate(cfg.trainer)

    inference = cfg.inference

    if cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    for split in inference.generate_for_splits:
        if split == train_split:
            dataloader = D.train_dataloader()
        elif split == tuning_split:
            dataloader = D.val_dataloader()
        elif split == held_out_split:
            dataloader = D.test_dataloader()
        else:
            raise ValueError(f"Unknown split {split}.")

        for sample in range(inference.N_trajectories_per_task_sample):
            out_fp = Path(cfg.output_dir) / split / f"{sample}.parquet"
            out_fp.parent.mkdir(parents=True, exist_ok=True)

            if out_fp.is_file() and not cfg.do_overwrite:
                logger.info(f"Skipping {out_fp} as it already exists.")
                continue
            else:
                out_fp.parent.mkdir(parents=True, exist_ok=True)

            seed = hash_based_seed(cfg.get("seed", None), split, sample)

            logger.info(f"Generating trajectories for {split} sample {sample} to {out_fp} with seed {seed}.")

            seed_everything(seed, workers=True)
            predictions = trainer.predict(model=M, dataloaders=dataloader)
            predictions_df = format_trajectories(dataloader.dataset, predictions)

            pa_table = GeneratedTrajectorySchema.align(predictions_df.to_arrow())
            pq.write_table(pa_table, out_fp)

    logger.info(f"Generation of trajectories complete in {datetime.now(tz=UTC) - st}")
