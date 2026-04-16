import logging
import os
import shutil
from datetime import UTC, datetime
from functools import partial
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
from torch.utils.data import DataLoader

from .generation import (
    RepeatedPredictionDataset,
    collate_with_meta,
    format_trajectories,
    get_timeline_end_token_idx,
    validate_rolling_cfg,
)
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
    rolling_kwargs = validate_rolling_cfg(cfg.get("rolling_generation", None))
    rolling_requested = "max_new_tokens" in rolling_kwargs

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

    n_samples = inference.N_trajectories_per_task_sample

    for split in inference.generate_for_splits:
        if split == train_split:
            base_loader = D.train_dataloader()
        elif split == tuning_split:
            base_loader = D.val_dataloader()
        elif split == held_out_split:
            base_loader = D.test_dataloader()
        else:
            raise ValueError(f"Unknown split {split}.")

        # Skip work for samples whose output parquet already exists. If every requested sample is
        # already on disk and ``do_overwrite`` is false, skip the predict pass entirely; otherwise
        # we still run a single pass over the full ``N``-expanded dataset and just don't write the
        # parquets that already exist. (Partial-skip support is a minor wrinkle — it keeps existing
        # checkpointed runs idempotent without making us special-case mid-run resumption.)
        sample_paths = {
            sample: Path(cfg.output_dir) / split / f"{sample}.parquet" for sample in range(n_samples)
        }
        for sample_fp in sample_paths.values():
            sample_fp.parent.mkdir(parents=True, exist_ok=True)
        if not cfg.do_overwrite and all(p.is_file() for p in sample_paths.values()):
            logger.info(f"Skipping all {n_samples} samples for split {split}: every parquet exists.")
            continue

        # Expand the base dataset so each subject contributes ``n_samples`` consecutive rows. See
        # issue #89 for the motivation: one predict pass instead of ``N``, tighter padding, and
        # prefix-cache reuse on backends that have one (#88, #97). The ordering invariant — subject
        # changes slow, sample changes fast — means rows for sample ``s`` extracted from each batch
        # in order land in subject-index order overall, which is what ``format_trajectories`` needs
        # so its sequential ``schema_df.slice(...)`` lines up with the right subject metadata.
        base_dataset = base_loader.dataset
        expanded_dataset = RepeatedPredictionDataset(base_dataset, n_samples=n_samples)
        expanded_loader = DataLoader(
            expanded_dataset,
            batch_size=base_loader.batch_size,
            shuffle=False,
            num_workers=base_loader.num_workers,
            collate_fn=partial(collate_with_meta, base_collate=base_dataset.collate),
            pin_memory=base_loader.pin_memory,
        )

        seed = hash_based_seed(cfg.get("seed", None), split)
        logger.info(
            f"Generating {n_samples} trajectories for each of {len(base_dataset)} subjects in split "
            f"{split} (one interleaved predict pass over {len(expanded_dataset)} expanded rows, "
            f"seed={seed})."
        )
        seed_everything(seed, workers=True)
        predictions = trainer.predict(model=M, dataloaders=expanded_loader)

        # Demux the flat predictions into per-sample, per-batch token lists. Within each batch the
        # rows for sample ``s`` are in subject-index order (because the expanded dataset was built
        # with subject-changes-slow ordering and ``shuffle=False``), and across batches the
        # subject-index ranges are non-overlapping and increasing — so the concatenation per sample
        # ``s`` is exactly the order that ``format_trajectories`` consumes from
        # ``base_dataset.schema_df``.
        per_sample_batches: dict[int, list[torch.Tensor]] = {s: [] for s in range(n_samples)}
        for pred in predictions:
            tokens = pred["tokens"]
            sample_idxs = pred["sample_idxs"]
            # Iterate over the samples actually present in this batch rather than always doing N
            # boolean compares. For batches that cover every sample (the common case when
            # batch_size >= N), this is the same work; for tail batches or small batch sizes, it
            # scales with the number of distinct sample_idxs in the batch instead.
            for s in sample_idxs.unique().tolist():
                mask = sample_idxs == s
                per_sample_batches[s].append(tokens[mask])

        for sample, out_fp in sample_paths.items():
            if out_fp.is_file() and not cfg.do_overwrite:
                logger.info(f"Skipping {out_fp} as it already exists.")
                continue
            logger.info(f"Writing {sample} sample for split {split} to {out_fp}.")
            predictions_df = format_trajectories(base_dataset, per_sample_batches[sample])
            pa_table = GeneratedTrajectorySchema.align(predictions_df.to_arrow())
            pq.write_table(pa_table, out_fp)

    logger.info(f"Generation of trajectories complete in {datetime.now(tz=UTC) - st}")
