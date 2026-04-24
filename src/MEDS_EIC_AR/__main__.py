"""CLI entry points for training (``MEICAR_pretrain``) and generation (``MEICAR_generate_trajectories``).

Both commands are thin Hydra wrappers that resolve the run's config, wire up the
:class:`~MEDS_EIC_AR.training.MEICARModule`, and hand off to Lightning. On initial run creation (not on
resume), pre-training additionally writes a pip-freeze-style environment snapshot (``environment.txt``)
and the resolved Hydra config to ``output_dir`` so re-runs and post-hoc debugging have a fingerprint of
the invocation. Generation loads a checkpoint via ``model_initialization_dir``, runs the rolling
sliding-window predict path, and emits per-task-sample trajectory parquets under
``output_dir/<split>/<sample>.parquet``.
"""

import logging
import os
import shutil
from datetime import UTC, datetime
from functools import partial
from importlib.resources import files
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from meds import held_out_split, train_split, tuning_split
from meds_torchdata import MEDSTorchBatch, MEDSTorchDataConfig

# ``load_yaml_file`` is imported for its side effect — it is decorated with ``@OmegaConfResolver``
# and importing the symbol registers the ``load_yaml_file`` OmegaConf interpolation used by
# ``src/MEDS_EIC_AR/configs/_generate_trajectories.yaml`` (``${load_yaml_file:...}``). Do not remove
# as "unused": static analysis can't see the config-side reference and dropping the import breaks
# generation CLI startup with ``UnsupportedInterpolationType``.
from MEDS_transforms.runner import load_yaml_file
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .generation import (
    RepeatedPredictionDataset,
    collate_with_meta,
    get_timeline_end_token_idx,
    validate_rolling_cfg,
)
from .generation.finalize import finalize_predictions, write_rank_output
from .training import MEICARModule, find_checkpoint_path, validate_resume_directory

# Import OmegaConf Resolvers
from .utils import (
    apply_saved_logger_run_ids,
    gpus_available,
    hash_based_seed,
    int_prod,
    is_mlflow_logger,
    num_cores,
    num_gpus,
    oc_min,
    resolve_generation_context_size,
    save_environment_snapshot,
    save_logger_run_ids,
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
        # Capture the Python environment once at run-creation time (not on resume — the point
        # of an environment snapshot is "what was installed when this run was configured",
        # and a resumed run reusing the same output_dir should point at that original snapshot
        # rather than a fresh one capturing whatever's installed today).
        save_environment_snapshot(output_dir / "environment.txt")

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

    apply_saved_logger_run_ids(cfg.trainer, output_dir)
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
    save_logger_run_ids(trainer.loggers, output_dir)

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

    # Wire ``inference.do_sample`` through to ``Model.generate``. Kept on the inference section
    # rather than ``rolling_generation`` because ``do_sample`` is not rolling-specific — it
    # applies to both single-chunk and rolling paths.
    M.generation_kwargs["do_sample"] = cfg.inference.do_sample
    if not cfg.inference.do_sample:
        logger.warning(
            "inference.do_sample is False — generation is greedy (argmax-per-step). This is an "
            "anti-pattern for real trajectory generation: every sample with the same prompt "
            "becomes identical, collapsing N_trajectories_per_task_sample's diversity value and "
            "destroying variance estimates downstream. The only legitimate use is correctness "
            "testing (e.g. regression tests that assert deterministic grammar-valid output). If "
            "you're running this for any other purpose, set inference.do_sample=true."
        )

    apply_saved_logger_run_ids(cfg.trainer, Path(cfg.model_initialization_dir))
    trainer = instantiate(cfg.trainer)

    inference = cfg.inference

    if cfg.get("seed", None):
        seed_everything(cfg.get("seed", 1), workers=True)

    n_trajectories = inference.N_trajectories_per_task_sample

    for split in inference.generate_for_splits:
        if split == train_split:
            base_loader = D.train_dataloader()
        elif split == tuning_split:
            base_loader = D.val_dataloader()
        elif split == held_out_split:
            base_loader = D.test_dataloader()
        else:
            raise ValueError(f"Unknown split {split}.")

        base_dataset = base_loader.dataset
        if len(base_dataset) == 0:
            logger.info(f"Split {split} has zero base-dataset rows; skipping generation.")
            continue

        # Skip work for trajectories whose output parquet already exists. If every requested
        # trajectory is already on disk and ``do_overwrite`` is false, skip the predict pass
        # entirely; otherwise we still run a single pass over the full ``N``-expanded dataset and
        # just don't write the parquets that already exist. (Partial-skip support is a minor
        # wrinkle — it keeps existing checkpointed runs idempotent without making us special-case
        # mid-run resumption.)
        trajectory_paths = {
            trajectory_idx: Path(cfg.output_dir) / split / f"{trajectory_idx}.parquet"
            for trajectory_idx in range(n_trajectories)
        }
        for trajectory_fp in trajectory_paths.values():
            trajectory_fp.parent.mkdir(parents=True, exist_ok=True)
        if not cfg.do_overwrite and all(p.is_file() for p in trajectory_paths.values()):
            logger.info(
                f"Skipping all {n_trajectories} trajectories for split {split}: every parquet exists."
            )
            continue

        # Stale-output guard + cleanup, BEFORE the predict pass. Rank outputs from a prior
        # failed run (or a rerun with a different ``world_size`` whose old ``rank_{r}`` files
        # no longer map to any current rank) would otherwise be picked up by
        # ``finalize_predictions``'s glob alongside the current run's outputs. With
        # ``do_overwrite=False``, every rank sees the shared filesystem state and raises
        # ``FileExistsError`` together — this avoids a rank-0-only raise where non-zero ranks
        # would then hang in a subsequent collective. Rank 0 does the actual cleanup +
        # ``mkdir``; other ranks just proceed to ``trainer.predict`` where the strategy setup
        # barrier synchronizes all ranks before any of them tries to write its rank output.
        rank_outputs_dir = Path(cfg.output_dir) / split / "_rank_outputs"
        stale = (
            sorted(rank_outputs_dir.glob("trajectory_*.rank_*.parquet")) if rank_outputs_dir.exists() else []
        )
        if stale and not cfg.do_overwrite:
            raise FileExistsError(
                f"Found {len(stale)} stale rank-output file(s) under {rank_outputs_dir}. "
                "A prior generation run did not finalize cleanly. Re-run with "
                "do_overwrite=True to discard them and retry, or investigate and remove "
                "them manually."
            )
        if trainer.is_global_zero:
            for p in stale:
                p.unlink()
            rank_outputs_dir.mkdir(parents=True, exist_ok=True)

        # Expand the base dataset so each base-dataset row contributes ``n_trajectories``
        # consecutive rows. See issue #89 for the motivation: one predict pass instead of
        # ``N``, tighter padding, and prefix-cache reuse on backends that have one (#88, #97).
        # The ordering invariant — ``dataset_row_idx`` changes slow, ``trajectory_idx``
        # changes fast — means rows for trajectory ``t`` retain stable identifiers all the way
        # through the shard-then-merge finalize step.
        expanded_dataset = RepeatedPredictionDataset(base_dataset, n_trajectories=n_trajectories)
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
            f"Generating {n_trajectories} trajectories for each of {len(base_dataset)} "
            f"base-dataset rows in split {split} (one interleaved predict pass over "
            f"{len(expanded_dataset)} expanded rows, seed={seed})."
        )
        seed_everything(seed, workers=True)
        predictions = trainer.predict(model=M, dataloaders=expanded_loader)

        # Shard-then-merge finalize. Every rank writes a per-trajectory rank output under
        # ``rank_outputs_dir``; a barrier; rank 0 merges the per-rank partials into the final
        # per-trajectory parquets. Works uniformly for single-device (``world_size=1``: one
        # file per trajectory, rank-0 merge is effectively a passthrough) and DDP (one file
        # per ``(trajectory, rank)``; rank 0's ``pl.concat + sort`` by ``dataset_row_idx``
        # stitches them together). No cross-rank gather, no ``BasePredictionWriter`` — the
        # filesystem is the coordination primitive. Two explicit barriers:
        # (1) post-write so rank 0 waits for every rank's output to land before reading,
        # (2) post-finalize so non-zero ranks don't race into the next split.
        write_rank_output(predictions, rank=trainer.global_rank, rank_outputs_dir=rank_outputs_dir)
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            finalize_predictions(
                rank_outputs_dir=rank_outputs_dir,
                trajectory_paths=trajectory_paths,
                base_dataset=base_dataset,
                n_dataset_rows=len(base_dataset),
                do_overwrite=cfg.do_overwrite,
            )
        trainer.strategy.barrier()

    # Save the generation run's logger ids into the *generation* ``output_dir``, not the
    # training checkpoint's ``model_initialization_dir``. A caller using the escape hatch
    # described in ``apply_saved_logger_run_ids`` (explicit fresh ``run_id`` for generation)
    # would otherwise overwrite the training run's saved ids and change what future pretrain-
    # resume attaches to. Separating the two directories keeps the pretrain save-point frozen.
    save_logger_run_ids(trainer.loggers, Path(cfg.output_dir))
    logger.info(f"Generation of trajectories complete in {datetime.now(tz=UTC) - st}")
