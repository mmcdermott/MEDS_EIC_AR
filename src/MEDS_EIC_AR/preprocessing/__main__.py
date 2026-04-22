"""CLI entry point for MEDS-EIC-AR preprocessing (``MEICAR_process_data``).

Shells out to two upstream CLIs back-to-back against a shared Hydra config:

1. ``MEDS_transform-pipeline`` — the general MEDS → MEDS pipeline transform pass (filtering, timeline
   tokens, value quantization). Takes the pipeline-config YAML as a positional argument. Shipped by
   :mod:`MEDS_transforms` (0.6.x+).
2. ``MTD_preprocess`` — the :mod:`meds_torchdata` tensorization pass that produces the directory layout
   consumed by :class:`meds_torchdata.MEDSPytorchDataset` and the downstream training loop.

The ``do_reshard`` toggle controls whether the raw input is re-sharded by split before stage 1 —
required for datasets that are not already sharded. ``do_demo`` lowers the rare-code / rare-subject
filtering thresholds so the pipeline runs on small fixtures without filtering everything out.
"""

import logging
import os
import subprocess
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_EIC_AR") / "preprocessing" / "configs"


def _run_streamed(cmd: list[str], *, env: dict[str, str] | None = None, stage_name: str) -> None:
    """Run a subprocess, streaming its output live to the parent's stdout/stderr.

    The two MTD stages (``MEDS_transform-pipeline`` and ``MTD_preprocess``) can run for many
    minutes on real datasets. Previously we used ``subprocess.run(..., capture_output=True)``
    which buffered the entire stdout/stderr in memory until the child exited — on a several-
    hour MIMIC-IV preprocessing job this meant zero progress visibility for the user and
    unbounded parent-process memory growth. Piping straight through to the parent's streams
    gives real-time progress and keeps memory flat at the cost of only surfacing output as
    lines are produced (which is what you want for a user-facing CLI).

    Also fixes the asymmetric error-logging pattern the pre-refactor code had: the old
    MEDS_transform block logged both stdout and stderr on failure, but the MTD_preprocess
    block only logged stderr, losing Python traceback context. Streaming both to the
    parent's streams removes that asymmetry entirely — failure output is already visible.
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    # Letting ``stdout`` / ``stderr`` default to ``None`` in ``subprocess.run`` means "inherit
    # the parent process's OS-level file descriptors," which is exactly the streaming behavior
    # we want. Previously we passed ``sys.stdout`` / ``sys.stderr`` explicitly, but those are
    # Python-level wrappers that can be swapped by pytest capture, Jupyter notebooks, or any
    # logging wrapper — in those environments ``sys.stdout`` has no real ``fileno()`` and
    # ``subprocess.run`` either raises or redirects to an unintended destination. Default
    # (inherit) is both sufficient and robust to all such wrappers.
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:  # pragma: no cover
        raise RuntimeError(
            f"{stage_name} failed with exit code {result.returncode}. "
            "Check the streamed output above for details."
        )


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_process_data")
def process_data(cfg: DictConfig):
    input_dir = Path(cfg.input_dir)
    intermediate_dir = Path(cfg.intermediate_dir)
    output_dir = Path(cfg.output_dir)
    do_demo = cfg.do_demo

    # 0. Pre-MTD pre-processing
    logger.info("Pre-MTD pre-processing")
    done_fp = intermediate_dir / ".done"
    if done_fp.exists():  # pragma: no cover
        logger.info("Pre-MTD pre-processing already done, skipping")
    else:
        # ``os.environ`` is ``str -> str``; no nested-mutable values to worry about, so a
        # shallow copy is sufficient. ``os.environ.copy()`` returns a plain ``dict`` we're
        # free to mutate without affecting the parent's environment.
        env = os.environ.copy()
        env["RAW_MEDS_DIR"] = str(input_dir)
        env["MTD_INPUT_DIR"] = str(intermediate_dir)

        if do_demo:
            env["MIN_SUBJECTS_PER_CODE"] = "2"
            env["MIN_EVENTS_PER_SUBJECT"] = "1"

        pipeline_config_fp = (CONFIGS / "_reshard_data.yaml") if cfg.do_reshard else (CONFIGS / "_data.yaml")
        _run_streamed(
            ["MEDS_transform-pipeline", str(pipeline_config_fp)],
            env=env,
            stage_name="MEDS_transform-pipeline",
        )

        logger.info("Pre-MTD pre-processing done")
        done_fp.touch()

    # 1. Run MTD pre-processing
    logger.info("Running MTD pre-processing")
    done_fp = output_dir / ".done"

    if done_fp.exists():  # pragma: no cover
        logger.info("MTD pre-processing already done, skipping")
    else:
        # This stage doesn't mutate the environment (unlike the Pre-MTD block above, which
        # adds ``RAW_MEDS_DIR`` / ``MTD_INPUT_DIR``), so pass the parent's ``os.environ``
        # directly. The earlier copy here was dead code.
        _run_streamed(
            [
                "MTD_preprocess",
                f"MEDS_dataset_dir={intermediate_dir!s}",
                f"output_dir={output_dir!s}",
            ],
            stage_name="MTD_preprocess",
        )

        logger.info("MTD pre-processing done")
        done_fp.touch()


# ``python -m MEDS_EIC_AR.preprocessing`` goes through this module as ``__main__``. Without
# this guard, ``@hydra.main`` decorates ``process_data`` but nothing invokes it, so the
# command silently no-ops. Console-script entrypoints registered in ``pyproject.toml`` call
# ``process_data`` directly and aren't affected, but ``python -m`` invocation is the
# standard pattern users expect to work for any single-entrypoint module.
if __name__ == "__main__":  # pragma: no cover
    process_data()
