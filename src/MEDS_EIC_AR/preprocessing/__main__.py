import copy
import json
import logging
import os
import subprocess
from importlib.resources import files
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig

from MEDS_EIC_AR import stages as _  # register custom stages

logger = logging.getLogger(__name__)

CONFIGS = files("MEDS_EIC_AR") / "preprocessing" / "configs"


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
        env = copy.deepcopy(os.environ)
        env["RAW_MEDS_DIR"] = str(input_dir)
        env["MTD_INPUT_DIR"] = str(intermediate_dir)

        # Determine which preprocessing configuration to use
        include_numeric = env.get("INCLUDE_NUMERIC_VALUES", "1") not in {"0", "false", "False"}
        quantiles_fp = env.get("NUMERIC_QUANTILES_FP")
        quantiles_list = env.get("NUMERIC_QUANTILES")
        n_q = env.get("N_VALUE_QUANTILES")

        if not include_numeric:
            pipeline_name = "_reshard_no_numeric.yaml" if cfg.do_reshard else "_data_no_numeric.yaml"
        elif quantiles_fp:
            try:
                with open(quantiles_fp) as f:
                    custom_bins = yaml.safe_load(f) or {}
            except Exception as e:
                raise RuntimeError(f"Failed loading NUMERIC_QUANTILES_FP: {quantiles_fp}") from e
            env["NUMERIC_CUSTOM_BINS"] = json.dumps(custom_bins)
            pipeline_name = "_reshard_custom_bins.yaml" if cfg.do_reshard else "_data_custom_bins.yaml"
        else:
            if not quantiles_list and n_q:
                try:
                    n = int(n_q)
                    if n <= 0:
                        raise ValueError
                    env["NUMERIC_QUANTILES"] = str([(i + 1) / (n + 1) for i in range(n)])
                except ValueError as e:
                    raise ValueError(f"Invalid N_VALUE_QUANTILES={n_q}") from e
            pipeline_name = "_reshard_data.yaml" if cfg.do_reshard else "_data.yaml"

        if do_demo:
            env["MIN_SUBJECTS_PER_CODE"] = "2"
            env["MIN_EVENTS_PER_SUBJECT"] = "1"

        pipeline_config_fp = CONFIGS / pipeline_name
        cmd = [
            "MEDS_transform-pipeline",
            f"pipeline_config_fp={pipeline_config_fp!s}",
        ]
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, env=env, capture_output=True, check=False)
        if result.returncode != 0:  # pragma: no cover
            logger.error("Error running MEDS_transform-pipeline")
            logger.error(result.stdout.decode())
            logger.error(result.stderr.decode())
            raise RuntimeError("Error running MEDS_transform-pipeline")

        logger.info("Pre-MTD pre-processing done")
        done_fp.touch()

    # 1. Run MTD pre-processing
    logger.info("Running MTD pre-processing")
    done_fp = output_dir / ".done"

    if done_fp.exists():  # pragma: no cover
        logger.info("MTD pre-processing already done, skipping")
    else:
        env = copy.deepcopy(os.environ)

        cmd = [
            "MTD_preprocess",
            f"MEDS_dataset_dir={intermediate_dir!s}",
            f"output_dir={output_dir!s}",
        ]
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, env=env, capture_output=True, check=False)
        if result.returncode != 0:  # pragma: no cover
            logger.error("Error running MTD_preprocess")
            logger.error(result.stderr.decode())
            raise RuntimeError("Error running MTD_preprocess")

        logger.info("MTD pre-processing done")
        done_fp.touch()

    return
