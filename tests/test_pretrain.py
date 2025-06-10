import shutil
import subprocess
from pathlib import Path

import pytest


def run_and_check(cmd: list[str]) -> dict[str, str]:
    """Runs a command and checks the output.

    Args:
        cmd: The command to run. This should be a list of strings, which will be run through subprocess.run in
            shell=False mode.

    Raises:
        ValueError: If the command fails.

    Returns:
        A dictionary with the keys "stdout" and "stderr", containing the output of the command.
    """

    cmd_out = subprocess.run(cmd, capture_output=True, check=False)

    out = {"stdout": cmd_out.stdout.decode(), "stderr": cmd_out.stderr.decode()}

    if cmd_out.returncode == 0:
        return

    err = [f"Command yielded code {cmd_out.returncode}", "Stdout:", out["stdout"], "Stderr:", out["stderr"]]
    raise ValueError("\n".join(err))


def test_pretrain_runs(pretrained_model: Path):
    out_files = list(pretrained_model.rglob("*.*"))
    assert len(out_files) > 0


def test_resumes(
    pretrained_model: Path,
    tmp_path_factory: pytest.TempPathFactory,
    preprocessed_dataset: Path,
):
    """Test that the pretraining can be resumed from a checkpoint.

    The pre-trained model output directory looks like this:
    pretrained_model
    ├── .logs
    │   ├── .hydra
    │   │   ├── config.yaml
    │   │   ├── hydra.yaml
    │   │   └── overrides.yaml
    │   └── __main__.log
    ├── best_model.ckpt
    ├── checkpoints
    │   ├── epoch=0-step=2.ckpt
    │   ├── epoch=1-step=4.ckpt
    │   └── last.ckpt
    ├── config.yaml
    └── loggers
        └── csv
            └── version_0
                ├── hparams.yaml
                └── metrics.csv

    We need to remove the following files before resuming:
      - best_model.ckpt
      - checkpoints/last.ckpt
      - checkpoints/epoch=1-step=4.ckpt

    This will ensure that the pretraining resumes from epoch 0, step 2. We'll also remove the logs to ensure
    that we have clean log files.
    """

    # 1. Copy the pretrained model to a temporary directory and remove some of the completed files.
    resume_from_dir = tmp_path_factory.mktemp("resume_from")

    shutil.copytree(pretrained_model, resume_from_dir, dirs_exist_ok=True)

    files_to_remove = ["best_model.ckpt", "checkpoints/last.ckpt", "checkpoints/epoch=1-step=4.ckpt"]

    for file in files_to_remove:
        fp = resume_from_dir / file
        if not fp.is_file():
            raise FileNotFoundError(f"File {fp} does not exist, cannot remove it for resuming.")
        fp.unlink()

    # 2. Remove the logs directory to ensure clean logs.
    logs_dir = resume_from_dir / ".logs"
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Logs directory {logs_dir} does not exist, cannot remove it for resuming.")
    shutil.rmtree(logs_dir)

    # 3. Run the pretraining with the modified directory.

    command = [
        "MEICAR_pretrain",
        "--config-name=_demo_pretrain",
        f"output_dir={resume_from_dir!s}",
        f"datamodule.config.tensorized_cohort_dir={preprocessed_dataset!s}",
        "do_resume=True",
        "do_overwrite=False",
    ]

    run_and_check(command)

    version_1_log_dir = resume_from_dir / "loggers" / "csv" / "version_1"
    assert version_1_log_dir.is_dir(), "No new log directory created after resuming."

    best_model_ckpt = resume_from_dir / "best_model.ckpt"
    assert best_model_ckpt.is_file(), "No new best model checkpoint created after resuming."

    last_ckpt = resume_from_dir / "checkpoints" / "last.ckpt"
    assert last_ckpt.is_file(), "No new last checkpoint created after resuming."
