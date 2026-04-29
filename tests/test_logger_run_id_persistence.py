"""Failing regression tests for the logger run-id save / restore flow.

Two distinct bugs are covered, intentionally co-located so the eventual fix can be
reviewed against both at once:

- **Issue #152 — save fires too late, resume never reattaches**:
  ``save_logger_run_ids`` is invoked at ``__main__.pretrain`` *after* ``trainer.fit(...)``
  returns. For a clean-completion run that's fine; for the case the helper actually
  exists to support (interrupted-and-resumed runs from OOM / SIGINT / OS reboot)
  ``trainer.fit`` never returns, the save line never executes, no
  ``wandb_run_id.txt`` is written, and the next ``MEICAR_pretrain do_resume=true``
  invocation finds nothing to restore — wandb spawns a fresh run and continuity is
  silently lost.

- **Issue #131 — generate save/restore paths don't connect**: ``generate_trajectories``
  *restores* run ids from ``cfg.model_initialization_dir`` but *saves* them to
  ``cfg.output_dir``. Nothing reads ``output_dir`` back on the next generation resume
  — it always re-restores from the training dir. The save is an orphan write.

The #152 test runs the *full real CLI* via subprocess: it ``Popen``-launches
``MEICAR_pretrain``, waits for wandb to materialize and a checkpoint to land, then
``SIGKILL``s the child (the OS-reboot / OOM-killer analog — atexit and signal
handlers do not run, so persistence has to come from a hook that fires inside
training itself, not from any clean-shutdown path). It then launches a second
``MEICAR_pretrain do_resume=true`` and asserts the resumed run reattaches to the
*same* wandb id, not a fresh one. That covers both halves of the bug — persistence
and resume — through the same code paths a user actually exercises.

Both tests use a real offline ``WandbLogger``, so upstream attribute-shape changes
in wandb / Lightning surface here too, not just inside helper-level dummy classes.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# Maximum time we'll wait for the first MEICAR_pretrain subprocess to (a) launch
# wandb and (b) save at least one checkpoint. CPU-bound demo training is fast (~ms
# per batch), but cold-start subprocess launch + dataset materialization + torch
# import all happen first; a generous ceiling avoids spurious CI flake without
# making the happy path slow (we exit the wait as soon as the conditions are met).
_PRE_KILL_TIMEOUT_SEC = 120.0
# Maximum time we'll wait for the resumed MEICAR_pretrain subprocess to materialize
# its WandbLogger (i.e. for an offline-run dir to exist whose run id matches the
# pre-kill saved id). Once that condition is observed, the resume side of the
# contract is proved and we can shut the second subprocess down.
_RESUME_TIMEOUT_SEC = 120.0
# Polling cadence used by both waits. Short enough that we exit the wait quickly
# once the disk-side condition is true; long enough that polling overhead is
# negligible against the underlying ms-scale training step.
_POLL_INTERVAL_SEC = 0.25


def _wait_for(predicate, timeout: float, *, on_timeout_msg: str) -> None:
    """Poll ``predicate()`` every ``_POLL_INTERVAL_SEC`` until true or timeout.

    Raised AssertionError carries ``on_timeout_msg`` so the failure mode is
    immediately legible — the test reader doesn't have to re-derive what we were
    waiting for from the predicate body.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(_POLL_INTERVAL_SEC)
    raise AssertionError(on_timeout_msg)


def _wandb_offline_run_dirs(save_dir: Path) -> list[Path]:
    """Return all ``offline-run-{date}-{id}`` directories under a wandb save_dir.

    Wandb names offline runs ``offline-run-YYYYMMDD_HHMMSS-{id}`` under
    ``<save_dir>/wandb/``. We glob across two levels so the lookup is robust to
    minor layout changes in upstream wandb.
    """
    return list(save_dir.glob("**/offline-run-*-*"))


def _read_wandb_id_from_dir(d: Path) -> str:
    """Extract the trailing wandb run id from an ``offline-run-{date}-{id}`` directory name."""
    return d.name.rsplit("-", 1)[-1]


# ---------------------------------------------------------------------------
# Issue #152 — kill-and-resume against the real CLI
# ---------------------------------------------------------------------------


def test_pretrain_real_cli_sigkill_resumes_to_same_wandb_run(
    preprocessed_dataset: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Bug #152, end-to-end: SIGKILL ``MEICAR_pretrain`` mid-training, then resume.

    The user-visible contract this test pins down:

    1. After ``MEICAR_pretrain`` reaches the training loop, the wandb run id is on
       disk under ``output_dir/loggers/wandb_run_id.txt``. Even a SIGKILL (OS
       reboot / OOM-killer analog — no signal handlers, no atexit, no clean
       teardown) leaves the file behind.
    2. A subsequent ``MEICAR_pretrain do_resume=true`` against the same
       ``output_dir`` reattaches to the *same* wandb run id (creates a second
       ``offline-run-{date}-{id}`` dir with the matching id), rather than spawning
       a fresh run.

    Today both halves fail. ``__main__.pretrain`` calls ``save_logger_run_ids``
    only after ``trainer.fit(...)`` returns, so SIGKILL means no file gets
    written. Then the resume call's ``apply_saved_logger_run_ids`` finds no saved
    id and lets WandbLogger spawn a fresh run.

    Going through the real CLI via subprocess is deliberate: it exercises the
    actual ``@hydra.main`` decorator path, real signal-handling under SIGKILL
    (the OS terminates the process — no Python cleanup runs at all), and the real
    resume flow. The previous in-process test could pass against fixes that
    accidentally relied on Python-level cleanup; this one cannot.
    """
    pytest.importorskip("wandb")

    output_dir = tmp_path_factory.mktemp("kill_resume_run")
    wandb_save_dir = output_dir / "loggers"

    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(tmp_path_factory.mktemp("wandb_state")),
            "WANDB_CONFIG_DIR": str(tmp_path_factory.mktemp("wandb_cfg")),
            "WANDB_CACHE_DIR": str(tmp_path_factory.mktemp("wandb_cache")),
            "WANDB_SILENT": "true",
            "WANDB_DISABLE_GIT": "true",
        }
    )

    # Common base command. Both run 1 and run 2 use the same config so the resume
    # config-diff check (validate_resume_directory) passes.
    base_cmd = [
        "MEICAR_pretrain",
        "--config-name=_demo_pretrain",
        f"output_dir={output_dir}",
        f"datamodule.config.tensorized_cohort_dir={preprocessed_dataset}",
        "trainer/logger=wandb",
        "trainer.logger.offline=true",
        # Bump so demo training doesn't complete before we get a chance to
        # SIGKILL it. Demo settings (overfit_batches=2, val_check_interval=1)
        # save a checkpoint after every batch, so a checkpoint will land within
        # the first second or so of real training; the bump just prevents the
        # job from finishing on its own.
        "trainer.max_epochs=200",
        # Same reason — disable max_steps so it doesn't gate on step count.
        "trainer.max_steps=-1",
    ]

    # ---- Run 1: launch, wait for wandb + checkpoint to land, then SIGKILL. ----

    proc1 = subprocess.Popen(
        base_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for(
            lambda: (
                bool(_wandb_offline_run_dirs(wandb_save_dir))
                and (output_dir / "checkpoints").exists()
                and any((output_dir / "checkpoints").glob("*.ckpt"))
            ),
            timeout=_PRE_KILL_TIMEOUT_SEC,
            on_timeout_msg=(
                "Timed out waiting for run 1 to materialize wandb + write a checkpoint. "
                "Without these the kill-and-resume contract can't be exercised: no live "
                "wandb run to capture the id from, and no checkpoint for resume to attach "
                "to. Subprocess returncode (None == still running): "
                f"{proc1.poll()}"
            ),
        )

        run1_dirs = _wandb_offline_run_dirs(wandb_save_dir)
        assert len(run1_dirs) == 1, (
            f"Expected exactly one offline-run dir during run 1; got {[d.name for d in run1_dirs]}"
        )
        live_wandb_id = _read_wandb_id_from_dir(run1_dirs[0])

        # SIGKILL is the case the helper exists for: OS reboot, OOM-killer, hard
        # crash. No Python-level teardown runs — atexit hooks, finally blocks,
        # Lightning's ``on_exception`` hook, wandb's offline-finalize thread, all
        # skipped. Persistence has to come from inside training, not cleanup.
        proc1.send_signal(signal.SIGKILL)
    finally:
        try:
            proc1.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc1.kill()
            proc1.wait(timeout=5)

    saved_fp = output_dir / "loggers" / "wandb_run_id.txt"
    assert saved_fp.is_file(), (
        f"After SIGKILL, {saved_fp} does not exist. save_logger_run_ids fires only "
        "after trainer.fit returns, so an interrupted run never persists its id "
        "(#152). The fix is to persist on on_train_start via a Lightning Callback, "
        "before any compute that might crash."
    )
    saved_id = saved_fp.read_text().strip()
    assert saved_id == live_wandb_id, (
        f"Saved id {saved_id!r} does not match the live wandb run id {live_wandb_id!r} "
        f"(from offline-run dir {run1_dirs[0].name!r}). save_logger_run_ids wrote a "
        "string but it isn't the live ``WandbLogger.experiment.id`` — the fix needs "
        "to read the live id, not anything cached."
    )

    # ---- Run 2: resume, wait for second offline-run to materialize, then stop. ----

    resume_cmd = [
        *base_cmd,
        "do_resume=true",
        # ``do_resume`` does not bypass the config-diff check, so the resume cfg
        # must match run 1's exactly (modulo ALLOWED_DIFFERENCE_KEYS in
        # training/files.py). Both invocations use ``base_cmd`` so this holds.
    ]

    proc2 = subprocess.Popen(
        resume_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for(
            lambda: any(
                _read_wandb_id_from_dir(d) == live_wandb_id
                for d in _wandb_offline_run_dirs(wandb_save_dir)
                if d not in run1_dirs
            ),
            timeout=_RESUME_TIMEOUT_SEC,
            on_timeout_msg=(
                f"Timed out waiting for the resumed run to attach to wandb id "
                f"{live_wandb_id!r}. Either the resume launched but spawned a fresh "
                "wandb id (apply_saved_logger_run_ids didn't read the saved id back), "
                "or the resume failed entirely. "
                f"offline-run dirs seen: {[d.name for d in _wandb_offline_run_dirs(wandb_save_dir)]}. "
                f"resume subprocess returncode (None == still running): {proc2.poll()}"
            ),
        )
    finally:
        # Shut the resume run down cleanly. We've already proved the contract; no
        # need to let it train to completion. SIGTERM gives Lightning a chance to
        # finalize the wandb run dir cleanly so we don't leave a half-written
        # state behind.
        proc2.send_signal(signal.SIGTERM)
        try:
            proc2.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc2.kill()
            proc2.wait(timeout=5)

    # Final cross-check: there should be at least two offline-run dirs and at
    # least two of them should carry ``live_wandb_id`` (one from run 1, one from
    # run 2 attaching to the same id). This is what wandb-side continuity looks
    # like on disk.
    final_dirs = _wandb_offline_run_dirs(wandb_save_dir)
    matching = [d for d in final_dirs if _read_wandb_id_from_dir(d) == live_wandb_id]
    assert len(matching) >= 2, (
        f"Expected at least two offline-run dirs tagged with {live_wandb_id!r} "
        f"(run 1 + resume), got {[d.name for d in matching]} (out of "
        f"{[d.name for d in final_dirs]}). The resume spawned a fresh wandb run "
        "instead of attaching to the saved id."
    )


# ---------------------------------------------------------------------------
# Issue #131 — generate's saved run id must be readable on the next resume
# ---------------------------------------------------------------------------


def test_generate_trajectories_resume_reads_output_dir_id_not_training_dir_id(
    pretrained_model: Path,
    preprocessed_dataset_with_task: tuple[Path, Path, str],
    tmp_path_factory: pytest.TempPathFactory,
):
    """Bug #131: a saved generation run id under ``output_dir`` must be honored on the next generation resume
    — not silently shadowed by the training run id under ``model_initialization_dir``.

    Drives the full ``MEICAR_generate_trajectories`` CLI via subprocess (matches existing
    end-to-end test style). Pre-seeds two distinguishable wandb run ids: one in
    ``model_initialization_dir/loggers/`` (training save-point) and one in
    ``output_dir/loggers/`` (a prior generation run's save-point). Runs generation. Asserts
    the output_dir's saved id is preserved, proving:

    1. ``apply_saved_logger_run_ids`` read ``output_dir`` (not just
       ``model_initialization_dir``);
    2. Lightning's WandbLogger was instantiated with ``id=<prior_gen_id>`` and attached to
       the corresponding offline run;
    3. The trailing ``save_logger_run_ids(trainer.loggers, output_dir)`` then re-saved the
       same id back to ``output_dir/loggers/wandb_run_id.txt``.

    Under the bug, step 1 reads from ``model_initialization_dir`` instead, so the run
    attaches to the *training* id, and the trailing save **overwrites** ``output_dir``'s
    saved id with the training id. The assertion ``output_dir id is unchanged`` catches
    that overwrite.

    Going through the CLI (subprocess) instead of an in-process driver guards against
    fixes that patch the helper signature but forget to update the actual call site in
    ``__main__.generate_trajectories`` — the failure mode that would silently slip past
    a unit test on the helper alone.
    """
    pytest.importorskip("wandb")

    cohort_dir, task_root_dir, task_name = preprocessed_dataset_with_task

    # Distinguishable, wandb-valid 8-char ids. Wandb's ``run.id`` must be base36-y to pass
    # ``wandb.sdk.lib.runid.check_id``; alphanumeric works.
    training_id = "trainabc"
    prior_gen_id = "priorxyz"

    # Copy the session-shared pretrained_model dir to a private location so we can
    # mutate its loggers/ directory without affecting other tests.
    init_dir = tmp_path_factory.mktemp("init_dir")
    shutil.copytree(pretrained_model, init_dir, dirs_exist_ok=True)
    init_loggers = init_dir / "loggers"
    init_loggers.mkdir(parents=True, exist_ok=True)
    (init_loggers / "wandb_run_id.txt").write_text(training_id)

    # Pre-seed the generation output_dir with the prior-generation save-point.
    output_dir = tmp_path_factory.mktemp("generation_output")
    out_loggers = output_dir / "loggers"
    out_loggers.mkdir(parents=True, exist_ok=True)
    (out_loggers / "wandb_run_id.txt").write_text(prior_gen_id)

    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(tmp_path_factory.mktemp("wandb_state")),
            "WANDB_CONFIG_DIR": str(tmp_path_factory.mktemp("wandb_cfg")),
            "WANDB_CACHE_DIR": str(tmp_path_factory.mktemp("wandb_cache")),
            "WANDB_SILENT": "true",
            "WANDB_DISABLE_GIT": "true",
        }
    )

    cmd = [
        "MEICAR_generate_trajectories",
        "--config-name=_demo_generate_trajectories",
        f"output_dir={output_dir}",
        f"model_initialization_dir={init_dir}",
        f"datamodule.config.tensorized_cohort_dir={cohort_dir}",
        f"datamodule.config.task_labels_dir={task_root_dir / task_name}",
        "datamodule.batch_size=2",
        "trainer/logger=wandb",
        "trainer.logger.offline=true",
        f"trainer.logger.save_dir={output_dir / 'wandb_save'}",
        "trainer.logger.project=meds_eic_ar_test",
        # Drop the ``${hydra:runtime.choices.lightning_module/model}`` interpolation —
        # the generate config has no ``lightning_module/model`` group choice, so the tag
        # resolves to ``None`` and recent wandb / pydantic versions reject non-string tags.
        # Tags are not relevant to this test's invariant.
        "~trainer.logger.tags",
    ]

    result = subprocess.run(cmd, capture_output=True, env=env, check=False)
    if result.returncode != 0:
        raise ValueError(
            "MEICAR_generate_trajectories failed:\n"
            f"stdout:\n{result.stdout.decode()}\nstderr:\n{result.stderr.decode()}"
        )

    saved = (out_loggers / "wandb_run_id.txt").read_text().strip()
    assert saved == prior_gen_id, (
        f"output_dir's wandb_run_id.txt was overwritten with {saved!r} (expected the "
        f"pre-existing generation id {prior_gen_id!r}). Bug #131: "
        f"``apply_saved_logger_run_ids`` reads only from ``model_initialization_dir``, "
        f"so wandb attached to the training id ({training_id!r}) and the trailing "
        "save_logger_run_ids clobbered the output_dir save-point."
    )

    # Cross-check: the offline wandb run dir for this generation run should be tagged with
    # the prior_gen_id, proving wandb actually attached to it (not a coincidence of file
    # contents).
    wandb_run_dirs = list((output_dir / "wandb_save").glob("**/offline-run-*-*"))
    assert wandb_run_dirs, "Expected an offline wandb run dir under output_dir/wandb_save."
    matching = [d for d in wandb_run_dirs if d.name.endswith(f"-{prior_gen_id}")]
    assert matching, (
        f"No wandb offline-run dir tagged with {prior_gen_id!r}; "
        f"found: {[d.name for d in wandb_run_dirs]}. WandbLogger did not attach to the "
        "saved generation id — the apply path read the wrong dir."
    )
