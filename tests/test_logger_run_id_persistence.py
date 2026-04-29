"""Failing regression tests for the logger run-id save / restore flow.

Two distinct bugs are covered, intentionally co-located so the eventual fix can be
reviewed against both at once:

- **Issue #152 — save fires too late**: ``save_logger_run_ids`` is invoked at
  ``__main__.pretrain`` *after* ``trainer.fit(...)`` returns. For a clean-completion
  run that's fine; for the case the helper actually exists to support
  (interrupted-and-resumed runs from OOM / SIGINT / OS reboot) ``trainer.fit`` never
  returns, the save line never executes, no ``wandb_run_id.txt`` is written, and the
  next ``MEICAR_pretrain do_resume=True`` invocation finds nothing to restore — wandb
  spawns a fresh run and continuity is lost.

- **Issue #131 — generate save/restore paths don't connect**: ``generate_trajectories``
  *restores* run ids from ``cfg.model_initialization_dir`` but *saves* them to
  ``cfg.output_dir``. Nothing reads ``output_dir`` back on the next generation resume
  — it always re-restores from the training dir. The save is an orphan write.

Both tests are deliberately end-to-end and use a **real offline ``WandbLogger``**, not
a stub. The point is to catch this bug *and* any nearby ones — including upstream
attribute-shape changes in wandb / Lightning, callback wiring mistakes that go through
the helper without firing under real Lightning fit semantics, or fixes that patch the
helper signature but forget to update the CLI call sites.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Issue #152 — wandb run id must be on disk before trainer.fit returns
# ---------------------------------------------------------------------------


def test_pretrain_persists_wandb_run_id_when_real_fit_crashes_mid_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preprocessed_dataset: Path,
):
    """Bug #152: drive a *real* ``Trainer.fit`` for one training batch and crash from inside a callback. The
    wandb run id must be on disk before the exception propagates.

    Today ``__main__.pretrain`` calls ``save_logger_run_ids`` *after* ``trainer.fit(...)``
    returns. An interrupted run (the case the helper exists for) never persists its id
    and a subsequent ``MEICAR_pretrain do_resume=True`` finds no saved id — wandb spawns
    a fresh run and continuity is silently lost.

    This test is deliberately not a mocked-fit test: it builds a real Lightning Trainer
    from the demo pretrain config, attaches a real offline ``WandbLogger``, and lets fit
    run a real training batch. A ``_KillSwitch`` callback then raises ``RuntimeError``
    from ``on_train_batch_end`` (mimicking an OOM / driver fault). Running the test
    against a fix that wires save_logger_run_ids onto the wrong hook, or onto a callback
    list real Lightning never iterates, will still fail here even though a mocked-fit
    test would pass — which is the whole point.
    """
    pytest.importorskip("wandb")

    # Keep wandb fully offline so the test is hermetic — no remote contact, no auth, no
    # network flake. ``WANDB_DIR`` redirects state into the test's tmp_path so we don't
    # leave a ``./wandb/`` directory behind in the repo root.
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb_state"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb_cfg"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb_cache"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DISABLE_GIT", "true")

    output_dir = tmp_path / "interrupted_run"
    output_dir.mkdir()

    import lightning.pytorch as L
    from lightning.pytorch.callbacks import Callback

    class _KillSwitch(Callback):
        """Raises after the first training batch, mimicking a mid-fit OOM / driver fault.

        Hook choice (``on_train_batch_end``) is deliberate: it fires *after* Lightning has
        called ``on_train_start`` on every other callback, so any save-run-ids callback wired
        to ``on_train_start`` has already had its chance to persist the id by the time the
        crash hits. If the fix wires persistence to a hook that runs only on clean fit
        completion (e.g., ``on_fit_end``), the file won't be on disk when this test asserts
        — which is exactly the property #152 is asking the fix to guarantee.
        """

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            raise RuntimeError("simulated mid-training crash for #152 regression test")

    # Patch the ``instantiate`` symbol that ``__main__.pretrain`` imports so the *real*
    # Trainer Lightning builds gets our kill switch prepended. All other ``instantiate(...)``
    # calls (datamodule, lightning module) pass through unchanged. Prepending (vs appending)
    # makes the kill switch fire before ``ModelCheckpoint``'s post-batch hooks so we don't
    # accidentally exercise the checkpoint-save error path on the way out.
    import MEDS_EIC_AR.__main__ as main_mod

    orig_instantiate = main_mod.instantiate

    def patched_instantiate(cfg, *args, **kwargs):
        obj = orig_instantiate(cfg, *args, **kwargs)
        if isinstance(obj, L.Trainer):
            obj.callbacks.insert(0, _KillSwitch())
        return obj

    monkeypatch.setattr(main_mod, "instantiate", patched_instantiate)

    from hydra import compose, initialize_config_module

    from MEDS_EIC_AR.__main__ import pretrain

    with initialize_config_module(config_module="MEDS_EIC_AR.configs", version_base=None):
        cfg = compose(
            config_name="_demo_pretrain",
            overrides=[
                f"output_dir={output_dir}",
                f"datamodule.config.tensorized_cohort_dir={preprocessed_dataset}",
                "trainer/logger=wandb",
                "trainer.logger.offline=true",
                f"trainer.logger.save_dir={tmp_path / 'wandb_save'}",
                "trainer.logger.project=meds_eic_ar_test",
                # Drop the ``${hydra:runtime.choices...}`` interpolation in the wandb config
                # — under ``hydra.compose`` (no ``hydra.main``) ``HydraConfig`` is not set,
                # so the interpolation would fail to resolve. Tags are not relevant to this
                # test's invariant.
                "~trainer.logger.tags",
            ],
        )

    with pytest.raises(RuntimeError, match="simulated mid-training crash"):
        pretrain.__wrapped__(cfg)

    saved_fp = output_dir / "loggers" / "wandb_run_id.txt"
    assert saved_fp.is_file(), (
        "wandb_run_id.txt was not written before fit raised — save_logger_run_ids fires only "
        "after trainer.fit returns, so an interrupted run never persists its id (#152). The "
        "fix is to persist on on_train_start via a Lightning Callback."
    )
    saved_id = saved_fp.read_text().strip()
    assert saved_id, f"wandb_run_id.txt was written but is empty (contents: {saved_id!r})"

    # Cross-check: the saved id is the actual offline wandb run id. Look for a wandb
    # offline-run directory under save_dir whose suffix matches the saved id. Catches a
    # broken fix that writes some unrelated string instead of the live ``logger.experiment.id``.
    wandb_run_dirs = list((tmp_path / "wandb_save").glob("**/run-*-*"))
    assert wandb_run_dirs, "Expected at least one wandb offline-run directory to exist."
    matching = [d for d in wandb_run_dirs if d.name.endswith(f"-{saved_id}")]
    assert matching, (
        f"Saved wandb id {saved_id!r} does not match any actual wandb run dir under "
        f"{tmp_path / 'wandb_save'}: {[d.name for d in wandb_run_dirs]}. The fix wrote a "
        "string but it isn't the live ``WandbLogger.experiment.id``."
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
    wandb_run_dirs = list((output_dir / "wandb_save").glob("**/run-*-*"))
    assert wandb_run_dirs, "Expected an offline wandb run dir under output_dir/wandb_save."
    matching = [d for d in wandb_run_dirs if d.name.endswith(f"-{prior_gen_id}")]
    assert matching, (
        f"No wandb offline-run dir tagged with {prior_gen_id!r}; "
        f"found: {[d.name for d in wandb_run_dirs]}. WandbLogger did not attach to the "
        "saved generation id — the apply path read the wrong dir."
    )
