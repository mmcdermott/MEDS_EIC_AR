#!/usr/bin/env python
"""DGX Spark local smoke test + HF-vs-SGLang runtime baseline for the SGLang backend.

This script is **not** part of the pytest suite — it's a reproducible, single-command
integration test against a real SGLang ``Engine`` on the DGX Spark (aarch64 + NVIDIA GB10,
SM 12.1). It mirrors the setup of ``tests/grammar/test_cli_sglang.py`` (same synthetic
grammar data, same ``MEICAR_generate_trajectories`` CLI) and additionally does a matched
HF-vs-SGLang runtime comparison on the same pretrained checkpoint so we can report a
baseline speedup number on this hardware.

Run::

    PATH=$(pwd)/.venv/bin:$PATH .venv/bin/python scripts/dgx_spark_sglang_smoke.py

What it does
------------

1. Materializes a synthetic grammar MEDS dataset in a fresh temp directory (via
   ``tests.grammar._meds.build_grammar_meds_dataset`` — identical to the fixture the gated
   grammar test uses).
2. Runs ``MEICAR_process_data`` to tensorize it.
3. Runs ``MEICAR_pretrain`` on a slightly-bigger-than-demo model (matching the grammar
   fixture config) so greedy decoding has a realistic chance of staying grammar-valid.
4. Runs ``MEICAR_generate_trajectories`` twice on the same held-out prompts — once with
   ``backend=hf`` and once with ``backend=sglang_demo`` (CUDA graph capture disabled for
   fast cold-start). Both runs use greedy decoding so output is fully deterministic up to
   floating-point drift.
5. Validates that every generated grammar token in every held-out sample is a valid FSM
   continuation from the prompt's end-state (strict 100%-validity gate, same as the CLI
   grammar test).
6. Reports wall-clock numbers for each backend plus total generated tokens, so you can eyeball
   the SGLang speedup on the DGX Spark.

Known DGX-Spark-specific environment notes (these are **not** fixed in-repo because they'd
regress other hardware — flag to the reviewer and handle at install time):

- The default ``uv sync --extra sglang`` resolution brings CPU-only torch on aarch64. Install
  a CUDA-enabled torch wheel matching ``sgl_kernel``'s ABI::

      uv pip install --torch-backend=cu129 --reinstall torch==2.9.1 torchaudio==2.9.1 \\
          torchvision==0.24.1

  torch 2.11 + cu130 (what ``UV_TORCH_BACKEND=auto`` selects on this box) has an ABI
  mismatch with ``sgl_kernel==0.3.21`` — ``c10::cuda::c10_cuda_check_implementation``
  symbol differs. Cu129 + torch 2.9.1 is the combo sgl_kernel 0.3.21 was built against.

- torch 2.9.1 warns that GB10 (SM 12.1) exceeds its max compiled capability (SM 12.0). The
  warning is harmless for the kernels exercised here (matmul falls back to PTX JIT), and
  sgl_kernel ships an SM 10.0 variant that loads cleanly on SM 12.1 via the same fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Make ``tests.grammar._meds`` importable as a package. The script lives in ``scripts/``
# alongside the ``tests/`` top-level, so prepend the repo root to ``sys.path``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polars as pl  # noqa: E402
from meds import held_out_split  # noqa: E402

from tests.grammar._grammar import GrammarFSM  # noqa: E402
from tests.grammar._meds import (  # noqa: E402
    GRAMMAR_TASK_NAME,
    build_grammar_meds_dataset,
    grammar_tokens_from_output_df,
    prompt_grammar_tokens_by_subject,
)

# ----------------------------------------------------------------------------
# Pipeline-stage helpers
# ----------------------------------------------------------------------------


def _run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> None:
    """Run ``cmd`` through subprocess; dump stdout+stderr on failure."""
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, check=False, env=env, cwd=cwd)
    dt = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"--- stdout (exit {result.returncode}, {dt:.1f}s) ---")
        print(result.stdout.decode(errors="replace"))
        print("--- stderr ---")
        print(result.stderr.decode(errors="replace"))
        raise SystemExit(f"Command failed with code {result.returncode}")
    print(f"... done in {dt:.1f}s")


def build_raw_meds(root: Path, *, n_train: int, n_tuning: int, n_held_out: int) -> tuple[Path, Path]:
    """Write the synthetic grammar MEDS dataset under ``root`` and return the task-labels dir."""
    print(f"\n[1/5] Building synthetic grammar MEDS dataset (train={n_train}, held_out={n_held_out}).")
    build_grammar_meds_dataset(
        root,
        n_train=n_train,
        n_tuning=n_tuning,
        n_held_out=n_held_out,
    )
    task_dir = root / "task_labels" / GRAMMAR_TASK_NAME
    assert (task_dir / f"{held_out_split}.parquet").is_file(), f"missing {task_dir}/{held_out_split}.parquet"
    return root, task_dir


def preprocess(meds_dir: Path, out_root: Path) -> Path:
    """``MEICAR_process_data`` → tensorized output directory."""
    print("\n[2/5] Preprocessing (MEICAR_process_data).")
    out_dir = out_root / "output"
    _run(
        [
            "MEICAR_process_data",
            f"input_dir={meds_dir!s}",
            f"intermediate_dir={out_root / 'intermediate'!s}",
            f"output_dir={out_dir!s}",
            "do_demo=True",
            "do_reshard=False",
        ]
    )
    return out_dir


def pretrain(
    preprocessed_dir: Path,
    out_dir: Path,
    *,
    max_epochs: int,
    max_seq_len: int,
    check_val_every_n_epoch: int,
) -> Path:
    """``MEICAR_pretrain`` with a small Llama (matches the grammar-fixture config).

    ``check_val_every_n_epoch`` must be ``<= max_epochs`` or the run finishes without ever
    validating, ``ModelCheckpoint`` stays empty, and ``MEICAR_pretrain`` raises
    ``ValueError("No best checkpoint reported.")``. Defaults to ``max_epochs // 8`` (bounded
    below at 1) so scaled-down smoke runs still validate at least a few times.
    """
    print(f"\n[3/5] Pretraining ({max_epochs} epochs, max_seq_len={max_seq_len}).")
    _run(
        [
            "MEICAR_pretrain",
            "--config-name=_demo_pretrain",
            f"output_dir={out_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={preprocessed_dir!s}",
            "datamodule.batch_size=32",
            f"trainer.max_epochs={max_epochs}",
            "trainer.overfit_batches=0",
            "trainer.callbacks.early_stopping.patience=100000",
            "trainer.val_check_interval=1.0",
            f"trainer.check_val_every_n_epoch={check_val_every_n_epoch}",
            "trainer.detect_anomaly=False",
            "lightning_module.model.do_demo=false",
            f"max_seq_len={max_seq_len}",
            "lightning_module.model.gpt_kwargs.num_attention_heads=4",
            "lightning_module.model.gpt_kwargs.attention_head_dim=32",
        ]
    )
    return out_dir


def generate(
    *,
    backend: str,
    pretrained_dir: Path,
    preprocessed_dir: Path,
    task_dir: Path,
    output_dir: Path,
    n_trajectories: int,
    batch_size: int,
    rolling_max_new_tokens: int,
    do_sample: bool,
) -> float:
    """Invoke ``MEICAR_generate_trajectories`` with the given backend; return wall-clock seconds."""
    args = [
        "MEICAR_generate_trajectories",
        "--config-name=_demo_generate_trajectories",
        f"output_dir={output_dir!s}",
        f"model_initialization_dir={pretrained_dir!s}",
        f"datamodule.config.tensorized_cohort_dir={preprocessed_dir!s}",
        f"datamodule.config.task_labels_dir={task_dir!s}",
        f"datamodule.batch_size={batch_size}",
        "trainer=demo",
        f"inference.N_trajectories_per_task_sample={n_trajectories}",
        "inference.generate_for_splits=[held_out]",
        f"inference.do_sample={str(do_sample).lower()}",
        f"rolling_generation.max_new_tokens={rolling_max_new_tokens}",
    ]
    if backend != "hf":
        args.append(f"backend={backend}")
    t0 = time.perf_counter()
    _run(args)
    return time.perf_counter() - t0


# ----------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------


def _fsm_walk_from_prompt(prompt_tokens: list[int], generated: list[int]) -> tuple[int, int]:
    """Return (valid_count, total) for ``generated`` walked from the FSM state after the prompt."""
    fsm = GrammarFSM()
    for i, tok in enumerate(prompt_tokens):
        if fsm.step(tok) is None:
            raise AssertionError(f"Prompt token {tok} at {i} is not grammar-valid: {prompt_tokens}")
    valid = 0
    for tok in generated:
        if fsm.step(tok) is None:
            break
        valid += 1
    return valid, len(generated)


@dataclass
class ValidationResult:
    """Outcome of the strict-grammar gate + token-count stats for one backend run."""

    backend: str
    total_samples: int
    failing_samples: list[str]
    total_new_grammar_tokens: int
    rolling_samples: int
    wall_clock_seconds: float

    @property
    def pass_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return 1 - (len(self.failing_samples) / self.total_samples)


def validate_and_summarize(
    *,
    backend: str,
    output_root: Path,
    raw_meds_dir: Path,
    task_dir: Path,
    wall_clock_seconds: float,
    max_seq_len: int,
) -> ValidationResult:
    """Apply the strict grammar-validity gate and count rolling/total tokens."""
    prompts_by_subject = prompt_grammar_tokens_by_subject(raw_meds_dir, task_dir, held_out_split)

    held_out_dir = output_root / held_out_split
    parquets = sorted(held_out_dir.glob("*.parquet"))
    assert parquets, f"no {backend} output parquets under {held_out_dir}"

    failures: list[str] = []
    total_samples = 0
    total_new_grammar_tokens = 0
    rolling_samples = 0
    for pq in parquets:
        df = pl.read_parquet(pq, use_pyarrow=True)
        # Sort rows by time so we can measure per-subject total length against max_seq_len to
        # identify samples that exercised the rolling-generation path.
        sorted_df = df.sort(["subject_id", "time"])
        for _subject_id_tup, subject_df in sorted_df.group_by("subject_id", maintain_order=True):
            if len(subject_df) > max_seq_len:
                rolling_samples += 1
        tokens_by_subject = grammar_tokens_from_output_df(df)
        for subject_id, tokens in tokens_by_subject.items():
            if not tokens:
                continue
            total_samples += 1
            total_new_grammar_tokens += len(tokens)
            valid, total = _fsm_walk_from_prompt(prompts_by_subject[subject_id], tokens)
            if valid != total:
                failures.append(
                    f"[{backend}] traj={pq.stem} subject={subject_id}: {valid}/{total} valid "
                    f"(first invalid at gen pos {valid}): prompt={prompts_by_subject[subject_id]}, "
                    f"gen={tokens}"
                )

    return ValidationResult(
        backend=backend,
        total_samples=total_samples,
        failing_samples=failures,
        total_new_grammar_tokens=total_new_grammar_tokens,
        rolling_samples=rolling_samples,
        wall_clock_seconds=wall_clock_seconds,
    )


def print_summary(results: list[ValidationResult]) -> None:
    print("\n" + "=" * 72)
    print("DGX Spark SGLang smoke test — summary")
    print("=" * 72)
    for r in results:
        status = "PASS" if not r.failing_samples else "FAIL"
        print(
            f"  {r.backend:16s}  wall={r.wall_clock_seconds:7.2f}s  "
            f"samples={r.total_samples:3d}  new_grammar_toks={r.total_new_grammar_tokens:4d}  "
            f"rolling={r.rolling_samples:3d}  grammar_gate={status}"
        )

    if len(results) == 2:
        hf_r, sgl_r = results[0], results[1]
        if sgl_r.wall_clock_seconds > 0:
            ratio = hf_r.wall_clock_seconds / sgl_r.wall_clock_seconds
            print(
                f"\n  wall-clock(hf) / wall-clock(sglang) = {ratio:.2f}x  "
                f"(> 1 means SGLang faster end-to-end; < 1 means engine startup dominates)"
            )

    for r in results:
        if r.failing_samples:
            print(f"\n  Failures for {r.backend}:")
            for f in r.failing_samples[:5]:
                print(f"    {f}")
            if len(r.failing_samples) > 5:
                print(f"    ... ({len(r.failing_samples) - 5} more)")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, default=None, help="Staging dir (default: a fresh tempdir).")
    ap.add_argument("--keep-work-dir", action="store_true", help="Don't delete the staging dir on exit.")
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-tuning", type=int, default=16)
    ap.add_argument("--n-held-out", type=int, default=16)
    ap.add_argument("--n-trajectories", type=int, default=8)
    ap.add_argument("--generate-batch-size", type=int, default=3)
    ap.add_argument("--max-seq-len", type=int, default=16)
    ap.add_argument("--rolling-max-new-tokens", type=int, default=30)
    ap.add_argument("--pretrain-max-epochs", type=int, default=400)
    ap.add_argument(
        "--pretrain-check-val-every-n-epoch",
        type=int,
        default=None,
        help="Default: max(1, pretrain_max_epochs // 8). Must be <= max_epochs so at least one "
        "validation happens (the pretrain CLI raises if no best-ckpt is ever recorded).",
    )
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["hf", "sglang_demo"],
        help="Backends to run (in order). Default runs both so you get a runtime comparison.",
    )
    ap.add_argument("--skip-pretrain", type=Path, default=None, help="Reuse a prior pretrain output dir.")
    ap.add_argument(
        "--summary-json", type=Path, default=None, help="Write summary numbers to this JSON file."
    )
    args = ap.parse_args()

    # Fresh tempdir unless the caller provided one. Persisting via ``--keep-work-dir`` keeps
    # the export + logs inspectable after a failure.
    if args.work_dir is None:
        tmp = tempfile.mkdtemp(prefix="dgx_spark_sglang_smoke.")
        work_dir = Path(tmp)
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    # Put the project's .venv/bin on PATH so ``MEICAR_*`` scripts resolve. Mirrors the manual
    # ``PATH=$(pwd)/.venv/bin:$PATH`` the docstring's run line prepends.
    venv_bin = REPO_ROOT / ".venv" / "bin"
    env = os.environ.copy()
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    print(f"Working under {work_dir}")
    raw_meds_dir, task_dir = build_raw_meds(
        work_dir / "meds_raw",
        n_train=args.n_train,
        n_tuning=args.n_tuning,
        n_held_out=args.n_held_out,
    )

    preprocessed_dir = preprocess(raw_meds_dir, work_dir / "meds_proc")

    if args.skip_pretrain is not None:
        pretrained_dir = args.skip_pretrain
        print(f"\n[3/5] SKIPPED pretrain; reusing {pretrained_dir}")
    else:
        check_val_every_n_epoch = (
            args.pretrain_check_val_every_n_epoch
            if args.pretrain_check_val_every_n_epoch is not None
            else max(1, args.pretrain_max_epochs // 8)
        )
        pretrained_dir = pretrain(
            preprocessed_dir,
            work_dir / "pretrained",
            max_epochs=args.pretrain_max_epochs,
            max_seq_len=args.max_seq_len,
            check_val_every_n_epoch=check_val_every_n_epoch,
        )

    results: list[ValidationResult] = []
    for backend in args.backends:
        backend_out = work_dir / f"generated_{backend}"
        print(f"\n[4/5] Generating via backend={backend}")
        wall_clock = generate(
            backend=backend,
            pretrained_dir=pretrained_dir,
            preprocessed_dir=preprocessed_dir,
            task_dir=task_dir,
            output_dir=backend_out,
            n_trajectories=args.n_trajectories,
            batch_size=args.generate_batch_size,
            rolling_max_new_tokens=args.rolling_max_new_tokens,
            do_sample=False,  # greedy — strict validity gate requires determinism
        )
        print(f"\n[5/5] Validating {backend} output")
        result = validate_and_summarize(
            backend=backend,
            output_root=backend_out,
            raw_meds_dir=raw_meds_dir,
            task_dir=task_dir,
            wall_clock_seconds=wall_clock,
            max_seq_len=args.max_seq_len,
        )
        results.append(result)

    print_summary(results)

    if args.summary_json is not None:
        args.summary_json.write_text(
            json.dumps(
                [
                    {
                        "backend": r.backend,
                        "wall_clock_seconds": r.wall_clock_seconds,
                        "total_samples": r.total_samples,
                        "total_new_grammar_tokens": r.total_new_grammar_tokens,
                        "rolling_samples": r.rolling_samples,
                        "failing_samples": r.failing_samples,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        print(f"\nSummary written to {args.summary_json}")

    any_failed = any(r.failing_samples for r in results)
    if not args.keep_work_dir and args.work_dir is None:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Cleaned up {work_dir}")
    else:
        print(f"Left working dir at {work_dir}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
