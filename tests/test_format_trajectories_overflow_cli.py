r"""End-to-end CLI regression for issue #154 — polluted-bin handling at the validator layer.

Complements the unit-level doctests on :func:`validate_timeline_delta_bins_in_int64_range`
inside ``finalize.py``: those exercise the validator's own logic with synthetic
``pl.DataFrame``\ s, but they wouldn't have caught the original #154 bug shape (a polluted
bin coming through the real preprocessing → CLI path) because they only test code added in
the same PR. These tests pin the contract end-to-end:

1. **Polluted ``TIMELINE//DELTA`` bin** → ``MEICAR_generate_trajectories`` must refuse to
   start, surfacing the validator's ``ValueError`` on stderr.
2. **Polluted *non-delta* bin** (e.g. an HR bin with a huge ``value_mean`` — a measurement,
   not a time encoding) → CLI must complete normally; the validator only gates delta bins.

Both tests mutate the real ``codes.parquet`` produced by the preprocessing fixture and
re-tensorize the cohort directory, then drive the actual ``MEICAR_generate_trajectories``
CLI via subprocess. They share a helper that does the parquet surgery so the two failure
modes are tested through the same lens.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import polars as pl

from MEDS_EIC_AR.generation.finalize import TIMELINE_DELTA_TOKEN

if TYPE_CHECKING:
    from pathlib import Path


# Big enough that ``value_mean * 31556926 * 1e6`` overflows Int64. The validator's
# threshold is ``(2**63 - 1) / (seconds_per_year * 1e6) ~= 2.92e5`` years; 1e30 clears
# that by 25 orders of magnitude so the test isn't sensitive to the exact unit.
_OVERFLOWING_VALUES_SUM = 1e30


def _pollute_bin(codes_fp: Path, code_filter_expr: pl.Expr) -> str:
    """Mutate one row of ``codes.parquet`` in place to have an overflow-sized ``value_mean``.

    ``value_mean`` is computed by ``get_code_metadata`` as ``values/sum / values/n_occurrences``,
    so we set ``values/sum`` huge and ``values/n_occurrences`` to 1 on the targeted row.
    Returns the polluted code string so callers can assert it appears in the error message.
    """
    codes = pl.read_parquet(codes_fp, use_pyarrow=True)
    target = codes.filter(code_filter_expr).head(1)
    if target.height != 1:
        raise AssertionError(
            f"Test setup: filter {code_filter_expr} matched {target.height} rows in "
            f"{codes_fp} (expected exactly 1)."
        )
    polluted_code = target["code"][0]

    sum_dtype = codes.schema["values/sum"]
    nocc_dtype = codes.schema["values/n_occurrences"]
    polluted = codes.with_columns(
        pl.when(pl.col("code") == polluted_code)
        .then(pl.lit(_OVERFLOWING_VALUES_SUM, dtype=sum_dtype))
        .otherwise(pl.col("values/sum"))
        .alias("values/sum"),
        pl.when(pl.col("code") == polluted_code)
        .then(pl.lit(1, dtype=nocc_dtype))
        .otherwise(pl.col("values/n_occurrences"))
        .alias("values/n_occurrences"),
    )
    polluted.write_parquet(codes_fp)
    return polluted_code


def _stage_cohort_with_polluted_bin(
    src_cohort: Path, dst: Path, code_filter_expr: pl.Expr
) -> tuple[Path, str]:
    """Copy a tensorized cohort dir to ``dst`` and pollute one bin in its ``codes.parquet``.

    Returns ``(dst_cohort, polluted_code)`` so the caller can point the CLI at ``dst_cohort``
    and grep for ``polluted_code`` in the failure output.
    """
    shutil.copytree(src_cohort, dst)
    polluted_code = _pollute_bin(dst / "metadata" / "codes.parquet", code_filter_expr)
    return dst, polluted_code


def _generate_trajectories_cli(
    *, cohort_dir: Path, task_dir: Path, model_dir: Path, output_dir: Path
) -> subprocess.CompletedProcess:
    """Run ``MEICAR_generate_trajectories`` against the supplied (mutated) cohort.

    Mirrors the demo-config invocation used by ``generated_trajectories`` in conftest so the
    only meaningful axis under test is the polluted-bin behavior, not config drift.
    """
    return subprocess.run(
        [
            "MEICAR_generate_trajectories",
            "--config-name=_demo_generate_trajectories",
            f"output_dir={output_dir}",
            f"model_initialization_dir={model_dir}",
            f"datamodule.config.tensorized_cohort_dir={cohort_dir}",
            f"datamodule.config.task_labels_dir={task_dir}",
            "datamodule.batch_size=2",
            "trainer=demo",
        ],
        capture_output=True,
        check=False,
    )


def test_generate_trajectories_cli_refuses_polluted_timeline_delta_bin(
    pretrained_model: Path,
    preprocessed_dataset_with_task: tuple[Path, Path, str],
    tmp_path: Path,
):
    """A polluted ``TIMELINE//DELTA`` bin makes the CLI fail before any predict pass runs.

    This is the failure mode the validator was added for: a polluted bin would have crashed
    finalize partway through hours of generation; the validator's job is to catch it at the
    CLI startup so the user sees the bin-name error in the first second instead.
    """
    src_cohort, task_root_dir, task_name = preprocessed_dataset_with_task
    cohort_dir, polluted_code = _stage_cohort_with_polluted_bin(
        src_cohort,
        tmp_path / "polluted_delta_cohort",
        pl.col("code").str.starts_with(TIMELINE_DELTA_TOKEN),
    )

    result = _generate_trajectories_cli(
        cohort_dir=cohort_dir,
        task_dir=task_root_dir / task_name,
        model_dir=pretrained_model,
        output_dir=tmp_path / "out",
    )

    assert result.returncode != 0, (
        "Generation CLI should refuse a vocab with an overflowing TIMELINE//DELTA bin. "
        f"Stdout:\n{result.stdout.decode()}\nStderr:\n{result.stderr.decode()}"
    )
    stderr = result.stderr.decode()
    assert "TIMELINE//DELTA" in stderr, (
        f"Expected the failure to surface the validator's ValueError naming the offending "
        f"bin family; got stderr:\n{stderr}"
    )
    assert polluted_code in stderr, (
        f"Expected the polluted bin's code {polluted_code!r} to appear in the failure "
        f"message so the user can chase it upstream; got stderr:\n{stderr}"
    )


def test_generate_trajectories_cli_succeeds_with_polluted_non_delta_bin(
    pretrained_model: Path,
    preprocessed_dataset_with_task: tuple[Path, Path, str],
    tmp_path: Path,
):
    """A polluted *non-delta* bin (e.g. HR with a huge ``value_mean``) does not gate generation.

    The validator only checks ``TIMELINE//DELTA`` bins — those are the ones that get cast to
    Int64 microseconds. A non-delta bin's ``value_mean`` is the mean numeric_value of that
    code (e.g. mean heart rate within an HR bin) and never participates in time encoding;
    the CLI must still run to completion even if such a bin has an absurd value.
    """
    src_cohort, task_root_dir, task_name = preprocessed_dataset_with_task
    # Pick any non-delta bin that actually carries values (n_occurrences > 0). HR bins are
    # present in the demo vocab; filter by the "HR//" prefix to avoid touching the
    # null-value-mean codes (categorical events) where the pollution wouldn't even take.
    cohort_dir, _polluted_code = _stage_cohort_with_polluted_bin(
        src_cohort,
        tmp_path / "polluted_hr_cohort",
        pl.col("code").str.starts_with("HR//value_") & (pl.col("values/n_occurrences") > 0),
    )

    result = _generate_trajectories_cli(
        cohort_dir=cohort_dir,
        task_dir=task_root_dir / task_name,
        model_dir=pretrained_model,
        output_dir=tmp_path / "out",
    )

    assert result.returncode == 0, (
        "Generation CLI should run to completion when a non-delta bin is polluted — only "
        "TIMELINE//DELTA bins gate generation. "
        f"Stdout:\n{result.stdout.decode()}\nStderr:\n{result.stderr.decode()}"
    )
