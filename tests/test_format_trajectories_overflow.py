"""Regression tests for format_trajectories int64 overflow on extreme TIMELINE//DELTA bin means.

Issue #154: when any ``TIMELINE//DELTA`` bin's ``value_mean`` is large enough that
``value_mean * seconds_per_unit * 1e6`` exceeds the Int64 range, the strict cast in
:func:`MEDS_EIC_AR.generation.finalize.format_trajectories` raises ``InvalidOperationError``
and tears down the entire generation run — even when only one row in one trajectory
hits the polluted bin. The fix replaces the strict cast with a saturating one (clip to
the Int64-safe f64 bounds, then cast), so the run continues and the offending row gets
an out-of-range-but-finite timestamp instead of crashing the job.

These tests don't go through the heavyweight ``MEDSPytorchDataset`` — we patch
:func:`_get_code_metadata` and stand up a synthetic ``schema_df`` so the test isolates
the cast behavior in ``format_trajectories``.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset

from MEDS_EIC_AR.generation import finalize
from MEDS_EIC_AR.generation.finalize import TIMELINE_DELTA_TOKEN, format_trajectories

# value_mean (in years) large enough that value_mean * 3.1557e7 * 1e6 overflows Int64.
# Mirrors the polluted bin reported in issue #154 ("1.8985e19 years").
_POLLUTED_VALUE_MEAN = 1.8985e19

# Saturating cast lower bound: largest f64 <= INT64_MAX (= 2**63 - 1024).
_INT64_F64_MAX = 9223372036854774784


def _build_code_metadata() -> pl.DataFrame:
    """Synthetic vocab: one normal delta bin, one polluted delta bin, one non-delta code."""
    return pl.DataFrame(
        {
            "code_idx": [1, 2, 3],
            "code": [
                f"{TIMELINE_DELTA_TOKEN}//years//value_[0,1)",
                f"{TIMELINE_DELTA_TOKEN}//years//value_[POLLUTED)",
                "DISCHARGE",
            ],
            "value_mean": [0.000003, _POLLUTED_VALUE_MEAN, None],
        },
        schema={"code_idx": pl.Int64, "code": pl.Utf8, "value_mean": pl.Float32},
    )


def _build_base_dataset() -> SimpleNamespace:
    """Minimal stand-in for ``MEDSPytorchDataset``: only needs ``schema_df``."""
    schema_df = pl.DataFrame(
        {
            DataSchema.subject_id_name: [42],
            LabelSchema.prediction_time_name: [datetime(2024, 1, 1, 12, 0, 0)],  # noqa: DTZ001
            MEDSPytorchDataset.LAST_TIME: [datetime(2024, 1, 1, 11, 0, 0)],  # noqa: DTZ001
        },
        schema={
            DataSchema.subject_id_name: pl.Int64,
            LabelSchema.prediction_time_name: pl.Datetime("us"),
            MEDSPytorchDataset.LAST_TIME: pl.Datetime("us"),
        },
    )
    return SimpleNamespace(schema_df=schema_df)


def _merged_with_polluted_token() -> pl.DataFrame:
    # One row with the polluted-bin token (idx=2) followed by a non-delta token (idx=3).
    return pl.DataFrame(
        {"dataset_row_idx": [0], "tokens": [[2, 3]]},
        schema={"dataset_row_idx": pl.Int64, "tokens": pl.List(pl.Int64)},
    )


def _merged_normal_only() -> pl.DataFrame:
    return pl.DataFrame(
        {"dataset_row_idx": [0], "tokens": [[1, 3]]},
        schema={"dataset_row_idx": pl.Int64, "tokens": pl.List(pl.Int64)},
    )


def test_format_trajectories_saturates_on_polluted_bin_instead_of_crashing():
    """An extreme ``value_mean`` no longer crashes the run; the row gets a saturated time."""
    base_dataset = _build_base_dataset()
    code_metadata = _build_code_metadata()
    merged = _merged_with_polluted_token()

    with patch.object(finalize, "_get_code_metadata", return_value=code_metadata):
        out = format_trajectories(base_dataset, merged)

    # Two output rows: the polluted delta token and the trailing DISCHARGE.
    assert out.height == 2
    # Polluted-bin numeric_value passes through unchanged (the value is what diagnoses upstream
    # bin pollution; we don't silently scrub it on the trajectory side).
    polluted_row = out.filter(pl.col(DataSchema.code_name).str.contains("POLLUTED"))
    assert polluted_row.height == 1
    assert polluted_row[DataSchema.numeric_value_name][0] == pytest.approx(_POLLUTED_VALUE_MEAN, rel=1e-3)

    # The DISCHARGE row's timestamp is whatever ``LAST_TIME + saturated delta`` gives —
    # nonsensical clinically, but finite. The contract is "no crash" — we don't pin the
    # exact wall-clock timestamp here because polars datetime arithmetic on near-i64-max
    # microsecond deltas is implementation-defined (it may wrap or clamp). What matters is
    # that the run produced output for the trajectory.
    discharge_row = out.filter(pl.col(DataSchema.code_name) == "DISCHARGE")
    assert discharge_row.height == 1


def test_format_trajectories_normal_path_unchanged_by_saturation():
    """The clip is a no-op for non-polluted bins: timestamps match pre-fix behavior."""
    base_dataset = _build_base_dataset()
    code_metadata = _build_code_metadata()
    merged = _merged_normal_only()

    with patch.object(finalize, "_get_code_metadata", return_value=code_metadata):
        out = format_trajectories(base_dataset, merged)

    # Token 1 is a normal delta bin (value_mean = 3e-6 years ≈ 94.67 ms); token 3 is DISCHARGE.
    # The DISCHARGE row lands at ``LAST_TIME + value_mean(f32) * 31556926 s/yr * 1e6 us/s``
    # cast to Int64 — i.e., the same arithmetic the unfixed code produces, just without the
    # strict-cast crash on overflow. Pinning the exact microsecond confirms the saturating
    # cast doesn't perturb the non-overflow path.
    discharge_row = out.filter(pl.col(DataSchema.code_name) == "DISCHARGE")
    assert discharge_row.height == 1
    assert discharge_row[DataSchema.time_name][0] == datetime(2024, 1, 1, 11, 1, 34, 670784)  # noqa: DTZ001


def test_format_trajectories_logs_warning_on_polluted_bin(caplog):
    """Polluted bins surface as a warning so users can identify the bad bin in the vocab."""
    base_dataset = _build_base_dataset()
    code_metadata = _build_code_metadata()
    merged = _merged_with_polluted_token()

    with (
        caplog.at_level("WARNING", logger=finalize.__name__),
        patch.object(finalize, "_get_code_metadata", return_value=code_metadata),
    ):
        format_trajectories(base_dataset, merged)

    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("POLLUTED" in m for m in msgs), f"Expected a warning naming the polluted bin, got: {msgs!r}"
    assert any("saturat" in m.lower() or "clip" in m.lower() or "overflow" in m.lower() for m in msgs)
