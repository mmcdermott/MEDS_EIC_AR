"""Regression tests for the issue #154 vocabulary validator.

A polluted ``TIMELINE//DELTA`` bin (a single outlier numeric_value averaged into the bin
upstream) makes ``value_mean * seconds_per_unit * 1e6`` overflow Int64 microseconds, so
``format_trajectories`` cannot encode the per-token delta and the resulting trajectory
would be uninterpretable even if the cast were saturating. The fix is to refuse to start
generation at all when the vocab has any such bin —
:func:`validate_timeline_delta_bins_in_int64_range` runs at CLI startup, before any model
load or predict pass, and raises ``ValueError`` naming the offending bins so the user can
go fix the upstream bin reduction (``fit_quantile_binning`` / ``bin_numeric_values``).

These tests target the validator directly with synthetic ``codes.parquet`` files instead
of standing up a full ``MEDSPytorchDataset`` — the validator's contract is a path in,
nothing or an exception out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from MEDS_EIC_AR.generation.finalize import (
    TIMELINE_DELTA_TOKEN,
    validate_timeline_delta_bins_in_int64_range,
)

if TYPE_CHECKING:
    from pathlib import Path

_SECONDS_PER_YEAR = 31_556_926.08
_INT64_MAX = (1 << 63) - 1

# Pick a value_mean (in years) that's strictly larger than the largest one that would still
# fit in Int64 microseconds — derived from Int64 max so the test stays correct if the unit
# or the cast width ever changes, instead of pinning the issue's reported "1.9e19" magic
# number. ``* 2`` gives comfortable headroom past the boundary.
_OVERFLOWING_VALUE_MEAN = 2.0 * _INT64_MAX / (_SECONDS_PER_YEAR * 1_000_000)
_SAFE_VALUE_MEAN = 0.000003  # ~95 ms — the "normal" delta scale in the demo vocab.


def _write_codes_parquet(fp: Path, rows: list[dict]) -> None:
    """Write a minimal MEDS-shaped ``codes.parquet`` with just the columns the validator reads.

    The four-column subset (``code``, ``code/vocab_index``, ``values/n_occurrences``,
    ``values/sum``) matches what ``_load_code_metadata`` selects; ``value_mean`` is then
    computed inside the validator as ``values/sum / values/n_occurrences``.
    """
    pl.DataFrame(
        rows,
        schema={
            "code": pl.Utf8,
            "code/vocab_index": pl.Int64,
            "values/n_occurrences": pl.Int64,
            "values/sum": pl.Float64,
        },
    ).write_parquet(fp)


def _row(code: str, idx: int, value_mean: float | None) -> dict:
    """Encode a vocab row in (sum, n_occurrences) form so value_mean computes back to ``value_mean``."""
    if value_mean is None:
        return {"code": code, "code/vocab_index": idx, "values/n_occurrences": 0, "values/sum": 0.0}
    return {"code": code, "code/vocab_index": idx, "values/n_occurrences": 1, "values/sum": value_mean}


def test_validator_passes_on_clean_vocab(tmp_path: Path):
    """A vocabulary whose delta bins all fit in Int64 microseconds is accepted silently."""
    fp = tmp_path / "codes.parquet"
    _write_codes_parquet(
        fp,
        [
            _row(f"{TIMELINE_DELTA_TOKEN}//years//value_[0,1)", 1, _SAFE_VALUE_MEAN),
            _row(f"{TIMELINE_DELTA_TOKEN}//years//value_[1,2)", 2, _SAFE_VALUE_MEAN * 10),
            _row("DISCHARGE", 3, None),
        ],
    )

    validate_timeline_delta_bins_in_int64_range(fp)


def test_validator_rejects_polluted_vocab(tmp_path: Path):
    """A vocabulary with any TIMELINE//DELTA bin past Int64 microseconds is refused, naming the bin."""
    fp = tmp_path / "codes.parquet"
    polluted_code = f"{TIMELINE_DELTA_TOKEN}//years//value_[POLLUTED)"
    _write_codes_parquet(
        fp,
        [
            _row(f"{TIMELINE_DELTA_TOKEN}//years//value_[0,1)", 1, _SAFE_VALUE_MEAN),
            _row(polluted_code, 2, _OVERFLOWING_VALUE_MEAN),
            _row("DISCHARGE", 3, None),
        ],
    )

    with pytest.raises(ValueError, match="TIMELINE//DELTA") as exc_info:
        validate_timeline_delta_bins_in_int64_range(fp)
    msg = str(exc_info.value)
    assert polluted_code in msg, f"validator should name the offending bin: {msg!r}"
    assert "Int64" in msg, f"validator should explain the failure mode: {msg!r}"


def test_validator_ignores_overflow_in_non_delta_bins(tmp_path: Path):
    """Only TIMELINE//DELTA bins gate generation; other codes' value_mean is irrelevant here.

    A non-delta code's ``value_mean`` is the mean numeric_value of observations of that code
    (e.g. the mean heart rate within a HR bin) — it never participates in the time-encoding
    cast. Pin this so a future "validate every bin" change doesn't sneak in and start
    rejecting datasets with legitimately large numeric measurements.
    """
    fp = tmp_path / "codes.parquet"
    _write_codes_parquet(
        fp,
        [
            _row(f"{TIMELINE_DELTA_TOKEN}//years//value_[0,1)", 1, _SAFE_VALUE_MEAN),
            _row("BIG_NUMBER//value_[POLLUTED)", 2, _OVERFLOWING_VALUE_MEAN),
        ],
    )

    validate_timeline_delta_bins_in_int64_range(fp)
