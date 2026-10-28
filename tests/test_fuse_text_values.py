"""The preprocessing pipeline folds categorical text values into codes (the ``fuse_text_values`` stage)."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    from pathlib import Path

CATEGORY_COMMON = "SR (Sinus Rhythm)"
CATEGORY_RARE = "AF (Atrial Fibrillation)"
CATEGORY_DIGIT_LEADING = "30 Degrees"
CATEGORY_LOWERCASE = "3 rails up"
UNLETTERED_TEXT = "25/50/75/100"
BARE_CODES = {"GENDER//F", "HOSPITAL_ADMISSION//EW EMER.", "HOSPITAL_DISCHARGE//HOME", "MEDS_DEATH"}


def _write_tiny_meds(root: Path) -> Path:
    """Write a six-subject MEDS cohort whose chartevents-style codes carry their category in ``text_value``.

    Subjects 1-3 train, 4 tunes, 5-6 are held out. Every subject has the common rhythm category, the digit-
    leading head-of-bed category, a lowercase-only side-rails category, an unlettered text value, a number
    stored as text, a lab row with a number and a text flag, and code-only rows (a static GENDER row, an
    admission and a discharge); subject 2 alone has the rare rhythm category and a MEDS_DEATH row.
    """

    def rows(sid: int, rare: bool) -> list[tuple]:
        t = [f"2020-01-01T{h:02d}:00:00" for h in range(12)]
        out = [
            (sid, None, "GENDER//F", None, None),
            (sid, t[0], "HOSPITAL_ADMISSION//EW EMER.", None, None),
            (sid, t[1], "LAB//RHYTHM//UNK", None, CATEGORY_COMMON),
            (sid, t[2], "LAB//RHYTHM//UNK", None, CATEGORY_RARE if rare else CATEGORY_COMMON),
            (sid, t[3], "LAB//50912//mg/dL", 1.2 + sid / 10, None),
            (sid, t[4], "Blood Pressure", None, "120/80"),
            (sid, t[5], "LAB//HOB//UNK", None, CATEGORY_DIGIT_LEADING),
            (sid, t[5], "LAB//RAILS//UNK", None, CATEGORY_LOWERCASE),
            (sid, t[6], "LAB//50912//mg/dL", 0.9 + sid / 10, "flag"),
            (sid, t[7], "LAB//PCT//UNK", None, UNLETTERED_TEXT),
            (sid, t[8], "HOSPITAL_DISCHARGE//HOME", None, None),
        ]
        if rare:
            out.append((sid, t[9], "MEDS_DEATH", None, None))
        return out

    schema = {
        "subject_id": pl.Int64,
        "time": pl.Utf8,
        "code": pl.Utf8,
        "numeric_value": pl.Float32,
        "text_value": pl.Utf8,
    }
    splits = {"train": [1, 2, 3], "tuning": [4], "held_out": [5, 6]}
    for split, sids in splits.items():
        (root / "data" / split).mkdir(parents=True, exist_ok=True)
        data = [r for sid in sids for r in rows(sid, rare=(sid == 2))]
        pl.DataFrame(data, schema=schema, orient="row").with_columns(
            pl.col("time").str.to_datetime(time_unit="us")
        ).write_parquet(root / "data" / split / "0.parquet")

    (root / "metadata").mkdir(parents=True, exist_ok=True)
    codes = sorted({r[2] for r in rows(1, rare=True)})
    pl.DataFrame(
        {"code": codes, "description": codes, "parent_codes": [[] for _ in codes]},
        schema={"code": pl.Utf8, "description": pl.Utf8, "parent_codes": pl.List(pl.Utf8)},
    ).write_parquet(root / "metadata" / "codes.parquet")
    pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "split": ["train", "train", "train", "tuning", "held_out", "held_out"],
        },
        schema={"subject_id": pl.Int64, "split": pl.Utf8},
    ).write_parquet(root / "metadata" / "subject_splits.parquet")
    (root / "metadata" / "dataset.json").write_text(
        json.dumps(
            {
                "dataset_name": "tiny_text_values",
                "dataset_version": "0",
                "etl_name": "test",
                "etl_version": "0",
                "meds_version": "0.4.1",
                "created_at": "2026-10-27T00:00:00",
            }
        )
    )
    return root


def _run(meds: Path, root: Path, **overrides: object) -> tuple[Path, Path]:
    intermediate_dir, output_dir = root / "intermediate", root / "output"
    cmd = [
        "MEICAR_process_data",
        f"input_dir={meds!s}",
        f"intermediate_dir={intermediate_dir!s}",
        f"output_dir={output_dir!s}",
        "do_demo=True",
        *(f"{k}={v}" for k, v in overrides.items()),
    ]
    out = subprocess.run(cmd, capture_output=True, check=False)
    assert out.returncode == 0, f"stdout:\n{out.stdout.decode()}\nstderr:\n{out.stderr.decode()}"
    return intermediate_dir, output_dir


def _rows_per_split(root: Path) -> dict[str, int]:
    return {
        split.name: sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in split.glob("*.parquet"))
        for split in sorted(root.glob("*"))
        if split.is_dir() and not split.name.startswith(".")
    }


def _check_fused_vocabulary(codes: set[str]) -> None:
    fused_common = f"LAB//RHYTHM//UNK//{CATEGORY_COMMON}"
    assert fused_common in codes, f"the common category was not fused into the code: {sorted(codes)}"
    assert "LAB//RHYTHM//UNK" not in codes, "the bare item must not survive as its own token once fused"
    assert f"LAB//HOB//UNK//{CATEGORY_DIGIT_LEADING}" in codes, "a digit-leading category is still a category"
    assert f"LAB//RAILS//UNK//{CATEGORY_LOWERCASE}" in codes, "a lowercase-only category is still a category"
    # Numbers stored as text and text without a letter are not categories: the code is left alone.
    assert "Blood Pressure" in codes
    assert not any(c.startswith("Blood Pressure//") for c in codes)
    assert "LAB//PCT//UNK" in codes
    assert not any(c.startswith("LAB//PCT//UNK//") for c in codes)
    # A row that carries a numeric value keeps the numeric path even when a text flag is present.
    assert any(c.startswith("LAB//50912//mg/dL//value_") for c in codes)
    assert not any(c.endswith("//flag") for c in codes)
    # A rare category is a rare code: the demo threshold (2 subjects) drops it like any other.
    assert f"LAB//RHYTHM//UNK//{CATEGORY_RARE}" not in codes
    # Rows with neither text nor number pass through untouched.
    assert codes >= BARE_CODES, f"code-only rows were lost: {sorted(BARE_CODES - codes)}"


@pytest.mark.parametrize("do_reshard", [False, True])
def test_text_values_are_fused_into_codes(tmp_path: Path, do_reshard: bool) -> None:
    meds = _write_tiny_meds(tmp_path / "meds")
    intermediate_dir, output_dir = _run(meds, tmp_path, do_reshard=do_reshard)

    codes = set(pl.read_parquet(output_dir / "metadata" / "codes.parquet")["code"].to_list())
    _check_fused_vocabulary(codes)
    assert "fuse_text_values" in (intermediate_dir / ".logs" / "pipeline_config.yaml").read_text()

    # The stage keeps every row: it rewrites columns in place and never filters or concatenates.
    stage_root = intermediate_dir / "fuse_text_values"
    assert _rows_per_split(stage_root) == _rows_per_split(meds / "data")

    # Fused rows lose their ``text_value``; number-plus-flag rows and unlettered text keep theirs.
    stage_train = pl.concat([pl.read_parquet(f) for f in (stage_root / "train").glob("*.parquet")])
    fused = stage_train.filter(pl.col("code").str.starts_with("LAB//RHYTHM//UNK//"))
    assert fused.height > 0 and fused["text_value"].null_count() == fused.height
    assert stage_train.filter(pl.col("text_value") == "flag").height == 3
    assert stage_train.filter(pl.col("text_value") == UNLETTERED_TEXT).height == 3


def test_fusion_can_be_switched_off(tmp_path: Path) -> None:
    meds = _write_tiny_meds(tmp_path / "meds")
    intermediate_dir, output_dir = _run(meds, tmp_path, do_fuse_text_values=False)
    assert "fuse_text_values" not in (intermediate_dir / ".logs" / "pipeline_config.yaml").read_text()
    codes = set(pl.read_parquet(output_dir / "metadata" / "codes.parquet")["code"].to_list())
    assert "LAB//RHYTHM//UNK" in codes
    assert not any(c.startswith("LAB//RHYTHM//UNK//") for c in codes)


def test_fusion_stage_is_skipped_without_text_value(preprocessed_dataset: Path) -> None:
    """The shared static cohort has no ``text_value`` column: the stage is dropped and the run succeeds."""
    config_fp = preprocessed_dataset.parent / "intermediate" / ".logs" / "pipeline_config.yaml"
    assert config_fp.exists()
    assert "fuse_text_values" not in config_fp.read_text()
