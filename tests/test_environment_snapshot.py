"""Invariant tests for ``MEDS_EIC_AR.utils.save_environment_snapshot``.

The docstring doctest already covers the basic shape (header marker, python/platform
lines, ``name==version`` dep lines in case-insensitive sort order). This file adds the
invariants that are awkward to express in a doctest: every-line shape validation and the
missing-parent-dir handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from MEDS_EIC_AR.utils import save_environment_snapshot

if TYPE_CHECKING:
    from pathlib import Path


def _snapshot(tmp_path: Path) -> list[str]:
    """Write a snapshot under ``tmp_path`` and return its lines."""
    fp = tmp_path / "environment.txt"
    assert save_environment_snapshot(fp) is True
    return fp.read_text().splitlines()


def test_every_package_line_looks_like_pip_freeze(tmp_path: Path):
    """Every non-header line is ``name==version`` — same shape as ``pip freeze`` output.

    Matters because anyone reading the file and trying to recreate the environment will
    feed it to ``pip install -r`` or equivalent, which expects this format. The doctest
    only anchors on three named deps; this asserts the shape across *every* line.
    """
    lines = _snapshot(tmp_path)
    pkg_lines = [line for line in lines if not line.startswith("#")]
    assert pkg_lines, "snapshot should contain at least the meds-eic-ar self-entry"
    for line in pkg_lines:
        assert "==" in line, f"package line {line!r} is not in ``name==version`` format"


def test_missing_parent_directory_is_handled(tmp_path: Path):
    """Writing to a not-yet-existing parent directory should succeed; the helper creates it."""
    fp = tmp_path / "missing_subdir" / "nested" / "env.txt"
    assert save_environment_snapshot(fp) is True
    assert fp.is_file()
