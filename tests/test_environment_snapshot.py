"""Invariant tests for ``MEDS_EIC_AR.utils.save_environment_snapshot``.

These were originally packed into a single doctest inside the helper's docstring, which was flagged on PR
review as hard to read. Splitting into individual pytest cases lets each invariant live under a named
assertion with its own short explanation, which is also what fails loudly if one of them regresses in
isolation.
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


def test_header_line_is_stable(tmp_path: Path):
    """The first line of the snapshot is a fixed marker so tooling can recognize the file."""
    lines = _snapshot(tmp_path)
    assert lines[0] == "# MEDS_EIC_AR run environment snapshot"


def test_second_line_reports_python_version(tmp_path: Path):
    """The second line starts with ``# python:`` so a human scanning the file sees the interpreter upfront."""
    lines = _snapshot(tmp_path)
    assert lines[1].startswith("# python: ")


def test_third_line_reports_platform(tmp_path: Path):
    """The third line starts with ``# platform:`` so a human sees the host OS/arch upfront."""
    lines = _snapshot(tmp_path)
    assert lines[2].startswith("# platform: ")


def test_package_lines_look_like_pip_freeze(tmp_path: Path):
    """Every non-header line is ``name==version`` — same shape as ``pip freeze`` output.

    Matters because anyone reading the file and trying to recreate the environment will
    feed it to ``pip install -r`` or equivalent, which expects this format.
    """
    lines = _snapshot(tmp_path)
    pkg_lines = [line for line in lines if not line.startswith("#")]
    assert pkg_lines, "snapshot should contain at least the meds-eic-ar self-entry"
    for line in pkg_lines:
        assert "==" in line, f"package line {line!r} is not in ``name==version`` format"


def test_package_lines_are_case_insensitive_sorted(tmp_path: Path):
    """Deterministic ordering across macOS/Linux — the discovery order from
    ``importlib.metadata.distributions()`` returns mixed case on macOS but canonical lower-case on Linux.

    Case-insensitive sort stabilizes the file across both.
    """
    lines = _snapshot(tmp_path)
    pkg_lines = [line for line in lines if not line.startswith("#")]
    assert pkg_lines == sorted(pkg_lines, key=str.lower)


def test_missing_parent_directory_is_handled(tmp_path: Path):
    """Writing to a not-yet-existing parent directory should succeed; the helper creates it."""
    fp = tmp_path / "missing_subdir" / "nested" / "env.txt"
    assert save_environment_snapshot(fp) is True
    assert fp.is_file()


def test_self_entry_present(tmp_path: Path):
    """The snapshot must include ``MEDS_EIC_AR`` itself — the whole point is to pin the repo's own version
    alongside its deps."""
    lines = _snapshot(tmp_path)
    # ``importlib.metadata`` normalizes the distribution name per PEP 503 conventions;
    # accept either the canonical ``MEDS_EIC_AR`` or the underscore/hyphen variants HF
    # and others sometimes produce.
    pkg_names = {line.split("==")[0].lower().replace("-", "_") for line in lines if "==" in line}
    assert "meds_eic_ar" in pkg_names, f"meds-eic-ar should be present; got {sorted(pkg_names)[:10]}..."
