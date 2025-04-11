"""Test set-up and fixtures code."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    simple_static_MEDS: Path,
    simple_static_MEDS_dataset_with_task: Path,
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "simple_static_MEDS": simple_static_MEDS,
            "simple_static_MEDS_dataset_with_task": simple_static_MEDS_dataset_with_task,
        }
    )
