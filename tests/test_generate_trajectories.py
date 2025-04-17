from pathlib import Path


def test_generate_trajectories_runs(generated_trajectories: Path):
    out_files = list(generated_trajectories.rglob("*.*"))
    assert len(out_files) > 0
