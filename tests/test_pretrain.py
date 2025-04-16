from pathlib import Path


def test_pretrain_runs(pretrained_model: Path):
    out_files = list(pretrained_model.rglob("*.*"))
    assert len(out_files) > 0
