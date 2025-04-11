import subprocess
import tempfile
from pathlib import Path


def test_process_data_runs(simple_static_MEDS: Path):
    with tempfile.TemporaryDirectory() as test_root:
        test_root = Path(test_root)

        input_dir = simple_static_MEDS
        interemediate_dir = test_root / "intermediate"
        output_dir = test_root / "output"

        cmd = [
            "MEICAR_process_data",
            f"input_dir={input_dir!s}",
            f"intermediate_dir={interemediate_dir!s}",
            f"output_dir={output_dir!s}",
        ]

        out = subprocess.run(cmd, capture_output=True, check=False)

        err_lines = [
            "Command failed:",
            "Stdout:",
            out.stdout.decode(),
            "Stderr:",
            out.stderr.decode(),
        ]

        assert out.returncode == 0, "\n".join([*err_lines, f"Return code: {out.returncode}"])
