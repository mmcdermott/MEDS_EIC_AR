"""End-to-end grammar test driving ``MEICAR_generate_trajectories`` through the SGLang backend.

Mirrors the correctness gate from
``test_grammar_cli_greedy_output_is_fully_grammar_valid`` in
``tests/grammar/test_cli.py``, but runs
the full CLI pipeline with ``backend=sglang`` instead of the default ``hf``. If SGLang is wired
correctly — in-memory Llama checkpoint → ``export_lightning_to_hf_dir`` → ``SGLangBackend`` →
rolling-generation loop → parquet output — a greedy run should produce output that clears the
same 100%-strict grammar validity bar as the HF backend path does.

Not a byte-parity check: SGLang and HF's greedy paths differ in matmul kernel order and
softmax numerical-stability tricks, so exact token agreement is flaky (see issue #88 gotcha
§7 for the full rationale). We compare *properties* — every generated grammar token is a
valid FSM continuation from the prompt's end-state — which is robust to floating-point drift.

Gating
------

This test is skipped unless **both** are true:

1. ``sglang`` is importable (gated via module-level ``pytest.importorskip``).
2. A CUDA device is available at runtime (SGLang requires GPU — there is no CPU-only path).

The repo has no GPU runner, so CI collects-and-skips this test. A developer with ``sglang``
installed and a CUDA GPU can run it locally with::

    uv sync --extra sglang --prerelease=allow
    uv run pytest tests/grammar/test_cli_sglang.py

Expected runtime: a few minutes, dominated by engine startup (weight load + piecewise CUDA
graph capture — the fixture ``sglang.yaml`` sets ``disable_cuda_graph=true`` to skip capture,
but weight load still pays a cold-start cost on every ``MEICAR_generate_trajectories``
invocation).

Why this particular test (vs. a sampling-distribution check)
------------------------------------------------------------

The sampling-distribution test ``test_grammar_cli_model_shows_grammar_signal`` asserts
statistical signal, not a hard property — under sampling both HF and SGLang should clear the
per-sample 50% threshold on most held-out samples, but the specific token streams differ
between the two backends. Locking that in as a cross-backend regression gate adds a flakiness
axis without buying correctness signal.

The greedy 100%-validity bar is binary — either every token is a valid FSM step from the
prompt, or it isn't. SGLang's greedy path (``temperature=0.0``, the translation we apply
when ``do_sample=False``) producing *any* invalid transition means either the engine is
producing different tokens than HF greedy would (floating-point drift past the argmax tie
threshold — rare at our tiny hidden size) or the pipeline wiring is broken. Either way we
want the test to surface it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from meds import held_out_split

pytest.importorskip("sglang")

# Local helpers from the HF-path grammar test. Importing rather than duplicating keeps the
# strict-validity check byte-identical between the two backends — if `test_cli.py` evolves
# the scoring logic, this test picks it up automatically.
from tests.grammar._meds import grammar_tokens_from_output_df, prompt_grammar_tokens_by_subject
from tests.grammar.conftest import _run_grammar_generate
from tests.grammar.test_cli import _load_trajectories_by_split, _walk_generated_strictly

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SGLang requires a CUDA GPU; skipping on CPU-only runners.",
)


@pytest.fixture(scope="module")
def sglang_grammar_generated_greedy(
    grammar_pretrained: Path,
    grammar_preprocessed: Path,
    grammar_raw_meds: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
    grammar_n_trajectories: int,
    grammar_generate_batch_size: int,
    grammar_rolling_max_new_tokens: int,
) -> Path:
    """Drive ``MEICAR_generate_trajectories`` end-to-end through ``backend=sglang``, greedy path."""
    _, task_dir = grammar_raw_meds
    output_dir = tmp_path_factory.mktemp("sglang_grammar_generated_greedy")
    _run_grammar_generate(
        output_dir,
        grammar_pretrained,
        grammar_preprocessed,
        task_dir,
        do_sample=False,
        n_trajectories=grammar_n_trajectories,
        batch_size=grammar_generate_batch_size,
        rolling_max_new_tokens=grammar_rolling_max_new_tokens,
        # Uses the demo-scale SGLang config (``max_running_requests=8``,
        # ``disable_cuda_graph=true``) rather than the production ``sglang`` defaults.
        # Fast cold-start is a lot more useful than peak decode throughput for a single
        # grammar-validation test run that exits after one pass.
        backend="sglang_demo",
    )
    return output_dir


def test_sglang_greedy_output_is_fully_grammar_valid(
    sglang_grammar_generated_greedy: Path,
    grammar_raw_meds: tuple[Path, Path],
):
    """Every generated grammar token under greedy SGLang must be a valid FSM continuation.

    Exact analogue of the HF-backend test in ``test_cli.py``. Any failure here means either
    the SGLang engine emitted a different greedy token stream than HF would (rare — would
    require floating-point drift flipping an argmax tie) or, more likely, something in the
    pipeline wiring (HF export → engine init → new-token decoding → rolling-loop accumulation
    → parquet output) is off. The error message dumps the first offending sample.
    """
    raw_meds_dir, task_labels_dir = grammar_raw_meds
    by_split = _load_trajectories_by_split(sglang_grammar_generated_greedy)

    prompts_by_subject = prompt_grammar_tokens_by_subject(raw_meds_dir, task_labels_dir, held_out_split)

    total_samples = 0
    failures: list[str] = []
    for t, df in by_split[held_out_split].items():
        tokens_by_subject = grammar_tokens_from_output_df(df)
        for subject_id, tokens in tokens_by_subject.items():
            if not tokens:
                continue
            total_samples += 1
            prompt_tokens = prompts_by_subject[subject_id]
            valid, total = _walk_generated_strictly(prompt_tokens, tokens)
            if valid != total:
                failures.append(
                    f"traj={t} subject={subject_id}: {valid}/{total} valid "
                    f"(first invalid at generated position {valid}): "
                    f"prompt_tokens={prompt_tokens}, generated={tokens}"
                )

    assert total_samples > 0, (
        "No held-out greedy SGLang samples had any generated grammar tokens — the model "
        "either produced no output or produced only non-grammar codes. Suspect the SGLang "
        "export/engine/new-token-decoding path."
    )
    assert not failures, (
        f"Greedy SGLang output was not 100% grammar-valid on {len(failures)} / {total_samples} "
        f"held-out samples:\n" + "\n".join(failures)
    )
