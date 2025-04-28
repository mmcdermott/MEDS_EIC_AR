import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from meds import held_out_split, train_split, tuning_split
from meds_testing_helpers.dataset import MEDSDataset


def generate_transition_probs_matrix(transition_spec: str) -> np.ndarray:
    """Takes a spec of the form 'A -> B -> A; A -> C; C -> D -> E -> C' and generates transition probs for it.

    Args:
        transition_spec: A string representing the transition specification. All possible transitions are
            divided equally between options.

    Returns:
        A numpy array representing the transition probabilities.

    Examples:
        >>> generate_transition_probs_matrix("A -> B -> A")
        array([[0.0, 1.0],
               [1.0, 0.0]])
        >>> generate_transition_probs_matrix("A -> B -> C -> A; A -> D; D -> D -> B")
        array([[0.0, 0.5, 0.0, 0.5],
               [0.0, 0.0, 1.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.5, 0.0, 0.5]])
    """

    # Split the transition specification into individual transitions
    transitions = transition_spec.split(";")

    # Create a mapping of states to indices
    states = set()
    for transition in transitions:
        parts = transition.split("->")
        for part in parts:
            states.add(part.strip())

    states = sorted(states)
    state_to_index = {state: index for index, state in enumerate(states)}

    # Initialize the transition probability matrix
    num_states = len(states)
    transition_probs = np.zeros((num_states, num_states))

    # Fill the transition probability matrix
    for transition in transitions:
        parts = [part.strip() for part in transition.split("->")]
        num_parts = len(set(parts))
        prob_per_transition = 1.0 / (num_parts - 1)

        for i in range(num_parts - 1):
            from_state = state_to_index[parts[i]]
            to_state = state_to_index[parts[i + 1]]
            transition_probs[from_state, to_state] += prob_per_transition

    return transition_probs


def generate_toy_shard(num_subjects: int, measurements_per_subject: int) -> pl.DataFrame:
    """Generates a toy repeating shard with a simple code pattern.

    Args:
        num_subjects: Number of subjects to generate.
        measurements_per_subject: Number of measurements per subject.

    Returns:
        A DataFrame with a repeating pattern.

    Examples:
        >>> generate_toy_shard(2, 5)
    """

    raise NotImplementedError("This function is not implemented yet.")


def generate_toy_data(num_subjects_per_split: int, measurements_per_subject: int) -> MEDSDataset:
    """Generate a toy dataset with a simple code pattern.

    Args:
        num_subjects_per_split: How many subjects per split.
        measurements_per_subject: How many measurements per subject.

    Returns:
        A MEDSDataset with a repeating pattern.

    Examples:
        >>> generate_toy_data(2, 5)
    """

    data_shards = {
        f"data/{train_split}": generate_toy_shard(num_subjects_per_split, measurements_per_subject),
        f"data/{tuning_split}": generate_toy_shard(num_subjects_per_split, measurements_per_subject),
        f"data/{held_out_split}": generate_toy_shard(num_subjects_per_split, measurements_per_subject),
    }

    return MEDSDataset(data_shards)


def preprocessed_generated_dataset(*args, **kwargs) -> Path:
    """Fixture to create a preprocessed dataset."""

    generated_data = generate_toy_data(*args, **kwargs)

    with tempfile.TemporaryDirectory() as test_root:
        test_root = Path(test_root)

        input_dir = test_root / "raw_input"
        generated_data.write(input_dir)

        interemediate_dir = test_root / "intermediate"
        output_dir = test_root / "output"

        cmd = [
            "MEICAR_process_data",
            f"input_dir={input_dir!s}",
            f"intermediate_dir={interemediate_dir!s}",
            f"output_dir={output_dir!s}",
            "do_demo=True",
        ]

        out = subprocess.run(cmd, capture_output=True, check=False)

        err_lines = [
            "Command failed:",
            "Stdout:",
            out.stdout.decode(),
            "Stderr:",
            out.stderr.decode(),
        ]

        if out.returncode != 0:
            raise ValueError("\n".join([*err_lines, f"Return code: {out.returncode}"]))

        yield output_dir


def pretrained_toy_model(preprocessed_generated_dataset: Path) -> Path:
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)

        cmd = [
            "MEICAR_pretrain",
            "--config-name=_demo_pretrain",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={preprocessed_generated_dataset!s}",
        ]

        out = subprocess.run(cmd, capture_output=True, check=False)

        err_lines = [
            "Command failed:",
            "Stdout:",
            out.stdout.decode(),
            "Stderr:",
            out.stderr.decode(),
        ]

        if out.returncode != 0:
            raise ValueError("\n".join([*err_lines, f"Return code: {out.returncode}"]))
        yield output_dir


# def toy_generated_trajectories(pretrained_toy_model: Path) -> Path:
#    tensorized_cohort_dir, task_root_dir, task_name = preprocessed_dataset_with_task
#    model_initialization_dir = pretrained_model
#
#    with tempfile.TemporaryDirectory() as output_dir:
#        output_dir = Path(output_dir)
#
#        cmd = [
#            "MEICAR_generate_trajectories",
#            "--config-name=_demo_generate_trajectories",
#            f"output_dir={output_dir!s}",
#            f"model_initialization_dir={model_initialization_dir!s}",
#            f"datamodule.config.tensorized_cohort_dir={tensorized_cohort_dir!s}",
#            f"datamodule.config.task_labels_dir={(task_root_dir / task_name)!s}",
#            "datamodule.batch_size=2",
#            "trainer=demo",
#        ]
#
#        out = subprocess.run(cmd, capture_output=True, check=False)
#
#        err_lines = [
#            "Command failed:",
#            "Stdout:",
#            out.stdout.decode(),
#            "Stderr:",
#            out.stderr.decode(),
#        ]
#
#        if out.returncode != 0:
#            raise ValueError("\n".join([*err_lines, f"Return code: {out.returncode}"]))
#        yield output_dir
