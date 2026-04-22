"""Public entry point for finalizing ``Trainer.predict`` output into per-trajectory parquets.

``write_predictions`` is the single function ``__main__`` calls. Internally it composes three
module-private helpers (``_demux_predictions_per_trajectory``, ``_assert_expected_subject_index_order``,
``_write_per_trajectory_parquets``) that each carry focused doctests for the silent-failure modes
they guard against. Outside this module, only ``write_predictions`` is considered stable.
"""

import logging
from pathlib import Path

import pyarrow.parquet as pq
import torch
from MEDS_trajectory_evaluation.schema import GeneratedTrajectorySchema

from .format_trajectories import format_trajectories

logger = logging.getLogger(__name__)


def _demux_predictions_per_trajectory(
    predictions: list[dict[str, torch.Tensor]],
    n_trajectories: int,
) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]]]:
    """Demux a flat list of per-batch predict outputs into per-trajectory token + subject-idx lists.

    Each batch in ``predictions`` carries three tensors: ``tokens`` (``[B, L]``), ``subject_idxs``
    (``[B]``), and ``trajectory_idxs`` (``[B]``, values in ``0..n_trajectories-1``). We demux by
    trajectory so that downstream formatting can process each trajectory's rows as one stream. The
    per-batch grouping is preserved (rather than flattened) because each batch has its own ``L``
    from the generator.

    Iterates over ``trajectory_idxs.unique()`` per batch rather than doing ``n_trajectories``
    boolean compares unconditionally, so tail batches with fewer than ``n_trajectories`` rows
    don't pay for empty buckets.

    Examples:
        >>> preds = [
        ...     {"tokens": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        ...      "subject_idxs": torch.tensor([0, 0, 1, 1]),
        ...      "trajectory_idxs": torch.tensor([0, 1, 0, 1])},
        ... ]
        >>> tok, sidx = _demux_predictions_per_trajectory(preds, n_trajectories=2)
        >>> [b.tolist() for b in tok[0]]
        [[[1, 2], [5, 6]]]
        >>> [b.tolist() for b in tok[1]]
        [[[3, 4], [7, 8]]]
        >>> [b.tolist() for b in sidx[0]]
        [[0, 1]]
        >>> [b.tolist() for b in sidx[1]]
        [[0, 1]]

        A trajectory absent from a batch simply has no entry appended for that batch:

        >>> preds = [
        ...     {"tokens": torch.tensor([[1, 2], [3, 4]]),
        ...      "subject_idxs": torch.tensor([0, 1]),
        ...      "trajectory_idxs": torch.tensor([0, 0])},
        ... ]
        >>> _, sidx = _demux_predictions_per_trajectory(preds, n_trajectories=2)
        >>> [b.tolist() for b in sidx[0]]
        [[0, 1]]
        >>> [b.tolist() for b in sidx[1]]
        []

        Across multiple batches, per-trajectory batches are preserved in arrival order:

        >>> preds = [
        ...     {"tokens": torch.tensor([[1, 2]]),
        ...      "subject_idxs": torch.tensor([0]),
        ...      "trajectory_idxs": torch.tensor([0])},
        ...     {"tokens": torch.tensor([[3, 4]]),
        ...      "subject_idxs": torch.tensor([1]),
        ...      "trajectory_idxs": torch.tensor([0])},
        ... ]
        >>> _, sidx = _demux_predictions_per_trajectory(preds, n_trajectories=1)
        >>> [b.tolist() for b in sidx[0]]
        [[0], [1]]
    """
    per_trajectory_batches: dict[int, list[torch.Tensor]] = {t: [] for t in range(n_trajectories)}
    per_trajectory_subject_idxs: dict[int, list[torch.Tensor]] = {t: [] for t in range(n_trajectories)}
    for pred in predictions:
        tokens = pred["tokens"]
        trajectory_idxs = pred["trajectory_idxs"]
        subject_idxs = pred["subject_idxs"]
        for t in trajectory_idxs.unique().tolist():
            mask = trajectory_idxs == t
            per_trajectory_batches[t].append(tokens[mask])
            per_trajectory_subject_idxs[t].append(subject_idxs[mask])
    return per_trajectory_batches, per_trajectory_subject_idxs


def _assert_expected_subject_index_order(
    per_trajectory_subject_idxs: dict[int, list[torch.Tensor]],
    n_subjects: int,
) -> None:
    """Fail loud if per-trajectory subject_idxs are not ``[0, 1, ..., n_subjects - 1]``.

    ``format_trajectories`` walks ``base_dataset.schema_df`` via a running ``slice(st_i, B_i)``,
    assuming the concatenated predictions for each trajectory arrive in subject-index order. That
    assumption holds for the configured single-device predict path
    (``configs/trainer/generate.yaml`` pins ``devices: 1``), but any future move to DDP-parallel
    predict, any accidental shuffle, or any change to ``RepeatedPredictionDataset`` ordering would
    silently misalign the written ``(subject_id, prediction_time)`` metadata. This assert fails
    fast with a clear diagnostic instead. DDP-parallel generation is tracked at #142 and would
    replace this check with an explicit reassembly step.

    Examples:
        In-order predictions across multiple batches pass:

        >>> _assert_expected_subject_index_order(
        ...     {0: [torch.tensor([0, 1, 2]), torch.tensor([3, 4])]}, n_subjects=5,
        ... )

        Out-of-order (shuffle, DDP gather) raises:

        >>> _assert_expected_subject_index_order(
        ...     {0: [torch.tensor([2, 0, 1])]}, n_subjects=3,
        ... )
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 0 predictions arrived out of expected subject-index order...

        Short (missing rows) raises:

        >>> _assert_expected_subject_index_order(
        ...     {0: [torch.tensor([0, 1, 2])]}, n_subjects=5,
        ... )
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 0 predictions arrived out of expected subject-index order...

        Long (duplicate / oversample) raises:

        >>> _assert_expected_subject_index_order(
        ...     {0: [torch.tensor([0, 1, 2, 0])]}, n_subjects=3,
        ... )
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 0 predictions arrived out of expected subject-index order...

        Empty per-trajectory list raises if any rows were expected:

        >>> _assert_expected_subject_index_order({0: []}, n_subjects=3)
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 0 predictions arrived out of expected subject-index order...

        The second trajectory's failure is reported (not only the first):

        >>> _assert_expected_subject_index_order(
        ...     {0: [torch.tensor([0, 1])], 1: [torch.tensor([1, 0])]}, n_subjects=2,
        ... )
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 1 predictions arrived out of expected subject-index order...
    """
    expected_order = torch.arange(n_subjects)
    for t, subj_idxs_batches in per_trajectory_subject_idxs.items():
        got = torch.cat(subj_idxs_batches) if subj_idxs_batches else torch.empty(0, dtype=torch.long)
        if got.shape != expected_order.shape or not torch.equal(got, expected_order):
            raise RuntimeError(
                f"Trajectory {t} predictions arrived out of expected subject-index order: "
                f"got {got.tolist()[:10]}... (len {got.shape[0]}); "
                f"expected [0, 1, ..., {n_subjects - 1}]. This can happen under multi-device "
                "predict (DDP stripes rows across ranks) or if the expanded dataloader is "
                "shuffled. Pin `trainer.devices=1` for generation; see issue #142 for the "
                "streaming-reassembly follow-up."
            )


def _write_per_trajectory_parquets(
    per_trajectory_batches: dict[int, list[torch.Tensor]],
    trajectory_paths: dict[int, Path],
    base_dataset,
    do_overwrite: bool,
) -> None:
    """Format each trajectory's demuxed batches via ``format_trajectories`` and write them out.

    Idempotent per-file: if ``do_overwrite`` is ``False`` and a trajectory's parquet is already on
    disk, that trajectory is skipped and no ``format_trajectories`` work is done for it. The
    split-level all-parquets-exist short-circuit in the caller handles the zero-work case.
    """
    for trajectory_idx, out_fp in trajectory_paths.items():
        if out_fp.is_file() and not do_overwrite:
            logger.info(f"Skipping {out_fp} as it already exists.")
            continue
        logger.info(f"Writing trajectory {trajectory_idx} to {out_fp}.")
        predictions_df = format_trajectories(base_dataset, per_trajectory_batches[trajectory_idx])
        pa_table = GeneratedTrajectorySchema.align(predictions_df.to_arrow())
        pq.write_table(pa_table, out_fp)


def write_predictions(
    predictions: list[dict[str, torch.Tensor]],
    *,
    n_trajectories: int,
    trajectory_paths: dict[int, Path],
    base_dataset,
    do_overwrite: bool,
) -> None:
    """Format a flat list of ``Trainer.predict`` per-batch outputs into per-trajectory parquets.

    Given:

    - ``predictions``: what ``Trainer.predict(model=MEICARModule, dataloaders=...)`` returns when
      the Lightning module's ``predict_step`` emits the three-key dict this module expects
      (``tokens``, ``subject_idxs``, ``trajectory_idxs``).
    - ``n_trajectories``: the ``N`` used to expand the base dataset via
      :class:`~MEDS_EIC_AR.generation.RepeatedPredictionDataset`.
    - ``trajectory_paths``: the mapping ``trajectory_idx → output parquet path`` for the split
      being written.
    - ``base_dataset``: the un-expanded ``MEDSPytorchDataset`` whose ``schema_df`` supplies
      ``(subject_id, prediction_time, last_time)`` metadata per row, in order.
    - ``do_overwrite``: if ``False``, per-trajectory parquets that already exist are left alone.

    Writes one parquet per trajectory in ``trajectory_paths``.

    Internally three steps: (1) demux predictions into per-trajectory streams, (2) assert the
    arrival order is subject-index-sorted so ``format_trajectories``'s ``slice(st_i, B)`` walk is
    valid, (3) format + write each trajectory's parquet. The assert-step is the safety net that
    turns any future DDP / shuffle / sampler regression into a loud error rather than silent
    parquet corruption.
    """
    per_trajectory_batches, per_trajectory_subject_idxs = _demux_predictions_per_trajectory(
        predictions, n_trajectories
    )
    _assert_expected_subject_index_order(per_trajectory_subject_idxs, n_subjects=len(base_dataset))
    _write_per_trajectory_parquets(
        per_trajectory_batches, trajectory_paths, base_dataset, do_overwrite=do_overwrite
    )
