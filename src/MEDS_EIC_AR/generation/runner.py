"""Public entry point for finalizing ``Trainer.predict`` output into per-trajectory parquets.

``write_predictions`` is the single function ``__main__`` calls. Internally it composes three
module-private helpers (``_demux_predictions_per_trajectory``,
``_sort_per_trajectory_by_dataset_row_index``, ``_write_per_trajectory_parquets``) that each carry
focused doctests for the invariants they enforce. Outside this module, only ``write_predictions``
is considered stable.
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
    """Demux a flat list of per-batch predict outputs into per-trajectory token + dataset-row-idx lists.

    Each batch in ``predictions`` carries three tensors: ``tokens`` (``[B, L]``), ``dataset_row_idxs``
    (``[B]``), and ``trajectory_idxs`` (``[B]``, values in ``0..n_trajectories-1``). We demux by
    trajectory so that downstream formatting can process each trajectory's rows as one stream. The
    per-batch grouping is preserved (rather than flattened) because each batch has its own ``L``
    from the generator — ``model.generate`` stops at the batch-local stopping condition (all rows
    EOS or ``max_new_tokens``), which varies batch-to-batch.

    Iterates over ``trajectory_idxs.unique()`` per batch rather than doing ``n_trajectories``
    boolean compares unconditionally, so tail batches with fewer than ``n_trajectories`` rows
    don't pay for empty buckets.

    Examples:
        >>> preds = [
        ...     {"tokens": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        ...      "dataset_row_idxs": torch.tensor([0, 0, 1, 1]),
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
        ...      "dataset_row_idxs": torch.tensor([0, 1]),
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
        ...      "dataset_row_idxs": torch.tensor([0]),
        ...      "trajectory_idxs": torch.tensor([0])},
        ...     {"tokens": torch.tensor([[3, 4]]),
        ...      "dataset_row_idxs": torch.tensor([1]),
        ...      "trajectory_idxs": torch.tensor([0])},
        ... ]
        >>> _, sidx = _demux_predictions_per_trajectory(preds, n_trajectories=1)
        >>> [b.tolist() for b in sidx[0]]
        [[0], [1]]
    """
    per_trajectory_batches: dict[int, list[torch.Tensor]] = {t: [] for t in range(n_trajectories)}
    per_trajectory_dataset_row_idxs: dict[int, list[torch.Tensor]] = {t: [] for t in range(n_trajectories)}
    for pred in predictions:
        tokens = pred["tokens"]
        trajectory_idxs = pred["trajectory_idxs"]
        dataset_row_idxs = pred["dataset_row_idxs"]
        for t in trajectory_idxs.unique().tolist():
            mask = trajectory_idxs == t
            per_trajectory_batches[t].append(tokens[mask])
            per_trajectory_dataset_row_idxs[t].append(dataset_row_idxs[mask])
    return per_trajectory_batches, per_trajectory_dataset_row_idxs


def _sort_per_trajectory_by_dataset_row_index(
    per_trajectory_batches: dict[int, list[torch.Tensor]],
    per_trajectory_dataset_row_idxs: dict[int, list[torch.Tensor]],
    n_dataset_rows: int,
) -> dict[int, list[torch.Tensor]]:
    """Reassemble each trajectory's rows into dataset-row-index order, one row per output element.

    ``format_trajectories`` downstream walks ``base_dataset.schema_df`` via a running
    ``slice(st_i, B_i)``, which requires the per-row stream for each trajectory to be in ascending
    dataset-row-index order. On the default single-device predict path that holds trivially, but
    any reshuffle — an accidental ``shuffle=True`` on the expanded loader, a manual cross-rank
    gather in a DDP-enabled successor (see #146; Lightning's ``Trainer.predict`` does *not*
    auto-gather across ranks, so any DDP story has to wire one explicitly), or a future change to
    ``RepeatedPredictionDataset``'s expansion order — would otherwise silently misalign the
    ``(subject_id, prediction_time)`` metadata. This helper closes that gap by using the
    already-tracked ``dataset_row_idxs`` to reassemble.

    Because different batches carry different ``L`` (``model.generate`` stops at per-batch
    conditions), we avoid padding entirely: each row is emitted as a ``[1, L_i]`` single-row
    tensor, preserving its native length. ``format_trajectories`` already iterates a list of
    per-batch tensors with its own ``L``, so 1-row batches are a no-op change to it.

    Raises ``RuntimeError`` if the set of dataset_row_idxs for any trajectory is not exactly
    ``{0, 1, ..., n_dataset_rows - 1}``. This catches missing rows, duplicates, or bookkeeping errors
    in the reassembly itself — it is a post-condition on the sorted output, not a pre-condition
    on arrival order.

    Examples:
        In-order single-batch input — output is just the rows split into 1-row tensors:

        >>> batches = {0: [torch.tensor([[10, 11], [20, 21], [30, 31]])]}
        >>> sidxs = {0: [torch.tensor([0, 1, 2])]}
        >>> out = _sort_per_trajectory_by_dataset_row_index(batches, sidxs, n_dataset_rows=3)
        >>> [b.tolist() for b in out[0]]
        [[[10, 11]], [[20, 21]], [[30, 31]]]

        Shuffled input (simulating DDP gather) — rows come out in dataset-row-index order,
        carrying the right token rows:

        >>> batches = {0: [torch.tensor([[20, 21], [0, 1], [10, 11]])]}
        >>> sidxs = {0: [torch.tensor([2, 0, 1])]}
        >>> out = _sort_per_trajectory_by_dataset_row_index(batches, sidxs, n_dataset_rows=3)
        >>> [b.tolist() for b in out[0]]
        [[[0, 1]], [[10, 11]], [[20, 21]]]

        Mixed-``L`` across batches (this is what lets us avoid padding). Batch 1 has ``L=2``,
        batch 2 has ``L=3``; output keeps the native lengths:

        >>> batches = {
        ...     0: [torch.tensor([[10, 11], [30, 31]]),
        ...         torch.tensor([[0, 1, 2], [20, 21, 22]])],
        ... }
        >>> sidxs = {0: [torch.tensor([1, 3]), torch.tensor([0, 2])]}
        >>> out = _sort_per_trajectory_by_dataset_row_index(batches, sidxs, n_dataset_rows=4)
        >>> [b.tolist() for b in out[0]]
        [[[0, 1, 2]], [[10, 11]], [[20, 21, 22]], [[30, 31]]]

        Multiple trajectories are sorted independently:

        >>> batches = {
        ...     0: [torch.tensor([[1, 1], [0, 0]])],
        ...     1: [torch.tensor([[2, 2], [3, 3]])],
        ... }
        >>> sidxs = {0: [torch.tensor([1, 0])], 1: [torch.tensor([0, 1])]}
        >>> out = _sort_per_trajectory_by_dataset_row_index(batches, sidxs, n_dataset_rows=2)
        >>> [b.tolist() for b in out[0]]
        [[[0, 0]], [[1, 1]]]
        >>> [b.tolist() for b in out[1]]
        [[[2, 2]], [[3, 3]]]

        Missing rows raises:

        >>> _sort_per_trajectory_by_dataset_row_index(
        ...     {0: [torch.tensor([[10, 11], [20, 21]])]},
        ...     {0: [torch.tensor([0, 1])]},
        ...     n_dataset_rows=3,
        ... )
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 0 dataset-row-index set is not {0, 1, ..., 2}: got [0, 1]...

        Duplicate dataset_row_idxs raises:

        >>> _sort_per_trajectory_by_dataset_row_index(
        ...     {0: [torch.tensor([[10, 11], [10, 11], [20, 21]])]},
        ...     {0: [torch.tensor([0, 0, 1])]},
        ...     n_dataset_rows=2,
        ... )
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 0 dataset-row-index set is not {0, 1, ..., 1}: got [0, 0, 1]...

        Out-of-range dataset_row_idx raises:

        >>> _sort_per_trajectory_by_dataset_row_index(
        ...     {0: [torch.tensor([[10, 11], [20, 21], [30, 31]])]},
        ...     {0: [torch.tensor([0, 1, 5])]},
        ...     n_dataset_rows=3,
        ... )
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 0 dataset-row-index set is not {0, 1, ..., 2}: got [0, 1, 5]...

        Empty input when rows are expected raises:

        >>> _sort_per_trajectory_by_dataset_row_index({0: []}, {0: []}, n_dataset_rows=3)
        Traceback (most recent call last):
          ...
        RuntimeError: Trajectory 0 dataset-row-index set is not {0, 1, ..., 2}: got []...
    """
    sorted_per_trajectory: dict[int, list[torch.Tensor]] = {}
    for t, token_batches in per_trajectory_batches.items():
        dataset_row_idx_batches = per_trajectory_dataset_row_idxs[t]

        if token_batches:
            all_tokens_per_row = [row for batch in token_batches for row in batch]
            all_dataset_row_idxs = torch.cat(dataset_row_idx_batches)
        else:
            all_tokens_per_row = []
            all_dataset_row_idxs = torch.empty(0, dtype=torch.long)

        # 1:1 correspondence between token rows and dataset_row_idxs. Should always hold —
        # both come from the same predict_step output dict and are built by the same demux
        # mask above — but guard explicitly so any upstream bookkeeping drift fails here
        # with a clear diagnostic instead of silently mis-pairing rows or raising IndexError
        # inside the sort loop below.
        if len(all_tokens_per_row) != all_dataset_row_idxs.numel():
            raise RuntimeError(
                f"Trajectory {t} token-row / dataset_row_idx length mismatch: "
                f"{len(all_tokens_per_row)} tokens vs {all_dataset_row_idxs.numel()} indices. "
                "This should be impossible — both are demuxed from the same predict_step "
                "output dict. Check for upstream masking / metadata-plumbing drift."
            )

        # Validate and sort in a single tensor pass. Argsort + compare against ``arange`` is
        # the set-equality test for "indices form a complete permutation of ``range(n)``" and
        # avoids materializing a Python ``set(range(n))`` (which costs 40+ bytes per int for
        # tens of millions of rows).
        order = torch.argsort(all_dataset_row_idxs, stable=True)
        sorted_idxs = all_dataset_row_idxs[order]
        expected = torch.arange(n_dataset_rows, dtype=sorted_idxs.dtype)
        if sorted_idxs.shape != expected.shape or not torch.equal(sorted_idxs, expected):
            raise RuntimeError(
                f"Trajectory {t} dataset-row-index set is not {{0, 1, ..., {n_dataset_rows - 1}}}: "
                f"got {all_dataset_row_idxs.tolist()[:10]}... (len {all_dataset_row_idxs.numel()}). "
                "Every base-dataset row index must appear exactly once per trajectory. A "
                "missing or duplicate index indicates a sampler bug, an accidental dataloader "
                "shuffle, or — for a hypothetical DDP path — an incomplete cross-rank gather "
                "(Lightning's ``Trainer.predict`` does not auto-gather; see issue #146)."
            )

        sorted_per_trajectory[t] = [all_tokens_per_row[i].unsqueeze(0) for i in order.tolist()]
    return sorted_per_trajectory


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
      (``tokens``, ``dataset_row_idxs``, ``trajectory_idxs``).
    - ``n_trajectories``: the ``N`` used to expand the base dataset via
      :class:`~MEDS_EIC_AR.generation.RepeatedPredictionDataset`.
    - ``trajectory_paths``: the mapping ``trajectory_idx → output parquet path`` for the split
      being written.
    - ``base_dataset``: the un-expanded ``MEDSPytorchDataset`` whose ``schema_df`` supplies
      ``(subject_id, prediction_time, last_time)`` metadata per row, in order.
    - ``do_overwrite``: if ``False``, per-trajectory parquets that already exist are left alone.

    Writes one parquet per trajectory in ``trajectory_paths``.

    Internally: (1) demux predictions into per-trajectory streams, (2) reassemble each
    trajectory's rows into dataset-row-index order using the already-tracked ``dataset_row_idxs`` (robust
    to DDP gather order, shuffled loaders, or future sampler changes), (3) format + write each
    trajectory's parquet. The reassembly step makes arrival order irrelevant for correctness; the
    validation inside it catches missing / duplicate / out-of-range dataset_row_idxs as a loud error
    rather than silent parquet corruption.
    """
    per_trajectory_batches, per_trajectory_dataset_row_idxs = _demux_predictions_per_trajectory(
        predictions, n_trajectories
    )
    sorted_per_trajectory = _sort_per_trajectory_by_dataset_row_index(
        per_trajectory_batches, per_trajectory_dataset_row_idxs, n_dataset_rows=len(base_dataset)
    )
    _write_per_trajectory_parquets(
        sorted_per_trajectory, trajectory_paths, base_dataset, do_overwrite=do_overwrite
    )
