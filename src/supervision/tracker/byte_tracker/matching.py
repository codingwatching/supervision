from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from supervision.detection.utils.iou_and_nms import box_iou_batch

if TYPE_CHECKING:
    from supervision.tracker.byte_tracker.single_object_track import STrack


def indices_to_matches(
    cost_matrix: npt.NDArray[np.float32], indices: npt.NDArray[np.int_], thresh: float
) -> tuple[npt.NDArray[np.int_], tuple[int, ...], tuple[int, ...]]:
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b


def linear_assignment(
    cost_matrix: npt.NDArray[np.float32], thresh: float
) -> tuple[npt.NDArray[np.int_], tuple[int, ...], tuple[int, ...]]:
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    indices = np.column_stack((row_ind, col_ind))

    return indices_to_matches(cost_matrix, indices, thresh)


def iou_distance(
    atracks: list[STrack] | list[npt.NDArray[np.float32]],
    btracks: list[STrack] | list[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs: list[npt.NDArray[np.float32]] = cast(
            list[npt.NDArray[np.float32]], atracks
        )
        btlbrs: list[npt.NDArray[np.float32]] = cast(
            list[npt.NDArray[np.float32]], btracks
        )
    else:
        tracks_a = cast(list[STrack], atracks)
        tracks_b = cast(list[STrack], btracks)
        atlbrs = [track.tlbr for track in tracks_a]
        btlbrs = [track.tlbr for track in tracks_b]

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    ious: npt.NDArray[np.float32] = box_iou_batch(
        np.asarray(atlbrs), np.asarray(btlbrs)
    )
    cost_matrix: npt.NDArray[np.float32] = np.asarray(1 - ious, dtype=np.float32)

    return cost_matrix


def fuse_score(
    cost_matrix: npt.NDArray[np.float32], stracks: list[STrack]
) -> npt.NDArray[np.float32]:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores: npt.NDArray[np.float32] = np.asarray(
        [strack.score for strack in stracks], dtype=np.float32
    )
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim: npt.NDArray[np.float32] = np.asarray(
        iou_sim * det_scores, dtype=np.float32
    )
    fuse_cost: npt.NDArray[np.float32] = np.asarray(1 - fuse_sim, dtype=np.float32)
    return fuse_cost
