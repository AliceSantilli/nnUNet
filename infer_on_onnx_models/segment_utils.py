"""Utility functions for inference of the onnx model for PET Segmentation
Author: Alice Santilli <santila@mskcc.org>
"""

from __future__ import annotations

__all__ = [
    "calculate_gaussian_filter",
    "compute_steps_for_sliding_window",
    "internal_3Dconv_tiled",
    "left_squeeze",
]

import builtins
import os
import typing
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage.filters import gaussian_filter

# import torch



ort_verbosity = int(os.getenv("ORT_VERBOSITY", default=2))
if ort_verbosity < 1:
    verbose = True
else:
    verbose = False


def left_squeeze(array: npt.NDArray, *, n: builtins.int = 1) -> npt.NDArray:
    """remove up to 'n' empty dimensions from the left-most dims of an ndarray
    Examples:
        >>> left_squeeze(np.array([[0]]))
        array([0])
        >>> left_squeeze(np.zeros((1,1,2)), n=2)
        array([0., 0.])
        >>> left_squeeze(np.zeros((2,1,2)), n=2)
        array([[0., 0.],
               [0., 0.]])
    """
    axis = tuple(i for i, s in enumerate(array.shape[:n]) if s <= 1)
    return array.squeeze(axis=axis)


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


def internal_3Dconv_tiled(
    data: np.ndarray,
    patch_size: tuple,
    steps,
    step_size: float = 0.5,
    num_classes: int = 1,
    do_mirroring: bool = False,
    mirror_axes: Tuple[int, ...] = (0, 1, 2),
    # all_in_gpu: False, code can be extended to run on GPU as done in the code base
) -> typing.Generator[np.ndarray, int, int, int, int, int, int]:
    """Creates patches for prediction as done in the prediction code of the nnUNet"""
    # https://github.com/MIC-DKFZ/nnUNet/blob/5e5a151649af7386a2885fe2529fe6aa49bd2866/nnunet/network_architecture/neural_network.py

    # better safe than sorry
    assert len(data.shape) == 4, "x must be (c, x, y, z)"

    if verbose:
        print("step_size:", step_size)
        print("do mirror:", do_mirroring)

    assert patch_size is not None, "patch_size cannot be None for tiled prediction"

    if verbose:
        print("data shape:", data.shape)
        print("patch size:", patch_size)
        # print("steps (x, y, and z):", steps)

    for x in steps[0]:
        lb_x = x
        ub_x = x + patch_size[0]
        for y in steps[1]:
            lb_y = y
            ub_y = y + patch_size[1]
            for z in steps[2]:
                lb_z = z
                ub_z = z + patch_size[2]

                # send to ONNX model in main app
                yield data, lb_x, ub_x, lb_y, ub_y, lb_z, ub_z


def calculate_gaussian_filter(patch_size: tuple, num_tiles):
    # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
    # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
    # can be expensive, so it makes sense to save and reuse them.
    _gaussian_3d = _patch_size_for_gaussian_3d = None
    # we only need to compute that once. It can take a while to compute this due to the large sigma in
    # gaussian_filter
    if num_tiles > 1:
        if _gaussian_3d is None or not all(
            [i == j for i, j in zip(patch_size, _patch_size_for_gaussian_3d)]
        ):
            if verbose:
                print("computing Gaussian")
            gaussian_importance_map = get_gaussian(patch_size, sigma_scale=1.0 / 8)

            _gaussian_3d = gaussian_importance_map
            _patch_size_for_gaussian_3d = patch_size
        else:
            if verbose:
                print("using precomputed Gaussian")
            gaussian_importance_map = _gaussian_3d
    else:
        gaussian_importance_map = None

    if num_tiles > 1:
        add_for_nb_of_preds = _gaussian_3d
    else:
        add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)

    return gaussian_importance_map, add_for_nb_of_preds


def compute_steps_for_sliding_window(
    patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float
) -> List[List[int]]:
    # compute the steps for sliding window
    assert [
        i >= j for i, j in zip(image_size, patch_size)
    ], "image size must be as large or larger than patch_size"
    assert (
        0 < step_size <= 1
    ), "step_size must be larger than 0 and smaller or equal to 1"
    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)
    ]
    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = (
                99999999999  # does not matter because there is only one step at 0
            )
        steps_here = [
            int(np.round(actual_step_size * i)) for i in range(num_steps[dim])
        ]
        steps.append(steps_here)

    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])
    return steps, num_tiles


def get_gaussian(patch_size, sigma_scale=1.0 / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    gaussian_importance_map = (
        gaussian_importance_map / np.max(gaussian_importance_map) * 1
    )
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0]
    )

    return gaussian_importance_map
