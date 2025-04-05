import os
from typing import List, Tuple, Optional

from tqdm import tqdm
import numpy as np
import cv2
import PIL.Image

from .opendtm import OpenDTM
from . import config
from .files import PathConfig


def command_reproject_preview(
        pathconfig: PathConfig,
        zoom: int,
        resolution: Optional[int],
        padding: int,
        verbose: bool,
):
    _reproject_preview(
        normal=False,
        pathconfig=pathconfig,
        zoom=zoom,
        resolution=resolution,
        padding=padding,
        verbose=verbose,
    )


def command_normal_map_preview(
        pathconfig: PathConfig,
        zoom: int,
        resolution: Optional[int],
        padding: int,
        verbose: bool,
):
    _reproject_preview(
        normal=True,
        pathconfig=pathconfig,
        zoom=zoom,
        resolution=resolution,
        padding=padding,
        verbose=verbose,
    )


def _reproject_preview(
        normal: bool,
        pathconfig: PathConfig,
        zoom: int,
        resolution: Optional[int],
        padding: int,
        verbose: bool,
):
    tiles_map = pathconfig.tile_cache_file_map(zoom=zoom, normal=normal)

    min_x = min(t[0] for t in tiles_map)
    min_y = min(t[1] for t in tiles_map)
    max_x = max(t[0] for t in tiles_map)
    max_y = max(t[1] for t in tiles_map)

    array = None
    small_res = resolution
    for (x, y), filename in tqdm(tiles_map.items(), desc="tiles", disable=not verbose):
        x = x - min_x
        y = y - min_y

        data = np.load(filename).get("arr_0")
        if resolution is None:
            resolution = data.shape[0]
        if array is None:
            array = np.zeros((
                (max_y - min_y + 1) * (resolution + padding),
                (max_x - min_x + 1) * (resolution + padding),
                *data.shape[2:]
            ))
            array -= 10_000.

        if small_res is not None:
            data = cv2.resize(data, (small_res, small_res), cv2.INTER_NEAREST)
        array[
            y * (resolution + padding): y * (resolution + padding) + resolution,
            x * (resolution + padding): x * (resolution + padding) + resolution
        ] = data

    nan_mask = np.isnan(array) | (array <= -10_000)
    if nan_mask.ndim == 3:
        nan_mask = nan_mask[..., 0]
    array[nan_mask] = -1 if normal else 0

    if normal:
        array = array * .5 + .5
        array = np.pad(array, ((0, 0), (0, 0), (0, 1)))
    else:
        array /= 2000  # TODO: get max height in dataset
        array = array[..., None].repeat(4, -1)

    array[..., 3] = 1. - nan_mask

    image = PIL.Image.fromarray(
        (array * 255).clip(0, 255).astype(np.uint8)
    )
    image.save(f"{'normal' if normal else 'reproject'}-preview-z{zoom}-r{resolution}-p{padding}.png")
