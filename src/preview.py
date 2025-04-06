import os
from pathlib import Path
from typing import List, Tuple, Optional

from tqdm import tqdm
import numpy as np
import cv2
import PIL.Image

from .opendtm import OpenDTM
from . import config
from .files import PathConfig, DeleteFileOnException


def command_preview(
        pathconfig: PathConfig,
        modality: str,
        zoom: int,
        resolution: Optional[int],
        padding: int,
        numbers: bool,
        verbose: bool,
):
    tiles_map = pathconfig.tile_cache_file_map(zoom=zoom, modality=modality)

    if not tiles_map:
        print(f"No tiles at {pathconfig.tile_cache_path(modality=modality)}/{zoom}")
        return

    min_x = min(t[0] for t in tiles_map)
    min_y = min(t[1] for t in tiles_map)
    max_x = max(t[0] for t in tiles_map)
    max_y = max(t[1] for t in tiles_map)

    if numbers:
        print(f"z={zoom}")
        print(f"x={(min_x, max_x)} width={max_x - min_x}")
        print(f"y={(min_y, max_y)} height={max_y - min_y}")
        return

    array = None
    small_res = resolution
    for (x, y), filename in tqdm(tiles_map.items(), desc="tiles", disable=not verbose):
        x = x - min_x
        y = y - min_y

        try:
            data = np.load(filename).get("arr_0")
        except KeyboardInterrupt:
            raise
        except:
            if resolution is None:
                raise
            data = np.zeros((resolution, resolution, *array.shape[2:]))
            if data.ndim == 3:
                data[..., 0] = 1

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
    array[nan_mask] = -1 if modality == "normal" else 0

    if modality == "normal":
        array = array * .5 + .5
        array = np.pad(array, ((0, 0), (0, 0), (0, 1)))
    else:
        #array /= 2000  # TODO: get max height in dataset
        array = (1. - np.abs(.5 - array % 1.)*10.).clip(0, 1)
        array = array[..., None].repeat(4, -1)

    array[..., 3] = 1. - nan_mask

    image = PIL.Image.fromarray(
        (array * 255).clip(0, 255).astype(np.uint8)
    )
    filename = Path(
        f"preview/{modality}-preview-z{zoom}-r{resolution}-p{padding}.png"
    ).resolve()
    os.makedirs(filename.parent, exist_ok=True)
    with DeleteFileOnException(filename):
        image.save(filename)
    if verbose:
        print(f"file://{filename}")

