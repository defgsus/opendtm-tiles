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
    tiles_map = pathconfig.tile_cache_file_map(zoom=zoom)

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
            resolution = data[0]
        if array is None:
            array = np.zeros((
                (max_y - min_y + 1) * (resolution + padding),
                (max_x - min_x + 1) * (resolution + padding),
                *data.shape[2:]
            ))

        if small_res is not None:
            data = cv2.resize(data, (small_res, small_res), cv2.INTER_NEAREST)
        array[
            y * (resolution + padding): y * (resolution + padding) + resolution,
            x * (resolution + padding): x * (resolution + padding) + resolution
        ] = data

    image = PIL.Image.fromarray(
        (array[..., None].repeat(3, -1) / 4).clip(0, 255).astype(np.uint8)
    )
    image.save(f"reproject-preview-z{zoom}-r{resolution}.png")
