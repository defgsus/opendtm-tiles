import os
import warnings
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool
from typing import List, Tuple, Optional

from tqdm import tqdm
import numpy as np
import cv2
import PIL.Image

from .files import PathConfig, DeleteFileOnException, split_tile_file_map


def command_downsample(
        pathconfig: PathConfig,
        modality: str,
        zoom: List[int],
        workers: int,
        overwrite: bool,
        verbose: bool,
):
    if len(zoom) == 1:
        zoom = [zoom[0], zoom[0]]
    elif len(zoom) != 2:
        raise ValueError(f"zoom must be one or two numbers, got {zoom}")

    for zoom in range(zoom[0], zoom[1] - 1, -1):
        tiles_map = pathconfig.tile_output_file_map(zoom=zoom, modality=modality)
        downsampled_map = _get_downsampled_tiles_map(tiles_map)

        kwargs = dict(
            modality=modality,
            pathconfig=pathconfig,
            zoom=zoom,
            verbose=verbose,
            overwrite=overwrite,
        )

        if workers <= 1 or len(downsampled_map) < workers*10:
            _downsample_level(downsampled_map=downsampled_map, **kwargs)
        else:
            downsampled_map_batches = split_tile_file_map(downsampled_map, workers)
            with Pool(workers) as pool:
                pool.map(
                    _downsample_level_kwargs,
                    [
                        {
                            "downsampled_map": downsampled_batch,
                            "tqdm_position": i,
                            **kwargs,
                        }
                        for i, downsampled_batch in enumerate(downsampled_map_batches)
                    ]
                )


def _downsample_level_kwargs(kwargs: dict):
    _downsample_level(**kwargs)

def _downsample_level(
        downsampled_map,
        modality: str,
        pathconfig: PathConfig,
        zoom: int,
        verbose: bool,
        overwrite: bool,
        tqdm_position: int = 0,
):
    progress = tqdm(downsampled_map.items(), desc=f"downsampling {zoom}->{zoom-1}", disable=not verbose, position=tqdm_position)
    num_incomplete = 0
    num_skipped = 0
    for (x0, y0), up_tiles in progress:
        progress.set_postfix({"skipped": num_skipped, "incomplete": num_incomplete})

        if len(up_tiles) != 2:
            num_incomplete += 1

        if not overwrite and pathconfig.tile_output_exists(zoom - 1, x0, y0, modality=modality):
            num_skipped += 1
            continue

        tile = None
        num_pasted = 0
        for (x, y) in up_tiles:
            try:
                up_tile = pathconfig.load_tile_output_file(zoom, x, y, modality=modality)
            except Exception as e:
                warnings.warn(f"{type(e).__name__}: {e}: {pathconfig.tile_output_filename(zoom, x, y, modality=modality)}")
                continue
            if tile is None:
                tile = PIL.Image.new(
                    mode=up_tile.mode,
                    size=(up_tile.width * 2, up_tile.height * 2),
                    color=(0,) * len(up_tile.mode),
                )
            try:
                tile.paste(up_tile, ((x - x0*2) * up_tile.width, (y - y0*2) * up_tile.height))
                num_pasted += 1
            except Exception as e:
                warnings.warn(f"{type(e).__name__}: {e}: {pathconfig.tile_output_filename(zoom, x, y, modality=modality)}")

        if tile is None or not num_pasted:
            continue

        tile = tile.resize((up_tile.width, up_tile.height), PIL.Image.Resampling.BICUBIC)
        pathconfig.save_output_tile(zoom - 1, x0, y0, tile, modality=modality)



def _get_downsampled_tiles_map(tiles_map):
    downsampled_map = {}
    for (x, y), file in tiles_map.items():
        key = (x // 2, y // 2)
        if key not in downsampled_map:
            downsampled_map[key] = []
        downsampled_map[key].append((x, y))
    return downsampled_map
