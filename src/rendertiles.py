import math
import os
import warnings
from multiprocessing.pool import ThreadPool as Pool
from typing import List, Tuple, Optional

import mercantile
from tqdm import tqdm
import numpy as np
import cv2
import PIL.Image

from . import config
from .files import PathConfig, DeleteFileOnException, split_tile_file_map


def command_render(
        pathconfig: PathConfig,
        modality: str,
        cache_zoom: int,
        tile_zoom: int,
        resolution: Optional[int],
        workers: int,
        verbose: bool,
):
    kwargs = dict(
        modality=modality,
        pathconfig=pathconfig,
        cache_zoom=cache_zoom,
        tile_zoom=tile_zoom,
        resolution=resolution,
        verbose=verbose,
    )
    tiles_map = pathconfig.tile_cache_file_map(zoom=cache_zoom, modality=modality)

    if workers <= 1:
        _render_tiles(tiles_map=tiles_map, **kwargs)
    else:
        tiles_map_batches = split_tile_file_map(tiles_map, workers)
        with Pool(workers) as pool:
            pool.map(
                _render_tiles_kwargs,
                [
                    {
                        "tiles_map": tiles_batch,
                        "tqdm_position": i,
                        **kwargs,
                    }
                    for i, tiles_batch in enumerate(tiles_map_batches)
                ]
            )


def _render_tiles_kwargs(kwargs: dict):
    _render_tiles(**kwargs)

def _render_tiles(
        tiles_map,
        modality: str,
        pathconfig: PathConfig,
        cache_zoom: int,
        tile_zoom: int,
        resolution: Optional[int],
        verbose: bool,
        interpolation: int = cv2.INTER_CUBIC,
        tqdm_position: int = 0,
):
    progress = tqdm(tiles_map.items(), desc="tiles", disable=not verbose, position=tqdm_position)

    def _iter_tiles(resolution: Optional[int]):
        data_resolution = None
        for (x, y), filename in progress:
            source_tile = mercantile.Tile(x, y, cache_zoom)

            data = None
            if data_resolution is None:
                try:
                    data = pathconfig.load_tile_cache_file(cache_zoom, x, y, modality=modality)
                except Exception as e:
                    warnings.warn(f"{type(e).__name__}: {e}: {pathconfig.tile_cache_filename(cache_zoom, x, y, modality=modality)}")
                    continue
                data_resolution = data.shape[0]
            if resolution is None:
                resolution = int(data_resolution * math.pow(2, cache_zoom - tile_zoom))
                if verbose:
                    print(f"Resampling from resolution {data.shape[0]}² to {resolution}²")

            if tile_zoom == cache_zoom:
                tiles_and_slices = [
                    (mercantile.Tile(x=x, y=y, z=tile_zoom), (slice(None, None), slice(None, None))),
                ]

            elif tile_zoom < cache_zoom:
                div = pow(2, cache_zoom - tile_zoom)
                tiles_and_slices = [
                    (mercantile.Tile(x=x//div, y=y//div, z=tile_zoom), (slice(None, None), slice(None, None))),
                ]

            else:
                fac = pow(2, tile_zoom - cache_zoom)
                sw = sh = data_resolution // fac
                tiles_and_slices = []
                for sy in range(fac):
                    for sx in range(fac):
                        tiles_and_slices.append((
                            mercantile.Tile(x=x * fac + sx, y=y * fac + sy, z=tile_zoom),
                            (slice(sy * sh, (sy + 1) * sh), slice(sx * sw, (sx + 1) * sw)),
                         ))

            do_it = False
            for tile, (slice_y, slice_x) in tiles_and_slices:
                if not pathconfig.output_tile_exists(tile.z, tile.x, tile.y, modality=modality):
                    do_it = True
                    break

            if do_it:
                if data is None:
                    try:
                        data = pathconfig.load_tile_cache_file(cache_zoom, x, y, modality=modality)
                    except Exception as e:
                        warnings.warn(f"{type(e).__name__}: {e}: {pathconfig.tile_cache_filename(cache_zoom, x, y, modality=modality)}")
                        continue
                #print("DATA", np.isnan(data).mean(), data.min(), data.max())
                for tile, (slice_y, slice_x) in tiles_and_slices:
                    #print(tile, slice_y, slice_x)
                    data_slice = data[slice_y, slice_x]
                    #print("SLICE", np.isnan(data_slice).mean(), data_slice.min(), data_slice.max())
                    if data_slice.shape != (resolution, resolution):
                        data_slice = cv2.resize(
                            data_slice,
                            (resolution, resolution),
                            interpolation,
                        )
                    yield source_tile, tile, data_slice
    
    for source_tile, tile, array in _iter_tiles(resolution):
        if pathconfig.output_tile_exists(tile.z, tile.x, tile.y, modality=modality):
            continue

        array2d = array
        if array2d.ndim == 3:
            array2d = array2d[..., 0]
        nan_mask = np.isnan(array2d) | (array2d <= -10_000)

        progress.set_postfix({
            "tile": f"{source_tile.z}/{source_tile.x}/{source_tile.y}->{tile.z}/{tile.x}/{tile.y}",
            "filled": f"{round(float((1.-nan_mask.mean())*100), 1)}%",
        })

        if nan_mask.ndim == 3:
            nan_mask = nan_mask[..., 0]
        array[nan_mask] = -1 if modality == "normal" else 0
    
        if modality == "normal":
            array = array * .5 + .5
            array = np.pad(array, ((0, 0), (0, 0), (0, 1)))
        else:
            array /= 2000  # TODO: get max height in dataset
            array = array[..., None].repeat(4, -1)

        array[..., 3] = 1. - nan_mask

        pathconfig.save_output_tile(tile.z, tile.x, tile.y, array, modality=modality)

