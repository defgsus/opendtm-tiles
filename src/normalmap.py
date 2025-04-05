import os
import warnings
from typing import List, Tuple, Optional

from tqdm import tqdm
import numpy as np
import cv2

from . import config
from .files import DeleteFileOnException, get_tile_file_map, PathConfig, MemoryCache


def command_normal_map(
        pathconfig: PathConfig,
        zoom: int,
        edge_cache_size: int,
        tile_cache_size: int,
        approximate: bool,
        overwrite: bool,
        verbose: bool,
        eps: float = 0.000001,
):
    tiles_map = pathconfig.tile_cache_file_map(zoom=zoom)

    edge_cache = MemoryCache(max_items=edge_cache_size)
    tile_cache = MemoryCache(max_items=tile_cache_size)

    def _cache_edges(x: int, y: int, data: np.ndarray):
        for name, edge in (
                ("left", data[:, :1]),
                ("right", data[:, -1:]),
                ("bottom", data[:1]),
                ("top", data[-1:]),
        ):
            edge_cache.put((x, y, name), edge)

    def _get_tile(x: int, y: int) -> Optional[np.ndarray]:
        tile = tile_cache.get((x, y))
        if tile is False:
            return None
        if tile is None:
            fn = pathconfig.tile_cache_filename(zoom, x, y)
            if not fn.exists():
                tile = False
            else:
                tile = np.load(fn).get("arr_0")
                _cache_edges(x, y, tile)
            tile_cache.put((x, y), tile)
        return None if tile is False else tile

    def _get_edge(x: int, y: int, name: str) -> Optional[np.ndarray]:
        edge = edge_cache.get((x, y, name))
        if edge is False:
            edge = None
        elif edge is None:
            tile = _get_tile(x, y)
            if tile is None:
                edge_cache.put((x, y, name), False)
                edge = None

        if edge is None and not approximate:
            warnings.warn(f"Approximating {name} edge of tile {zoom}/{x}/{y}")
        return edge

    num_skipped = 0
    progress = tqdm(tiles_map.items(), desc="tiles", disable=not verbose)
    for (x, y), filename in progress:
        normal_filename = pathconfig.tile_cache_filename(zoom, x, y, normal=True)

        if normal_filename.exists() and not overwrite:
            num_skipped += 1

        else:
            tile = _get_tile(x, y)

            left = _get_edge(x - 1, y, "right")
            right = _get_edge(x + 1, y, "left")
            bottom = _get_edge(x, y - 1, "top")
            top = _get_edge(x, y + 1, "bottom")

            if left is None:
                left = tile[:, :1] * 2 - tile[:, 1:2]
            if right is None:
                right = tile[:, -1:] * 2 - tile[:, -2:-1]
            if bottom is None:
                bottom = tile[:1] * 2 - tile[1:2]
            if top is None:
                top = tile[-1:] * 2 - tile[-2:-1]

            bottom = np.pad(bottom, ((0, 0), (1, 1)))
            top = np.pad(top, ((0, 0), (1, 1)))

            tile = np.concat([left, tile, right], 1)
            tile = np.concat([bottom, tile, top], 0)

            z_factor = 2 * tile.shape[0] / 40_000
            tile = np.concat(
                [
                    (tile[2:,   1:-1] - tile[ :-2, 1:-1])[..., None],
                    (tile[1:-1,  :-2] - tile[1:-1, 2:  ])[..., None],
                    np.ones((tile.shape[0] - 2, tile.shape[1] - 2, 1)) * z_factor,
                ],
                axis=-1,
            )
            tile /= np.linalg.norm(tile, axis=2, keepdims=True) + eps

            os.makedirs(normal_filename.parent, exist_ok=True)
            with DeleteFileOnException(normal_filename):
                np.savez_compressed(normal_filename, tile)

        progress.set_postfix({
            "skipped": num_skipped,
            "edge_hits/misses": f"{edge_cache.num_hits}/{edge_cache.num_misses}",
            "tile hits/misses": f"{tile_cache.num_hits}/{tile_cache.num_misses}",
        })
