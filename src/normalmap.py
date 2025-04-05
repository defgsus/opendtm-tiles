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
        verbose: bool,
        edge_cache_size: int = 10_000,
):
    tiles_map = pathconfig.tile_cache_file_map(zoom=zoom)

    edge_cache = MemoryCache(max_items=edge_cache_size)
    tile_cache = MemoryCache(max_items=10)

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
            return None
        if edge is None:
            tile = _get_tile(x, y)
            if tile is None:
                edge_cache.put((x, y, name), False)
                return None
        return edge

    progress = tqdm(tiles_map.items(), desc="tiles", disable=not verbose)
    for (x, y), filename in progress:
        data = _get_tile(x, y)

        left = _get_edge(x - 1, y, "right")
        right = _get_edge(x + 1, y, "left")
        bottom = _get_edge(x, y - 1, "top")
        top = _get_edge(x, y + 1, "bottom")

        progress.set_postfix({
            "edge_hits/misses": f"{edge_cache.num_hits}/{edge_cache.num_misses}",
            "tile hits/misses": f"{tile_cache.num_hits}/{tile_cache.num_misses}"
        })




