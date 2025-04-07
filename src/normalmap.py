from typing import List, Tuple, Optional

from tqdm import tqdm
import numpy as np
import cv2

from .files import PathConfig, MemoryCache


class NormalMapper:

    def __init__(
            self,
            pathconfig: PathConfig,
            zoom: int,
            edge_cache_size: int,
            tile_cache_size: int,
            approximate: bool,
            eps: float = 0.000001,
    ):
        self.pathconfig = pathconfig
        self.zoom = zoom
        self.edge_cache = MemoryCache(max_items=edge_cache_size)
        self.tile_cache = MemoryCache(max_items=tile_cache_size)
        self.eps = eps
        self.approximate = approximate
        self.num_edges_approximated = 0

    def cache_edges(self, x: int, y: int, data: np.ndarray):
        for name, edge in (
                ("left", data[:, :1]),
                ("right", data[:, -1:]),
                ("bottom", data[:1]),
                ("top", data[-1:]),
        ):
            self.edge_cache.put((x, y, name), edge)

    def get_tile(self, x: int, y: int) -> Optional[np.ndarray]:
        tile = self.tile_cache.get((x, y))
        if tile is False:
            return None
        if tile is None:
            if not self.pathconfig.tile_cache_file_exists(self.zoom, x, y):
                tile = False
            else:
                tile = self.pathconfig.load_tile_cache_file(self.zoom, x, y)
                self.cache_edges(x, y, tile)
            self.tile_cache.put((x, y), tile)
        return None if tile is False else tile

    def get_edge(self, x: int, y: int, name: str) -> Optional[np.ndarray]:
        if self.approximate:
            return None
        edge = self.edge_cache.get((x, y, name))
        if edge is False:
            edge = None
        elif edge is None:
            tile = self.get_tile(x, y)
            if tile is None:
                self.edge_cache.put((x, y, name), False)
                edge = None
            else:
                edge = self.edge_cache.get((x, y, name))

        #if edge is None and not approximate:
        #    warnings.warn(f"Approximating {name} edge of tile {zoom}/{x}/{y}")
        return edge

    def normal_map(self, x: int, y: int) -> np.ndarray:

        tile = self.get_tile(x, y)

        left = self.get_edge(x - 1, y, "right")
        right = self.get_edge(x + 1, y, "left")
        bottom = self.get_edge(x, y - 1, "top")
        top = self.get_edge(x, y + 1, "bottom")

        if left is None:
            self.num_edges_approximated += 1
            left = tile[:, :1] * 2 - tile[:, 1:2]
        if right is None:
            self.num_edges_approximated += 1
            right = tile[:, -1:] * 2 - tile[:, -2:-1]
        if bottom is None:
            self.num_edges_approximated += 1
            bottom = tile[:1] * 2 - tile[1:2]
        if top is None:
            self.num_edges_approximated += 1
            top = tile[-1:] * 2 - tile[-2:-1]

        bottom = np.pad(bottom, ((0, 0), (1, 1)))
        top = np.pad(top, ((0, 0), (1, 1)))

        tile = np.concat([left, tile, right], 1)
        tile = np.concat([bottom, tile, top], 0)

        # TODO this does not account for window in DTM sector
        #   z_factor = 2 * tile.shape[0] / 40_000
        #   so currently assume 1:1 reprojection
        z_factor = 2

        tile = np.concat(
            [
                (tile[2:,   1:-1] - tile[ :-2, 1:-1])[..., None],
                (tile[1:-1,  :-2] - tile[1:-1, 2:  ])[..., None],
                np.ones((tile.shape[0] - 2, tile.shape[1] - 2, 1)) * z_factor,
            ],
            axis=-1,
        )
        tile /= np.linalg.norm(tile, axis=2, keepdims=True) + self.eps

        return tile

    def stats(self) -> dict:
        return {
            "edge_hits/misses": f"{self.edge_cache.num_hits}/{self.edge_cache.num_misses}",
            "tile_hits/misses": f"{self.tile_cache.num_hits}/{self.tile_cache.num_misses}",
            "approxed_edges": self.num_edges_approximated,
        }

