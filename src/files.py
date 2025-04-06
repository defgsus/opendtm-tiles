import os
import random
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, Hashable, Any, List

import numpy as np
import PIL.Image

from . import config


class PathConfig:

    def __init__(self, **kwargs):
        self.web_cache_path = Path(kwargs.get("web_cache_path", config.OPENDTM_WEB_CACHE_PATH))
        self._tile_cache_path = Path(kwargs.get("tile_cache_path", config.OPENDTM_TILE_CACHE_PATH))
        self._tile_output_path = Path(kwargs.get("tile_output_path", config.OPENDTM_TILE_OUTPUT_PATH))
        self.is_random_order = kwargs.get("random_order", False)
        self.tile_range_x: Optional[Tuple[int, int]] = kwargs.get("tile_x")
        self.tile_range_y: Optional[Tuple[int, int]] = kwargs.get("tile_y")

    def web_cache_file(self, sector_x: int, sector_y: int, extension: str = ".tif"):
        return self.web_cache_path / f"{extension[1:]}/E{sector_x}N{sector_y}{extension}"

    def tile_cache_path(self, modality: Optional[str] = "height") -> Path:
        path = self._tile_cache_path
        if modality:
            path = path / modality
        return path

    def tile_output_path(self, modality: str = "height") -> Path:
        path = self._tile_output_path / modality
        return path

    def tile_cache_filename(self, z: int, x: int, y: int, modality: str = "height"):
        return self.tile_cache_path(modality=modality) / f"{z}/{x}/{y}.npz"

    def tile_cache_file_map(self, zoom: int, modality: str = "height"):
        path = self.tile_cache_path(modality=modality)
        tile_map = get_tile_file_map(path, zoom, ".npz", tile_range_x=self.tile_range_x, tile_range_y=self.tile_range_y)
        if self.is_random_order:
            tile_map = randomize_tile_file_map(tile_map)
        return tile_map

    def tile_output_filename(self, z: int, x: int, y: int, modality: str = "height"):
        return self.tile_output_path(modality=modality) / f"{z}/{x}/{y}.png"

    def tile_cache_file_exists(self, z: int, x: int, y: int, modality: str = "height") -> bool:
        return self.tile_cache_filename(z, x, y, modality=modality).exists()

    def load_tile_cache_file(self, z: int, x: int, y: int, modality: str = "height") -> np.ndarray:
        filename = self.tile_cache_filename(z, x, y, modality=modality)
        return np.load(filename).get("arr_0")

    def output_tile_exists(self, z: int, x: int, y: int, modality: str = "height") -> bool:
        return self.tile_output_filename(z, x, y, modality=modality).exists()

    def tile_output_file_map(self, zoom: int, modality: str = "height"):
        path = self.tile_output_path(modality=modality)
        tile_map = get_tile_file_map(path, zoom, ".png", tile_range_x=self.tile_range_x, tile_range_y=self.tile_range_y)
        if self.is_random_order:
            tile_map = randomize_tile_file_map(tile_map)
        return tile_map

    def save_tile_cache_file(self, z: int, x: int, y: int, array: np.ndarray, modality: str = "height"):
        filename = self.tile_cache_filename(z, x, y, modality=modality)
        os.makedirs(filename.parent, exist_ok=True)
        with DeleteFileOnException(filename):
            np.savez_compressed(filename, array)

    def save_output_tile(self, z: int, x: int, y: int, array: Union[np.ndarray, PIL.Image.Image], modality: str = "height"):
        if isinstance(array, PIL.Image.Image):
            image = array
        else:
            image = PIL.Image.fromarray(
                (array * 255).clip(0, 255).astype(np.uint8)
            )
        filename = self.tile_output_filename(z, x, y, modality=modality)
        os.makedirs(filename.parent, exist_ok=True)
        with DeleteFileOnException(filename):
            image.save(filename)

    def load_tile_output_file(self, z: int, x: int, y: int, modality: str = "height") -> PIL.Image.Image:
        filename = self.tile_output_filename(z, x, y, modality=modality)
        return PIL.Image.open(filename)


class DeleteFileOnException:

    def __init__(self, filename: Union[str, Path]):
        self.filename = Path(filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            if self.filename.exists():
                os.remove(self.filename)


def get_tile_file_map(
        base_path: Union[str, Path],
        zoom: int,
        extension: str = ".png",
        tile_range_x: Optional[Tuple[int, int]] = None,
        tile_range_y: Optional[Tuple[int, int]] = None,
) -> Dict[Tuple[int, int], Path]:
    dic = {}
    for file in sorted((Path(base_path) / str(zoom)).rglob(f"**/*{extension}")):
        x, y = int(file.parent.name), int(file.with_suffix("").name)
        if tile_range_x and not tile_range_x[0] <= x <= tile_range_x[1]:
            continue
        if tile_range_y and not tile_range_y[0] <= y <= tile_range_y[1]:
            continue

        dic[(x, y)] = file

    return dic

def split_tile_file_map(
        tile_map: Dict[Tuple[int, int], Any],
        workers: int,
) -> List[Dict[Tuple[int, int], Any]]:
    keys = list(tile_map.keys())
    key_batches = []
    batch_size = len(keys) // workers
    for i in range(workers):
        key_batches.append(keys[i * batch_size: (i + 1) * batch_size])

    return [
        {key: tile_map[key] for key in key_batch}
        for key_batch in key_batches
    ]


def randomize_tile_file_map(
        tile_map: Dict[Tuple[int, int], Path],
):
    keys = list(tile_map.keys())
    random.shuffle(keys)
    return {key: tile_map[key] for key in keys}


class MemoryCache:

    def __init__(self, max_items: int):
        self._max_items = max_items
        self._cache = {}
        self._ever_seen = set()
        self._age = 0
        self.num_hits = 0
        self.num_misses = 0

    def has(self, key: Hashable) -> bool:
        return key in self._cache

    def get(self, key: Hashable) -> Optional[Any]:
        self._age += 1
        if key not in self._cache:
            if key in self._ever_seen:
                self.num_misses += 1
            return None
        self.num_hits += 1
        cache = self._cache[key]
        cache["age"] = self._age
        return cache["value"]

    def put(self, key: Hashable, value: Any):
        self._cache[key] = {
            "age": self._age,
            "value": value,
        }
        self._ever_seen.add(key)

        if len(self._cache) > self._max_items:
            keys = sorted(self._cache, key=lambda k: self._cache[k]["age"])
            keys = keys[:len(self._cache) - self._max_items]
            for key in keys:
                del self._cache[key]
