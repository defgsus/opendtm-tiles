import os
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, Hashable, Any

from . import config


class PathConfig:

    def __init__(self, **kwargs):
        self.web_cache_path = Path(kwargs.get("web_cache_path", config.OPENDTM_WEB_CACHE_PATH))
        self._tile_cache_path = Path(kwargs.get("tile_cache_path", config.OPENDTM_TILE_CACHE_PATH))

    def web_cache_file(self, sector_x: int, sector_y: int, extension: str = ".tif"):
        return self.web_cache_path / f"{extension[1:]}/E{sector_x}N{sector_y}{extension}"

    def tile_cache_path(self, normal: bool = False) -> Path:
        path = self._tile_cache_path
        if normal:
            path = path / "normal"
        return path

    def tile_cache_filename(self, z: int, x: int, y: int, normal: bool = False):
        return self.tile_cache_path(normal=normal) / f"{z}/{x}/{y}.npz"

    def tile_cache_file_map(self, zoom: int, normal: bool = False):
        path = self.tile_cache_path(normal=normal)
        return get_tile_file_map(path, zoom, ".npz")


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
) -> Dict[Tuple[int, int], Path]:
    return {
        (int(file.parent.name), int(file.with_suffix("").name)): file
        for file in sorted((Path(base_path) / str(zoom)).rglob(f"**/*{extension}"))
    }


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
