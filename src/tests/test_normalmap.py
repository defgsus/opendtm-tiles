import unittest
from pathlib import Path

import numpy as np

from src.files import PathConfig
from src.normalmap import command_normal_map


class MockPathConfig(PathConfig):

    def __init__(self):
        super().__init__()
        self._mock_cache = {}
        self._mock_tiles = {}
        self.statements = []

    def dump_statements(self):
        for s, kwargs in self.statements:
            print(f"(\"{s}\", {kwargs}),")
            #kwargs = ", ".join(f"{key}:{value}" for key, value in kwargs.items())
            #print(f"{s}({kwargs})")

    def tile_cache_file_map(self, zoom: int, modality: str = "height"):
        return {
            (x, y): Path(f"{modality}/{z}/{y}/{x}.npz")
            for mod, z, x, y in self._mock_cache
            if mod == modality and z == zoom
        }

    def tile_output_file_map(self, zoom: int, modality: str = "height"):
        return {
            (x, y): Path(f"{modality}/{z}/{y}/{x}.npz")
            for mod, z, x, y in self._mock_tiles
            if mod == modality and z == zoom
        }

    def tile_cache_file_exists(self, z: int, x: int, y: int, modality: str = "height") -> bool:
        return (modality, z, x, y) in self._mock_cache

    def save_tile_cache_file(self, z: int, x: int, y: int, array: np.ndarray, modality: str = "height"):
        self.statements.append(("save_tile_cache_file", {"z": z, "x": x, "y": y, "array": array.shape, "modality": "height"}))
        self._mock_cache[(modality, z, x, y)] = array.copy()

    def load_tile_cache_file(self, z: int, x: int, y: int, modality: str = "height") -> np.ndarray:
        self.statements.append(("load_tile_cache_file", {"z": z, "x": x, "y": y, "modality": "height"}))
        return self._mock_cache[(modality, z, x, y)]

    def tile_output_exists(self, z: int, x: int, y: int, modality: str = "height") -> bool:
        return (modality, z, x, y) in self._mock_tiles

    def save_output_tile(self, z: int, x: int, y: int, array: np.ndarray, modality: str = "height"):
        self.statements.append(("save_output_tile", {"z": z, "x": x, "y": y, "array": array.shape, "modality": "height"}))
        self._mock_tiles[(modality, z, x, y)] = array.copy()

    def load_tile_output_file(self, z: int, x: int, y: int, modality: str = "height") -> np.ndarray:
        self.statements.append(("load_tile_output_file", {"z": z, "x": x, "y": y, "modality": "height"}))
        return self._mock_tiles[(modality, z, x, y)]


class TestNormalMapping(unittest.TestCase):

    def test_100_normal(self):
        zoom = 16
        res = 32
        pathconfig = MockPathConfig()
        for x in range(3):
            for y in range(3):
                pathconfig.save_tile_cache_file(zoom, 1000+x, 1000+y, np.zeros((res, res)))
        pathconfig.statements.clear()

        stats = command_normal_map(
            pathconfig=pathconfig,
            zoom=16,
            edge_cache_size=1000,
            tile_cache_size=1000,
            approximate=False,
            overwrite=False,
            workers=1,
            verbose=False,
        )
        # pathconfig.dump_statements()

        self.assertEqual(
            [
                # first tile is 1000, 1000
                ("load_tile_cache_file", {'z': 16, 'x': 1000, 'y': 1000, 'modality': 'height'}),
                # get edges of neighbours
                ("load_tile_cache_file", {'z': 16, 'x': 1001, 'y': 1000, 'modality': 'height'}),
                ("load_tile_cache_file", {'z': 16, 'x': 1000, 'y': 1001, 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1000, 'y': 1000, 'array': (32, 32, 3), 'modality': 'height'}),
                # next tile 1000, 1001, load neighbours
                ("load_tile_cache_file", {'z': 16, 'x': 1001, 'y': 1001, 'modality': 'height'}),
                ("load_tile_cache_file", {'z': 16, 'x': 1000, 'y': 1002, 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1000, 'y': 1001, 'array': (32, 32, 3), 'modality': 'height'}),
                # tile 1002, 1002, load neighbour (others are in cache)
                ("load_tile_cache_file", {'z': 16, 'x': 1001, 'y': 1002, 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1000, 'y': 1002, 'array': (32, 32, 3), 'modality': 'height'}),
                # aso...
                ("load_tile_cache_file", {'z': 16, 'x': 1002, 'y': 1000, 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1001, 'y': 1000, 'array': (32, 32, 3), 'modality': 'height'}),
                ("load_tile_cache_file", {'z': 16, 'x': 1002, 'y': 1001, 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1001, 'y': 1001, 'array': (32, 32, 3), 'modality': 'height'}),
                ("load_tile_cache_file", {'z': 16, 'x': 1002, 'y': 1002, 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1001, 'y': 1002, 'array': (32, 32, 3), 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1002, 'y': 1000, 'array': (32, 32, 3), 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1002, 'y': 1001, 'array': (32, 32, 3), 'modality': 'height'}),
                ("save_tile_cache_file", {'z': 16, 'x': 1002, 'y': 1002, 'array': (32, 32, 3), 'modality': 'height'}),
            ],
            pathconfig.statements,
        )
        self.assertEqual(
            {'skipped': 0, 'edge_hits': 24, 'edge_misses': 0, 'tile_hits': 8, 'tile_misses': 0, 'approximated_edges': 12},
            stats,
        )
