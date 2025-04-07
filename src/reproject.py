import math
import os
import shutil
import warnings
from typing import List, Tuple

from tqdm import tqdm
import rasterio
import rasterio.warp
import rasterio.windows
import mercantile
import numpy as np
import cv2

from .opendtm import OpenDTM
from . import config
from .files import DeleteFileOnException, PathConfig

# opendem's opendtm sectors are in
src_crs = rasterio.crs.CRS.from_epsg(25832)
# mercantile wants 4326
crs_4326 = rasterio.crs.CRS.from_epsg(4326)
# and outputs 3857
crs_3857 = rasterio.crs.CRS.from_epsg(3857)


def command_show_resolution(**kwargs):

    for zoom in range(1, 21):
        widths, heights = [], []
        for sx in (
                min(s[0] for s in OpenDTM.AVAILABLE_SECTORS),
                max(s[0] for s in OpenDTM.AVAILABLE_SECTORS),
        ):
            for sy in (
                    min(s[1] for s in OpenDTM.AVAILABLE_SECTORS),
                    max(s[1] for s in OpenDTM.AVAILABLE_SECTORS),
            ):
                bounds_4326 = rasterio.warp.transform_bounds(
                    src_crs, crs_4326,
                    sx * 1000, sy * 1000, (sx + 40) * 1000, (sy + 40) * 1000,
                )
                tile = next(mercantile.tiles(*bounds_4326, zooms=zoom))
                tile_bounds_3857 = mercantile.xy_bounds(*tile)
                tile_bounds = rasterio.warp.transform_bounds(crs_3857, src_crs, *tile_bounds_3857)
                widths.append(int(tile_bounds[2] - tile_bounds[0]))
                heights.append(int(tile_bounds[3] - tile_bounds[1]))

        min_str = f"{min(widths):10,} x {min(heights):,}"
        max_str = f"{max(widths):10,} x {max(heights):,}"
        print(f"zoom {zoom:2}: {min_str:23} - {max_str}")


def transform_coord(src_crs: rasterio.crs.CRS, dst_crs: rasterio.crs.CRS, x: float, y: float) -> Tuple[float, float]:
    x, y = rasterio.warp.transform(src_crs, dst_crs, [x], [y])
    return x[0], y[0]


def command_reproject(
        pathconfig: PathConfig,
        sectors: List[Tuple[int, int]],
        zoom: int,
        resolution: int,
        reset: bool,
        verbose: bool,
):
    dtm = OpenDTM(pathconfig=pathconfig, verbose=verbose)

    available_sectors = dtm.available_sectors(sectors)
    if not available_sectors:
        warnings.warn("No sectors found in cache")

    if reset:
        path = pathconfig.tile_cache_path() / str(zoom)
        if path.exists():
            shutil.rmtree(path)

    for sector in tqdm(available_sectors, desc="sectors", disable=not verbose):
        with dtm.open_sector(sector) as ds:
            bounds_4326 = rasterio.warp.transform_bounds(src_crs, crs_4326, *ds.bounds)
            tiles = list(mercantile.tiles(*bounds_4326, zooms=zoom))
            if pathconfig.tile_range_x:
                tiles = [tile for tile in tiles if pathconfig.tile_range_x[0] <= tile.x <= pathconfig.tile_range_x[1]]
            if pathconfig.tile_range_y:
                tiles = [tile for tile in tiles if pathconfig.tile_range_y[0] <= tile.y <= pathconfig.tile_range_y[1]]
            if not tiles:
                continue

            transformer = rasterio.transform.AffineTransformer(
                ds.transform
                # the sectors seem to be a little too small??
                #* rasterio.Affine.scale(40_000/39_993)
            )
            for tile in tqdm(tiles, position=1, desc="tiles", disable=not verbose):
                tile_bounds_3857 = mercantile.xy_bounds(*tile)
                bl = transform_coord(crs_3857, src_crs, tile_bounds_3857[0], tile_bounds_3857[1])
                br = transform_coord(crs_3857, src_crs, tile_bounds_3857[2], tile_bounds_3857[1])
                tl = transform_coord(crs_3857, src_crs, tile_bounds_3857[0], tile_bounds_3857[3])
                tr = transform_coord(crs_3857, src_crs, tile_bounds_3857[2], tile_bounds_3857[3])

                ptl, pbl, pbr, ptr = (transformer.rowcol(*c)[::-1] for c in (bl, tl, tr, br))
                p_extent = (
                    min(p[0] for p in (pbr, ptr, ptl, pbl)),
                    min(p[1] for p in (pbr, ptr, ptl, pbl)),
                    max(p[0] for p in (pbr, ptr, ptl, pbl)),
                    max(p[1] for p in (pbr, ptr, ptl, pbl)),
                )
                window = rasterio.windows.Window(col_off=p_extent[0], row_off=p_extent[1], width=p_extent[2]-p_extent[0], height=p_extent[3]-p_extent[1]+1)
                data = ds.read(1, window=window, boundless=True, fill_value=np.nan)

                vmask = ~np.isnan(data) & (data != -32768)
                if np.all(~vmask):
                    continue

                data[~vmask] = np.nan

                l, b, r, t = p_extent
                src = np.float32([[pbl[0]-l, pbl[1]-b], [pbr[0]-l, pbr[1]-b], [ptl[0]-l, ptl[1]-b], [ptr[0]-l, ptr[1]-b]])
                dst = np.float32([[0, 0], [r - l, 0], [0, t - b + 1], [r - l + 1, t - b]])
                src *= [[data.shape[1] / window.width, data.shape[0] / window.height]]
                dst *= [[data.shape[1] / window.width, data.shape[0] / window.height]]
                # try to fix the edges
                src = np.float32([
                    [math.ceil(src[0][0]), math.ceil(src[0][1])],
                    [math.floor(src[1][0]), math.ceil(src[1][1])],
                    [math.ceil(src[2][0]), math.floor(src[2][1])],
                    [math.floor(src[3][0]), math.floor(src[3][1])],
                ])

                mat = cv2.getPerspectiveTransform(src=src, dst=dst)
                data = cv2.warpPerspective(
                    data, mat, (data.shape[1], data.shape[0]),
                    flags=cv2.INTER_LINEAR,
                )

                data = cv2.resize(data, (resolution, resolution), cv2.INTER_CUBIC)

                sample_tile(pathconfig, tile, data)


def sample_tile(pathconfig: PathConfig, tile: mercantile.Tile, array: np.ndarray):
    if not pathconfig.tile_cache_file_exists(tile.z, tile.x, tile.y):
        sampler = array
    else:
        sampler = pathconfig.load_tile_cache_file(tile.z, tile.x, tile.y)
        if sampler.shape != array.shape:
            raise ValueError(
                f"The reprojection samplers have shape {sampler.shape} and reprojected"
                f" tiles have shape {array.shape}. Use --reset to delete the previous samplers"
            )

        vmask = ~np.isnan(array)
        sampler[vmask] = array[vmask]

    pathconfig.save_tile_cache_file(tile.z, tile.x, tile.y, sampler)

