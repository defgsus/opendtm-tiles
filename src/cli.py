import argparse
from typing import List, Tuple

from tqdm import tqdm

from src import config
from src.opendtm import OpenDTM
from src.files import PathConfig
from src.reproject import command_reproject, command_show_resolution
from src.preview import command_preview
from src.normalmap import command_normal_map
from src.rendertiles import command_render
from src.downsample import command_downsample


def parse_args() -> dict:
    main_parser = argparse.ArgumentParser()

    pathconfig = PathConfig()

    main_parser.add_argument("-q", "--quite", type=bool, nargs="?", default=False, const=True)

    main_parser.add_argument("-wc", "--web-cache-path", type=str, default=pathconfig.web_cache_path)
    main_parser.add_argument("-tc", "--tile-cache-path", type=str, default=pathconfig.tile_cache_path(None))

    subparsers = main_parser.add_subparsers()

    def _add_sector_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-x", "--sector-x", type=int, nargs="+", default=config.OPENDTM_SECTOR_X,
            help=f"Sector west->east extent to consider, default is {config.OPENDTM_SECTOR_X}",
        )
        parser.add_argument(
            "-y", "--sector-y", type=int, nargs="+", default=config.OPENDTM_SECTOR_Y,
            help=f"Sector north->south extent to consider, default is {config.OPENDTM_SECTOR_Y}",
        )

    def _add_tile_args(parser: argparse.ArgumentParser, prefix: str = ""):
        parser.add_argument(
            f"-{prefix}x", "--tile-x", type=int, nargs="+", default=None,
            help=f"Tile x extent to consider, two numbers for a range",
        )
        parser.add_argument(
            f"-{prefix}y", "--tile-y", type=int, nargs="+", default=None,
            help=f"Tile y extent to consider, two numbers for a range",
        )

    def _add_random_order(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-ro", "--random-order", type=bool, nargs="?", default=False, const=True,
            help="Randomize order of processing the tile files",
        )

    def _add_modality(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-m", "--modality", type=str, default="height",
            choices=("height", "normal"),
            help="Set the modality to preview, render or downsample",
        )

    parser = subparsers.add_parser("cache")
    parser.set_defaults(command="cache")
    _add_sector_args(parser)

    parser = subparsers.add_parser("reproject")
    parser.set_defaults(command="reproject")
    _add_sector_args(parser)
    _add_tile_args(parser, prefix="t")
    parser.add_argument("-r", "--resolution", type=int, default=256)
    parser.add_argument("-z", "--zoom", type=int, default=10)
    parser.add_argument(
        "-R", "--reset", type=bool, nargs="?", default=False, const=True,
        help="Delete the tile cache directory for that zoom level before sampling reprojections",
    )

    parser = subparsers.add_parser("normal-map")
    parser.set_defaults(command="normal_map")
    parser.add_argument("-z", "--zoom", type=int, default=10)
    _add_tile_args(parser)
    _add_random_order(parser)
    parser.add_argument("-tcs", "--tile-cache-size", type=int, default=1)
    parser.add_argument("-ecs", "--edge-cache-size", type=int, default=10_000)
    parser.add_argument(
        "-a", "--approximate", type=bool, nargs="?", default=False, const=True,
        help="Approximate edges instead of loading the tiles",
    )
    parser.add_argument(
        "-O", "--overwrite", type=bool, nargs="?", default=False, const=True,
        help="Overwrite existing normal map tiles",
    )
    parser.add_argument("-j", "--workers", type=int, default=1)

    parser = subparsers.add_parser(
        "preview",
        help="Preview the reprojected tiles (or normal-maps) by rendering them into one image",
    )
    parser.set_defaults(command="preview")
    _add_modality(parser)
    parser.add_argument("-z", "--zoom", type=int, default=10)
    parser.add_argument("-r", "--resolution", type=int, default=None, help="resolution per tile in preview")
    parser.add_argument("-p", "--padding", type=int, default=1, help="padding between tiles")
    parser.add_argument(
        "-n", "--numbers", type=bool, nargs="?", default=False, const=True,
        help="Just print the numbers of the grid",
    )
    _add_tile_args(parser)

    parser = subparsers.add_parser(
        "render",
        help="Render png images from the reprojected tiles"
    )
    parser.set_defaults(command="render")
    _add_modality(parser)
    parser.add_argument("-cz", "--cache-zoom", type=int, default=10)
    parser.add_argument("-tz", "--tile-zoom", type=int, default=10)
    parser.add_argument("-r", "--resolution", type=int, default=None, help="resolution per tile in tile-zoom")
    parser.add_argument("-j", "--workers", type=int, default=1)
    _add_random_order(parser)

    parser = subparsers.add_parser("downsample")
    parser.set_defaults(command="downsample")
    _add_modality(parser)
    parser.add_argument(
        "-z", "--zoom", type=int, nargs="+", default=[10],
        help="Zoom level to downsample, two numbers to set range, e.g. 17 1 for all level starting at 17"
    )
    parser.add_argument("-j", "--workers", type=int, default=1)
    _add_random_order(parser)

    parser = subparsers.add_parser("show-paths")
    parser.set_defaults(command="show_paths")

    parser = subparsers.add_parser("show-resolution")
    parser.set_defaults(command="show_resolution")

    kwargs = vars(main_parser.parse_args())

    kwargs["verbose"] = not kwargs.pop("quite")

    if "sector_x" in kwargs:
        sector_x = kwargs.pop("sector_x")
        sector_y = kwargs.pop("sector_y")
        if len(sector_x) == 1:
            sector_x *= 2
        elif len(sector_x) != 2:
            raise ValueError(f"--sector-x should be one or two numbers")
        if len(sector_y) == 1:
            sector_y *= 2
        elif len(sector_y) != 2:
            raise ValueError(f"--sector-y should be one or two numbers")
        kwargs["sectors"] = [
            (x, y)
            for x in range(sector_x[0], sector_x[1] + 1, 40)
            for y in range(sector_y[0], sector_y[1] + 1, 40)
            if (x, y) in OpenDTM.AVAILABLE_SECTORS
        ]

    tile_x = None
    tile_y = None
    if "tile_x" in kwargs:
        tile_x = kwargs.pop("tile_x")
        if tile_x:
            tile_x = tuple(tile_x)
            if len(tile_x) == 1:
                tile_x *= 2
            elif len(tile_x) != 2:
                raise ValueError(f"--tile-x should be one or two numbers")
    if "tile_y" in kwargs:
        tile_y = kwargs.pop("tile_y")
        if tile_y:
            tile_y = tuple(tile_y)
            if len(tile_y) == 1:
                tile_y *= 2
            elif len(tile_y) != 2:
                raise ValueError(f"--tile-y should be one or two numbers")


    kwargs["pathconfig"] = PathConfig(
        web_cache_path=kwargs.pop("web_cache_path"),
        tile_cache_path=kwargs.pop("tile_cache_path"),
        random_order=kwargs.pop("random_order") if "random_order" in kwargs else False,
        tile_x=tile_x,
        tile_y=tile_y,
    )

    return kwargs


def main(command: str, **kwargs):
    func = globals().get(f"command_{command}", None)
    if func is None:
        raise ValueError(f"Unknown command '{command}'")

    func(**kwargs)


def command_cache(
        pathconfig: PathConfig,
        sectors: List[Tuple[int, int]],
        verbose: bool,
):
    dtm = OpenDTM(verbose=verbose, pathconfig=pathconfig)
    for sector in tqdm(sectors, desc="sectors", disable=not verbose):
        dtm.download_sector(sector)
        dtm.extract_sector(sector)


def command_show_paths(pathconfig: PathConfig, **kwargs):
    print(f"web-cache:  {pathconfig.web_cache_path}")
    print(f"tile-cache: {pathconfig.tile_cache_path(modality="height")}")
    print(f"            {pathconfig.tile_cache_path(modality="normal")}")


if __name__ == "__main__":
    main(**parse_args())
