import argparse
from typing import List, Tuple

from tqdm import tqdm

from src import config
from src.opendtm import OpenDTM
from src.files import PathConfig
from src.reproject import command_reproject, command_show_resolution
from src.preview import command_reproject_preview, command_normal_map_preview
from src.normalmap import command_normal_map


def parse_args() -> dict:
    main_parser = argparse.ArgumentParser()

    pathconfig = PathConfig()

    main_parser.add_argument("-q", "--quite", type=bool, nargs="?", default=False, const=True)

    main_parser.add_argument("-wc", "--web-cache-path", type=str, default=pathconfig.web_cache_path)
    main_parser.add_argument("-tc", "--tile-cache-path", type=str, default=pathconfig.tile_cache_path())

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

    parser = subparsers.add_parser("cache")
    parser.set_defaults(command="cache")
    _add_sector_args(parser)

    parser = subparsers.add_parser("reproject")
    parser.set_defaults(command="reproject")
    _add_sector_args(parser)
    parser.add_argument("-r", "--resolution", type=int, default=256)
    parser.add_argument("-z", "--zoom", type=int, default=10)
    parser.add_argument(
        "-R", "--reset", type=bool, nargs="?", default=False, const=True,
        help="Delete the tile cache directory for that zoom level before sampling reprojections",
    )

    parser = subparsers.add_parser("normal-map")
    parser.set_defaults(command="normal_map")
    parser.add_argument("-z", "--zoom", type=int, default=10)
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

    for preview_type in ("reproject", "normal_map"):
        parser = subparsers.add_parser(f"{preview_type.replace('_', '-')}-preview")
        parser.set_defaults(command=f"{preview_type}_preview")
        parser.add_argument("-z", "--zoom", type=int, default=10)
        parser.add_argument("-r", "--resolution", type=int, default=None, help="resolution per tile in preview")
        parser.add_argument("-p", "--padding", type=int, default=1, help="padding between tiles")

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

    kwargs["pathconfig"] = PathConfig(
        web_cache_path=kwargs.pop("web_cache_path"),
        tile_cache_path=kwargs.pop("tile_cache_path"),
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
    print(f"tile-cache: {pathconfig.tile_cache_path(normal=False)}")
    print(f"            {pathconfig.tile_cache_path(normal=True)}")


if __name__ == "__main__":
    main(**parse_args())
