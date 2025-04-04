import argparse
from typing import List, Tuple

from tqdm import tqdm

from src import config
from src.opendtm import OpenDTM
from src.reproject import command_reproject, command_show_resolution


def parse_args() -> dict:
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument("-q", "--quite", type=bool, nargs="?", default=False, const=True)


    subparsers = main_parser.add_subparsers()

    def _add_sector_args(parser: argparse.ArgumentParser):
        parser.add_argument("-x", "--sector-x", type=int, nargs="+", default=config.OPENDTM_SECTOR_X)
        parser.add_argument("-y", "--sector-y", type=int, nargs="+", default=config.OPENDTM_SECTOR_Y)

    parser = subparsers.add_parser("cache")
    parser.set_defaults(command="cache")
    _add_sector_args(parser)

    parser = subparsers.add_parser("show-resolution")
    parser.set_defaults(command="show_resolution")

    parser = subparsers.add_parser("reproject")
    parser.set_defaults(command="reproject")

    _add_sector_args(parser)
    parser.add_argument("-r", "--resolution", type=int, default=256)
    parser.add_argument("-z", "--zoom", type=int, default=12)

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

    return kwargs


def main(command: str, **kwargs):
    func = globals().get(f"command_{command}", None)
    if func is None:
        raise ValueError(f"Unknown command '{command}'")

    func(**kwargs)


def command_cache(
        sectors: List[Tuple[int, int]],
        verbose: bool,
):
    dtm = OpenDTM(verbose=verbose)
    for sector in tqdm(sectors, desc="sectors", disable=not verbose):
        dtm.download_sector(sector)
        dtm.extract_sector(sector)



if __name__ == "__main__":
    main(**parse_args())
