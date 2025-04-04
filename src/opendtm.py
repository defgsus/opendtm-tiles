import json
import math
import sys
import warnings
import zipfile
import os
from pathlib import Path
from typing import Tuple, Union, Optional, List, Generator

import decouple
import numpy as np
import rasterio
import rasterio.windows

from . import config
from ._download import streaming_download


class OpenDTM:
    """
    Data cache for
    https://www.opendem.info/opendtm_de_download.html

    There are 281 zipped GeoTiff files, each containing ~40,000 * 40,000 pixels.

    One zip file is 2 to 4gb, which totals at about 850gb!!
    """
    Sector = Tuple[int, int]

    # Note: some sectors are 40_001*40_001, or even 40_001*40_000, ...
    AVAILABLE_SECTORS = {(280, 5440), (280, 5480), (280, 5520), (280, 5560), (280, 5600), (280, 5640), (280, 5680), (280, 5720), (320, 5440), (320, 5480), (320, 5520), (320, 5560), (320, 5600), (320, 5640), (320, 5680), (320, 5720), (320, 5760), (320, 5800), (320, 5920), (360, 5240), (360, 5280), (360, 5320), (360, 5400), (360, 5440), (360, 5480), (360, 5520), (360, 5560), (360, 5600), (360, 5640), (360, 5680), (360, 5720), (360, 5760), (360, 5800), (360, 5840), (360, 5880), (360, 5920), (400, 5240), (400, 5280), (400, 5320), (400, 5360), (400, 5400), (400, 5440), (400, 5480), (400, 5520), (400, 5560), (400, 5600), (400, 5640), (400, 5680), (400, 5720), (400, 5760), (400, 5800), (400, 5840), (400, 5880), (400, 5920), (400, 5960), (400, 6000), (440, 5240), (440, 5280), (440, 5320), (440, 5360), (440, 5400), (440, 5440), (440, 5480), (440, 5520), (440, 5560), (440, 5600), (440, 5640), (440, 5680), (440, 5720), (440, 5760), (440, 5800), (440, 5840), (440, 5880), (440, 5920), (440, 5960), (440, 6000), (440, 6040), (480, 5240), (480, 5280), (480, 5320), (480, 5360), (480, 5400), (480, 5440), (480, 5480), (480, 5520), (480, 5560), (480, 5600), (480, 5640), (480, 5680), (480, 5720), (480, 5760), (480, 5800), (480, 5840), (480, 5880), (480, 5920), (480, 5960), (480, 6000), (480, 6040), (480, 6080), (520, 5240), (520, 5280), (520, 5320), (520, 5360), (520, 5400), (520, 5440), (520, 5480), (520, 5520), (520, 5560), (520, 5600), (520, 5640), (520, 5680), (520, 5720), (520, 5760), (520, 5800), (520, 5840), (520, 5880), (520, 5920), (520, 5960), (520, 6000), (520, 6040), (520, 6080), (560, 5200), (560, 5240), (560, 5280), (560, 5320), (560, 5360), (560, 5400), (560, 5440), (560, 5480), (560, 5520), (560, 5560), (560, 5600), (560, 5640), (560, 5680), (560, 5720), (560, 5760), (560, 5800), (560, 5840), (560, 5880), (560, 5920), (560, 5960), (560, 6000), (560, 6040), (600, 5240), (600, 5280), (600, 5320), (600, 5360), (600, 5400), (600, 5440), (600, 5480), (600, 5520), (600, 5560), (600, 5600), (600, 5640), (600, 5680), (600, 5720), (600, 5760), (600, 5800), (600, 5840), (600, 5880), (600, 5920), (600, 5960), (600, 6000), (600, 6040), (640, 5240), (640, 5280), (640, 5320), (640, 5360), (640, 5400), (640, 5440), (640, 5480), (640, 5520), (640, 5560), (640, 5600), (640, 5640), (640, 5680), (640, 5720), (640, 5760), (640, 5800), (640, 5840), (640, 5880), (640, 5920), (640, 5960), (640, 6000), (640, 6040), (680, 5240), (680, 5280), (680, 5320), (680, 5360), (680, 5400), (680, 5440), (680, 5480), (680, 5520), (680, 5560), (680, 5600), (680, 5640), (680, 5680), (680, 5720), (680, 5760), (680, 5800), (680, 5840), (680, 5880), (680, 5920), (680, 5960), (680, 6000), (720, 5240), (720, 5280), (720, 5320), (720, 5360), (720, 5400), (720, 5440), (720, 5480), (720, 5520), (720, 5560), (720, 5600), (720, 5640), (720, 5680), (720, 5720), (720, 5760), (720, 5800), (720, 5840), (720, 5880), (720, 5920), (720, 5960), (720, 6000), (720, 6040), (760, 5240), (760, 5280), (760, 5320), (760, 5360), (760, 5400), (760, 5440), (760, 5480), (760, 5560), (760, 5600), (760, 5640), (760, 5680), (760, 5720), (760, 5760), (760, 5800), (760, 5840), (760, 5880), (760, 5920), (760, 5960), (760, 6000), (760, 6040), (800, 5240), (800, 5280), (800, 5320), (800, 5360), (800, 5400), (800, 5440), (800, 5600), (800, 5640), (800, 5680), (800, 5720), (800, 5760), (800, 5800), (800, 5840), (800, 5880), (800, 5920), (800, 5960), (800, 6000), (800, 6040), (840, 5360), (840, 5400), (840, 5600), (840, 5640), (840, 5680), (840, 5720), (840, 5760), (840, 5800), (840, 5840), (840, 5880), (840, 5920), (840, 5960), (880, 5640), (880, 5680), (880, 5720), (880, 5760), (880, 5800)}

    def __init__(
            self,
            download: bool = False,
            verbose: bool = True,
            cache_dtype: np.dtype = np.float32,
    ):
        self.srid = 25832
        self.web_cache_path = config.OPENDTM_WEB_CACHE_PATH
        self.tile_cache_path = config.OPENDTM_TILE_CACHE_PATH
        self.verbose = verbose
        self.cache_dtype = cache_dtype
        self._download = download

    def _log(self, *args, **kwargs):
        kwargs["file"] = sys.stderr
        if self.verbose:
            print(f"{self.__class__.__name__}:", *args, **kwargs)

    def available_sectors(self, sectors: Optional[List[Sector]] = None):
        if sectors is None:
            sectors = self.AVAILABLE_SECTORS
        else:
            sectors = [s for s in sectors if s in self.AVAILABLE_SECTORS]
        return [
            s for s in sectors
            if self.sector_filename(s).exists()
        ]

    def sector_at(self, e: float, n: float) -> Optional[Sector]:
        sector = (
            int(math.floor(e / 40000) * 40),
            int(math.floor(n / 40000) * 40),
        )
        return sector if sector in self.AVAILABLE_SECTORS else None

    def download_sector(self, sector: Sector):
        if sector not in self.AVAILABLE_SECTORS:
            raise ValueError(f"Sector {sector} does not exist")

        filename_part = f"E{sector[0]}N{sector[1]}"
        cache_filename = self.web_cache_path / f"{filename_part}.zip"
        if cache_filename.exists():
            return

        streaming_download(
            url=f"https://openmaps.online/dtm_ger_download/{filename_part}.zip",
            local_filename=cache_filename,
            verbose=self.verbose,
        )

    def extract_sector(self, sector: Sector):
        if sector not in self.AVAILABLE_SECTORS:
            raise ValueError(f"Sector {sector} does not exist")

        filename_part = f"E{sector[0]}N{sector[1]}"
        cache_zip_filename = self.web_cache_path / f"{filename_part}.zip"
        sector_filename = self.sector_filename(sector)
        if not sector_filename.exists():
            if not cache_zip_filename.exists():
                if self._download:
                    self.download_sector(sector)
                else:
                    raise ValueError(f"Sector {sector} not downloaded")

            with zipfile.ZipFile(cache_zip_filename) as zf:
                possible_names = (
                    f"{filename_part}.tif",
                    f"{filename_part}/{filename_part}.tif"
                )
                tif_filename = None
                for possible_name in possible_names:
                    try:
                        zf.getinfo(possible_name)
                        tif_filename = possible_name
                        break
                    except KeyError:
                        pass
                if not tif_filename:
                    files = [f.filename for f in zf.filelist]
                    raise KeyError(f"None of {possible_names} found in zip, zipped files are:\n{files}")

                self._log(f"Extracting {filename_part}")
                with zf.open(tif_filename) as fp_src:
                    os.makedirs(sector_filename.parent, exist_ok=True)
                    with sector_filename.open("wb") as fp_dst:
                        while True:
                            data = fp_src.read(2**20)
                            if data:
                                fp_dst.write(data)
                            else:
                                break

    def sector_filename(self, sector: Sector) -> Path:
        return self.tile_cache_path / f"E{sector[0]}N{sector[1]}.tif"

    def open_sector(self, sector: Sector) -> rasterio.DatasetReader:
        if sector not in self.AVAILABLE_SECTORS:
            raise ValueError(f"Sector {sector} does not exist")
        filename = self.sector_filename(sector)
        if not filename.exists():
            if self._download:
                self.extract_sector(sector)
            else:
                raise ValueError(f"Sector {sector} not downloaded or extracted")
        return rasterio.open(filename)
