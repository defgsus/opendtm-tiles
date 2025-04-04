from pathlib import Path
import decouple

PROJECT_PATH = Path(__file__).resolve().parent.parent


OPENDTM_WEB_CACHE_PATH = Path(
    decouple.config("OPENDTM_WEB_CACHE_PATH", PROJECT_PATH / "cache" / "web")
)

OPENDTM_TILE_CACHE_PATH = Path(
    decouple.config("OPENDTM_WEB_CACHE_PATH", PROJECT_PATH / "cache" / "tiles")
)

OPENDTM_SECTOR_X = [int(i) for i in decouple.config("OPENDTM_SECTOR_X", "280 880").split()]
OPENDTM_SECTOR_Y = [int(i) for i in decouple.config("OPENDTM_SECTOR_Y", "5200 6080").split()]