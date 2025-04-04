import os
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm


def streaming_download(
        url: str,
        local_filename: Union[str, Path],
        chunk_size: int = 64_000,
        verbose: bool = False,
):
    """
    Download a BIG file from web.

    :param url: str, full url of file in web
    :param local_filename: str or Path, local filename (with or without directories)
        The directories will be created if they not exist.
    :param chunk_size: int, number of bytes to download & write at once
    :param verbose: bool, show download progress
    """
    local_filename = Path(local_filename)

    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            raise IOError(f"Status {r.status_code} for {r.request.url}")

        size = int(r.headers['content-length'])

        os.makedirs(local_filename.parent, exist_ok=True)
        try:
            with tqdm(
                    total=size,
                    desc=f"downloading {url}",
                    disable=not verbose,
                    unit_scale=True,
                    unit_divisor=1000,
                    unit="b",
            ) as progress:
                with open(str(local_filename), "wb") as fp:
                    for data in r.iter_content(chunk_size=chunk_size):
                        fp.write(data)
                        progress.update(len(data))

        except:
            # don't leave half files
            os.remove(str(local_filename))
            raise
