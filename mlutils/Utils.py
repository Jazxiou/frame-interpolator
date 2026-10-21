# These utilities are from Jon Macey's workshops.
import platform
import re

import requests
import torch
from tqdm import tqdm


def get_device() -> torch.device:
    """
    Returns the appropriate device for our current environment.
    If GPU can be found it will use this.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def in_lab() -> bool:
    hostname = platform.node()
    return re.search(r"(pg|w)\d+", hostname) is not None


def download(url: str, fname: str):
    resp = requests.get(url, stream=True, verify=False)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(fname, "wb") as file,
        tqdm(desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024) as progress_bar,
    ):
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
            