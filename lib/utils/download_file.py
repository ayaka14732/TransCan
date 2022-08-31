from os.path import join
import subprocess
from typing import Optional

def download_file(url: str, filename: str, dir_prefix: Optional[str]=None) -> None:
    try:
        if dir_prefix is None:
            subprocess.call(['aria2c', '-x16', '-s16', '-o', filename, url])
        else:
            subprocess.call(['aria2c', '-x16', '-s16', '-d', dir_prefix, '-o', filename, url])
    except FileNotFoundError:
        pass
    else:
        return

    if dir_prefix is not None:
        filename = join(dir_prefix, filename)

    try:
        subprocess.call(['curl', '-o', filename, url])
    except FileNotFoundError:
        pass
    else:
        return

    subprocess.call(['wget', '-O', filename, url])
