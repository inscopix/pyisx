import os
from pathlib import Path
from typing import Optional

import requests
from beartype import beartype
from beartype.typing import List, Union

if os.path.exists("/ideas/data/") and os.access("/ideas/data/", os.W_OK):
    data_root = "/ideas/data/"
elif os.path.exists("/tmp"):
    # on a POSIX system, use /tmp
    # this will work even if this repo is installed in
    # "non-editable" mode inside a "site-packages" folder
    # where you don't have write permissions
    data_root = os.path.join("/tmp", "data")
    if not os.path.isdir(data_root):
        os.makedirs(data_root)

else:
    # we're on a windows system, or some other weird system
    # attempt to use the install dir. this may fail
    # if we don't have permissions to write here
    data_root = os.path.join((Path(__file__).parent.parent), "data")

    if not os.path.isdir(data_root):
        os.makedirs(data_root)


@beartype
download(file_name:str)->None:
    response = requests.get("https://github.com/inscopix/py_isx/releases/download/1.0.0/movie_128x128x100_part1.isxd")
    file_name = "/Users/srinivas/Desktop/test.isxd"
    with open(file_name, "wb") as file:
            file.write(response.content)
