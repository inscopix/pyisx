"""this module contains helper code that downloads test
data from a github releases page"""

import os
from pathlib import Path

import requests
from beartype import beartype

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
def download(file_name: str) -> str:
    """helper function that downloads test data (if needed)
    and returns a path to the file on the local filesystem"""

    file_path = os.path.join(data_root, file_name)

    if os.path.exists(file_path):
        return file_path

    response = requests.get(
        f"https://github.com/inscopix/py_isx/releases/download/test-data/{file_name}"
    )

    with open(file_path, "wb") as file:
        file.write(response.content)

    return file_path
