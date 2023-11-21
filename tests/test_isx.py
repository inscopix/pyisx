""" tests for the Python based ISX module"""
import os
from pathlib import Path

import isx
import pytest
from ideas_data.datasets import fetch

repo_root = Path(__file__).parent.parent
token_loc = os.path.join(repo_root, ".ideas-github-token")
with open(token_loc, "r") as file:
    token = file.read()


cell_set_files = [
    "striatum_cellset.isxd",
    "example_cellset.isxd",
    "sample_pca_ica_cellset.isxd",
]

cell_set_files = [fetch(file, token=token) for file in cell_set_files]


@pytest.mark.parametrize("file", cell_set_files)
def test_isxd_type(file):
    assert (
        isx.isxd_type(file) == "cell_set"
    ), f"Expected {file} to be of type cell_set"
