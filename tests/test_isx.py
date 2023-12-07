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

vessel_set_files = ["vessel_set_diameters.isxd"]

cell_set_files = [fetch(file, token=token) for file in cell_set_files]


@pytest.mark.parametrize("file", cell_set_files)
def test_isxd_type(file):
    assert (
        isx.isxd_type(file) == "cell_set"
    ), f"Expected {file} to be of type cell_set"


def test_vessel_set():
    vessel_set_file = fetch("vessel_set_diameters.isxd", token=token)
    vessel_obj = isx.VesselSet.read(vessel_set_file)
    im = vessel_obj.get_vessel_image_data(0)
    assert im.shape[0] > 0 and im.shape[1] > 0, "Expected image to have data"

    trace = vessel_obj.get_vessel_trace_data(0)
    assert trace.shape[0] > 0, "Expected trace to have data"

    name = vessel_obj.get_vessel_name(0)
    assert name == "V00", "Expected name to be V00"

    status = vessel_obj.get_vessel_status(0)
    assert status == "accepted", "Expected status to be accepted"
