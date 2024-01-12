""" tests for the Python based ISX module"""


import isx
import numpy as np
import pytest
from data import download

# information about each movie that we will check
movie_info = [
    dict(
        name="movie_128x128x100_part1.isxd",
        dtype=np.float32,
        num_pixels=(128, 128),
        num_samples=100,
        frame_max=1146.0001,
        frame_min=703.0001,
        frame_sum=15429191.0,
    ),
    dict(
        name="movie_longer_than_3_min.isxd",
        dtype=np.uint16,
        num_pixels=(33, 29),
        num_samples=1248,
        frame_max=2658,
        frame_min=492,
        frame_sum=1400150,
    ),
    dict(
        name="movie_u8.isxd",
        dtype=np.uint8,
        num_pixels=(3, 4),
        num_samples=5,
        frame_max=11,
        frame_min=0,
        frame_sum=66,
    ),
]

cellset_info = [
    dict(
        name="empty_cellset.isxd",
        num_cells=0,
        num_pixels=(4, 5),
        num_samples=7,
        n_accepted=0,
        n_rejected=0,
    ),
    dict(
        name="cellset.isxd",
        num_cells=4,
        num_pixels=(366, 398),
        num_samples=5444,
        n_accepted=3,
        n_rejected=1,
    ),
    dict(
        name="cellset_series_part1.isxd",
        num_cells=6,
        num_pixels=(21, 21),
        num_samples=100,
        n_accepted=0,
        n_rejected=0,
    ),
]

# download files if needed and resolve to local File system
for item in movie_info:
    item["name"] = download(item["name"])
for item in cellset_info:
    item["name"] = download(item["name"])


cell_set_methods = [
    "get_cell_name",
    "get_cell_image_data",
    "get_cell_trace_data",
    "get_cell_status",
]


def _read_all_status(cell_set: isx.CellSet) -> list[str]:
    """helper function to read all status in cellset"""
    cell_status = []

    for i in range(cell_set.num_cells):
        cell_status.append(cell_set.get_cell_status(i))

    return cell_status


@pytest.mark.parametrize("item", cellset_info)
def test_cellset_status(item):
    """check that we can read the number of samples correctly"""

    cell_set = isx.CellSet.read(item["name"])
    cell_status = _read_all_status(cell_set)

    assert (
        cell_status.count("accepted") == item["n_accepted"]
    ), f"Could not read the number of accepted cells correctly for {item['name']}"

    assert (
        cell_status.count("rejected") == item["n_rejected"]
    ), f"Could not read the number of accepted cells correctly for {item['name']}"


@pytest.mark.parametrize("item", cellset_info)
def test_cellset_num_samples(item):
    """check that we can read the number of samples correctly"""

    cell_set = isx.CellSet.read(item["name"])

    assert (
        cell_set.timing.num_samples == item["num_samples"]
    ), f"Could not read the number of pixels correctly for {item['name']}"


@pytest.mark.parametrize("item", cellset_info)
def test_cellset_num_pixels(item):
    """check that we can read the number of pixels correctly"""

    cell_set = isx.CellSet.read(item["name"])

    assert (
        cell_set.spacing.num_pixels == item["num_pixels"]
    ), f"Could not read the number of pixels correctly for {item['name']}"


@pytest.mark.parametrize("item", cellset_info)
def test_read_num_cells(item):
    """check that we can read the number of cells in a cellset correctly"""

    cell_set = isx.CellSet.read(item["name"])

    assert (
        cell_set.num_cells == item["num_cells"]
    ), f"Could not read the number of cells correctly for {item['name']}"


@pytest.mark.parametrize("method", cell_set_methods)
@pytest.mark.parametrize("item", cellset_info)
def test_error_on_bad_cell_index(item, method):
    """check that we get the correct error message when we try to read info from a cell that doesn't exist"""

    cell_set = isx.CellSet.read(item["name"])

    with pytest.raises(IndexError, match="Cell ID must be >=0"):
        getattr(cell_set, method)(-1)

    with pytest.raises(IndexError, match="Cannot access cell"):
        getattr(cell_set, method)(cell_set.num_cells + 1)


@pytest.mark.parametrize("item", movie_info)
def test_isxd_type_movie(item):
    """test that we can identify file types correctly"""

    movie_file = item["name"]

    assert (
        isx.isxd_type(movie_file) == "miniscope_movie"
    ), f"Expected {movie_file} to be of type miniscope_movie"


@pytest.mark.parametrize("item", movie_info)
def test_movie_data_type(item):
    """check that we can correctly identify movie data type"""

    movie_name = item["name"]
    movie = isx.Movie.read(movie_name)

    assert (
        movie.data_type == item["dtype"]
    ), f"Could not correctly read data type of movie {movie_name}"


@pytest.mark.parametrize("item", movie_info)
def test_movie_num_pixels(item):
    """check that we can correctly read the frame size of each movie"""

    movie_name = item["name"]
    movie = isx.Movie.read(movie_name)

    assert (
        movie.spacing.num_pixels == item["num_pixels"]
    ), f"Could not correctly read num_pixels of movie {movie_name}"


@pytest.mark.parametrize("item", movie_info)
def test_movie_num_samples(item):
    """check that we can correctly read the number of samples (number of frames) from the movie"""

    movie_name = item["name"]
    movie = isx.Movie.read(movie_name)

    assert (
        movie.timing.num_samples == item["num_samples"]
    ), f"Could not correctly read num_samples of movie {movie_name}"


@pytest.mark.parametrize("item", movie_info)
def test_movie_read_frame(item):
    """check that we can correctly read the first frame of the movie by checking that frame's min, sum and max"""

    movie_name = item["name"]
    movie = isx.Movie.read(movie_name)
    frame = movie.get_frame_data(0)

    assert np.isclose(
        frame.max(), item["frame_max"]
    ), f"Could not correctly the first frame of {movie_name}"

    assert np.isclose(
        frame.min(), item["frame_min"]
    ), f"Could not correctly the first frame of {movie_name}"

    assert np.isclose(
        frame.sum(), item["frame_sum"]
    ), f"Could not correctly the first frame of {movie_name}"
