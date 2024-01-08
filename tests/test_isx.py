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
    ),
    dict(
        name="movie_longer_than_3_min.isxd",
        dtype=np.uint16,
        num_pixels=(33, 29),
        num_samples=1248,
    ),
    dict(
        name="movie_u8.isxd",
        dtype=np.uint8,
        num_pixels=(3, 4),
        num_samples=5,
    ),
]

# download files if needed and resolve to local File system
for movie in movie_info:
    movie["name"] = download(movie["name"])


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
