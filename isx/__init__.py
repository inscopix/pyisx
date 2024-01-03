"""this module contains a pure-python implementation
of the I/O code found in the IDPS API to read ISXD files
of the Movie and CellSet variety. 

This module is not at feature parity with the IDPS C++ API
yet, and some features may not be supported. 

"""


import json
import os
import struct

import numpy as np
from beartype import beartype

NOT_IMPLEMENTED_MESSAGE = """
This functionality has not been implemented in the pure python
API yet. If you need this, please use the IDPS Python API"""


class Movie:
    """Movie"""

    def __init__(self):
        self.file_path = None
        self.footer = None
        self.timing = Timing()
        self.spacing = Spacing()

    @property
    def data_type(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def read(cls, file_path):
        """
        Open an existing movie from a file for reading.

        This is a light weight operation that simply reads the meta-data from the movie,
        and does not read any frame data.

        Arguments
        ---------
        file_path : str
            The path of the file to read.

        Returns
        -------
        :class:`isx.Movie`
            The movie that was read. Meta-data is immediately available.
            Frames must be read using :func:`isx.Movie.get_frame`.
        """
        self = cls()
        self.file_path = file_path

        footer = _extract_footer(file_path)

        self.spacing.num_pixels = (
            footer["spacingInfo"]["numPixels"]["y"],
            footer["spacingInfo"]["numPixels"]["x"],
        )

        self.timing.num_samples = footer["timingInfo"]["numTimes"]

        self.timing.period = Duration(
            footer["timingInfo"]["period"]["num"]
            / footer["timingInfo"]["period"]["den"]
        )

        # save the footer too
        self.footer = footer

        return self

    @classmethod
    def write(cls, file_path, timing, spacing, data_type):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @beartype
    def get_frame_data(self, index: int):
        """
        Get a frame from the movie by index.

        Arguments
        ---------
        index : int >= 0
            The index of the frame. If this is out of range, this should error.

        Returns
        -------
        :class:`numpy.ndarray`
            The retrieved frame data.
        """

        if self.footer["dataType"] == 0:
            bytes_per_pixel = 2
            format_string = "H"
        elif self.footer["dataType"] == 1:
            bytes_per_pixel = 4
            format_string = "f"
        elif self.footer["dataType"] == 2:
            bytes_per_pixel = 1
            format_string = "b"
        else:
            print

            raise NotImplementedError(
                "Unknown number of bytes per pixel. Cannot decode this frame."
            )

        if self.footer["hasFrameHeaderFooter"]:
            raise NotImplementedError(
                """[UNIMPLEMENTED] Cannot extract frame from this
        movie because frames have footers and headers."""
            )

        n_frames = self.footer["timingInfo"]["numTimes"]

        if index >= n_frames:
            raise IndexError(
                f"""[INVALID FRAME NUMBER] This movie has
        {n_frames}, so accessing frame number {index} 
        is impossible."""
            )

        n_pixels = self.spacing.num_pixels[0] * self.spacing.num_pixels[1]

        n_bytes_per_frame = n_pixels * bytes_per_pixel

        with open(self.file_path, mode="rb") as file:
            file.seek(index * n_bytes_per_frame)

            data = file.read(bytes_per_pixel * n_pixels)

            frame = struct.unpack(format_string * n_pixels, data)
            frame = np.reshape(frame, self.spacing.num_pixels)

        return frame

    def get_frame_timestamp(self, index):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def set_frame_data(self, index, frame):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def flush(self):
        """this method exists for drop-in compatibility
        with the IDPS API, but doesn't do anything"""
        pass

    def get_acquisition_info(self):
        return None

    def __del__(self):
        pass


class Duration:
    """dummy class to mimic what IDPS isx.core.Duration does"""

    def __init__(self, secs_float: float):
        self.secs_float = secs_float


class Spacing:
    """stores spacing information for compatibility with IDPS"""

    def __int__(self):
        self.num_pixels = None


class Timing:
    """dummy class to mimic what IDPS isx.core.Timing does"""

    def __init__(self):
        self.period = None
        self.num_samples = None


class CellSet:
    """class to maintain partial compatibility with isx.core.CellSet"""

    def __init__(self):
        self.num_cells: int = 0
        self.timing = Timing()
        self.file_path = None

    def get_cell_image_data(self, cell_id: int) -> np.array:
        """return footprint of a single cell"""
        return _read_footprint(self.file_path, cell_id)

    def get_cell_trace_data(self, cell_id: int) -> np.array:
        """return trace for a single cell"""
        return _read_trace(self.file_path, cell_id)

    def get_cell_name(self, cell_id: int) -> str:
        """return name of cell"""
        return _read_cell_name(self.file_path, cell_id)

    def get_cell_status(self, cell_id: int) -> str:
        """return status of cell"""
        return _read_status(self.file_path, cell_id)

    @classmethod
    def read(cls, file_path: str):
        """method to maintain compatibility with IDPS API. This doesn't
        actually do anything very interesting other than set the timing
        and num_cells property"""

        self = cls()
        self.file_path = file_path

        footer = _extract_footer(file_path)

        self.num_cells = len(footer["CellNames"])

        self.timing.num_samples = footer["timingInfo"]["numTimes"]

        self.timing.period = Duration(
            footer["timingInfo"]["period"]["num"]
            / footer["timingInfo"]["period"]["den"]
        )

        return self


@beartype
def isxd_type(file_path: str) -> str:
    """infer ISXD file type"""

    metadata = _extract_footer(file_path)

    isx_datatype_mapping = {
        0: "miniscope_movie",
        1: "cell_set",
        2: "isxd_behavioral_movie",  # not currently supported on IDEAS
        3: "gpio_data",
        4: "miniscope_image",
        5: "neural_events",
        6: "isxd_metrics",  # not currently supported on IDEAS
        7: "imu_data",
        8: "vessel_set",
    }
    return isx_datatype_mapping[metadata["type"]]


@beartype
def _read_cell_name(cell_set_file: str, cell_id: int) -> str:
    """return the name of a cell"""
    footer = _extract_footer(cell_set_file)
    return footer["CellNames"][cell_id]


@beartype
def _read_trace(cell_set_file: str, cell_id: int):
    """stand-alone function to read a single cell's trace
    from a cellset file
    """

    footer = _extract_footer(cell_set_file)
    n_frames = footer["timingInfo"]["numTimes"]

    # get frame dimensions
    size_x = footer["spacingInfo"]["numPixels"]["x"]
    size_y = footer["spacingInfo"]["numPixels"]["y"]
    n_pixels = size_y * size_x

    n_bytes_per_cell = 4 * (n_pixels + n_frames)

    with open(cell_set_file, mode="rb") as file:
        file.seek(cell_id * n_bytes_per_cell + (4 * n_pixels))

        # read cell trace
        data = file.read(4 * n_frames)
        trace = struct.unpack("f" * n_frames, data)
        trace = np.array(trace)

    return trace


@beartype
def _read_footprint(cell_set_file: str, cell_id):
    """standalone function to read a footprint of a single
    cell from a cellset file

    """

    footer = _extract_footer(cell_set_file)
    n_frames = footer["timingInfo"]["numTimes"]

    # get frame dimensions
    size_x = footer["spacingInfo"]["numPixels"]["x"]
    size_y = footer["spacingInfo"]["numPixels"]["y"]
    n_pixels = size_y * size_x

    n_bytes_per_cell = 4 * (n_pixels + n_frames)

    with open(cell_set_file, mode="rb") as file:
        file.seek(cell_id * n_bytes_per_cell)
        data = file.read(4 * n_pixels)

    footprint = struct.unpack("f" * n_pixels, data)
    footprint = np.array(footprint).reshape((size_y, size_x))

    return footprint


@beartype
def _read_status(cell_set_file: str, cell_id: int) -> str:
    """standalone function to read the status of a given cell
    from a cellset file, without needing the IDPS API

    """

    footer = _extract_footer(cell_set_file)

    if footer["CellStatuses"][cell_id] == 0:
        return "accepted"
    elif footer["CellStatuses"][cell_id] == 1:
        return "undecided"
    else:
        return "rejected"


@beartype
def _footer_length(isxd_file: str) -> int:
    """find the length of the footer in bytes"""

    with open(isxd_file, mode="rb") as file:
        file.seek(-8, os.SEEK_END)
        data = file.read()
    footer_length = struct.unpack("ii", data)[0]

    return footer_length


@beartype
def _extract_footer(isxd_file: str) -> dict:
    """extract movie footer from ISXD file"""

    footer_length = _footer_length(isxd_file)

    with open(isxd_file, mode="rb") as file:
        file.seek(-8 - footer_length - 1, os.SEEK_END)
        data = file.read(footer_length)

    footer = data.decode("utf-8")
    return json.loads(footer)


def _get_isxd_times(input_filename):
    """Get the timestamps of every sample of an isxd file from its metadata.

    The timestamps are generated by getting the average sampling period
    of the isxd file.

    :param input_filename str: path to the input file (.isxd)
    :return: The timestamps of every sample in the isxd file
    """

    metadata = _extract_footer(input_filename)
    period = (
        metadata["timingInfo"]["period"]["num"]
        / metadata["timingInfo"]["period"]["den"]
    )
    num_times = metadata["timingInfo"]["numTimes"]
    times = np.linspace(
        0,
        (num_times - 1) * period,
        num_times,
    )
    return times
