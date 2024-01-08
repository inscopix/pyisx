"""this module contains a pure-python implementation
of the I/O code found in the IDPS API to read ISXD files
of the Movie and CellSet variety. 

This module is not at feature parity with the IDPS C++ API
yet, and some features may not be supported. 

"""

import json
import os
import struct

import importlib_metadata
import numpy as np
from beartype import beartype

NOT_IMPLEMENTED_MESSAGE = """
This functionality has not been implemented in the pure python
API yet. If you need this, please use the IDPS Python API"""

__version__ = importlib_metadata.version("isx")


class Duration:
    """
    !!! info "IDPS Equivalent"
        This class is designed to be equivalent of the `isx.Duration`class in the IDPS Python API`. Not all
        features of the IDPS Python API are mirrored here.

    Attributes:
        secs_float: A period of time, expressed in seconds.

    """

    secs_float: float = None

    def __init__(self, secs_float: float):
        self.secs_float = secs_float


class Spacing:
    """
    !!! info "IDPS Equivalent"
        This class is designed to be equivalent of the `isx.Spacing`class in the IDPS Python API`. Not all
        features of the IDPS Python API are mirrored here.

    Attributes:
        num_pixels: A 2-tuple containing the dimensions of the frame.

    """

    num_pixels: tuple[int, int] = None

    def __int__(self):
        self.num_pixels = None


class Timing:
    """
    !!! info "IDPS Equivalent"
        This class is designed to be equivalent of the `isx.Timing`class in the IDPS Python API`. Not all
        features of the IDPS Python API are mirrored here.

    Attributes:
        period: An instance of isx.Duration with information about the period
        num_samples: The number of time samples in this object.

    """

    period: Duration = None
    num_samples: int = None

    def __init__(self):
        self.period = None
        self.num_samples = None


class Movie:
    """
    !!! info "IDPS Equivalent"
        This class is designed to be equivalent of the `isx.Movie`class in the IDPS Python API

    The Movie class allows you to create objects
    that represent ISXD movies. Every Movie object
    is bound to a ISXD movie file that exists on disk.

    Attributes:
        file_path: path to ISXD file
        footer: A dictioanry containing data in the JSON    footer of ISXD Movies
        timing: a isx.Timing object containing timing information for this movie
        spacing: a isx.Spacing object containing spacing information for this movie


    """

    file_path: str = None
    footer: dict = None
    timing: Timing = Timing()
    spacing: Timing = Spacing()

    def __init__(self):
        pass

    @property
    def data_type(self):
        if self.footer["dataType"] == 0:
            return np.uint16
        elif self.footer["dataType"] == 1:
            return np.float32
        elif self.footer["dataType"] == 2:
            return np.uint8
        else:
            raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def read(cls, file_path: str):
        """
        Open an existing movie from a file for reading.

        This is a light weight operation that simply reads the meta-data from the movie,
        and does not read any frame data.

        Parameters:
            file_path: The path of the file to read.

        Returns:
            A `isx.Movie` object. The movie that was read. Meta-data is immediately available. Frames must be read using `isx.Movie.get_frame`.
        """
        self = cls()
        self.file_path = file_path

        footer = _extract_footer(file_path)

        # y maps to rows, x maps to columns
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
        Read the contents of a single frame in a movie

        Parameters:
            index : The numeric index of the frame.

        Returns:
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


class CellSet:
    """

    The CellSet class allows you to read ISXD CellSets.

    !!! info "How to use the CellSet class"
        To see how to use this class to read data from
        ISXD Cellsets, click [here](../how-to/read-cellset.md).
        This reference page describes each member of this
        class and what each function does."""

    num_cells: int = 0
    timing = None
    file_path = None
    footer = None

    def __init__(self):
        self.num_cells: int = 0
        self.timing = Timing()
        self.file_path = None

    def get_cell_image_data(self, cell_id: int) -> np.array:
        """This method reads the spatial footprint of a single
        cell and returns that as a Numpy array.

        Parameters:
            cell_id: index of cell of interest

        Returns:
            A MxN Numpy array containing frame data where M and N are the pixel dimensions
        """

        n_frames = self.footer["timingInfo"]["numTimes"]

        # get frame dimensions
        size_x = self.footer["spacingInfo"]["numPixels"]["x"]
        size_y = self.footer["spacingInfo"]["numPixels"]["y"]
        n_pixels = size_y * size_x

        n_bytes_per_cell = 4 * (n_pixels + n_frames)

        with open(self.file_path, mode="rb") as file:
            file.seek(cell_id * n_bytes_per_cell)
            data = file.read(4 * n_pixels)

        footprint = struct.unpack("f" * n_pixels, data)
        footprint = np.array(footprint).reshape((size_y, size_x))

        return footprint

    def get_cell_trace_data(self, cell_id: int) -> np.array:
        """return trace for a single cell"""

        n_frames = self.footer["timingInfo"]["numTimes"]

        # get frame dimensions
        size_x = self.footer["spacingInfo"]["numPixels"]["x"]
        size_y = self.footer["spacingInfo"]["numPixels"]["y"]
        n_pixels = size_y * size_x

        n_bytes_per_cell = 4 * (n_pixels + n_frames)

        with open(self.file_path, mode="rb") as file:
            file.seek(cell_id * n_bytes_per_cell + (4 * n_pixels))

            # read cell trace
            data = file.read(4 * n_frames)
            trace = struct.unpack("f" * n_frames, data)
            trace = np.array(trace)

        return trace

    def get_cell_name(self, cell_id: int) -> str:
        """return name of cell"""

        return self.footer["CellNames"][cell_id]

    def get_cell_status(self, cell_id: int) -> str:
        """return status of cell"""
        if self.footer["CellStatuses"][cell_id] == 0:
            return "accepted"
        elif self.footer["CellStatuses"][cell_id] == 1:
            return "undecided"
        else:
            return "rejected"

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
    """infer ISXD file type

    Parameters:
        file_path: path to ISXD file

    """

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
