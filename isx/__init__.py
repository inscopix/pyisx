"""this module contains an experimental API to read ISXD files,
without using the isx API

this uses the filespec found here:

https://inscopix.atlassian.net/wiki/spaces/MOS/pages/64133259/Mosaic+2.0+File+Formats#Mosaic2.0FileFormats-DataFiles-.ISXD

"""


import json
import os
import struct
from enum import Enum

import numpy as np
from beartype import beartype


class Duration:
    """dummy class to mimic what IDPS isx.core.Duration does"""

    def __init__(self, secs_float: float):
        self.secs_float = secs_float


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
        return _read_cell_trace(self.file_path, cell_id)

    def get_cell_name(self, cell_id: int) -> str:
        """return name of cell"""
        return _read_name(self.file_path, cell_id, "Cell")

    def get_cell_status(self, cell_id: int) -> str:
        """return status of cell"""
        return _read_status(self.file_path, cell_id, "Cell")

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


class VesselSet:
    """class to maintain partial compatibility with isx.core.VesselSet"""

    def __init__(self):
        self.num_vessels: int = 0
        self.timing = Timing()
        self.file_path = None

    def get_vessel_image_data(self) -> np.array:
        """return footprint of a single vessel"""
        return _read_footprint(self.file_path, 0)

    def get_vessel_line_data(self, vessel_id: int) -> np.array:
        """return contour for a single vessel"""
        return _read_vessel_contour(self.file_path, vessel_id)

    def get_vessel_trace_data(self, vessel_id: int) -> np.array:
        """return trace for a single cell"""
        return _read_vessel_trace(self.file_path, vessel_id)

    def get_vessel_set_type(self) -> str:
        """return type of vessel set"""
        metadata = _extract_footer(self.file_path)
        return metadata["extraProperties"]["idps"]["vesselset"]["type"]

    def get_vessel_name(self, cell_id: int) -> str:
        """return name of cell"""
        return _read_name(self.file_path, cell_id, "Vessel")

    def get_vessel_status(self, cell_id: int) -> str:
        """return status of cell"""
        return _read_status(self.file_path, cell_id, "Vessel")

    @classmethod
    def read(cls, file_path: str):
        """method to maintain compatibility with IDPS API. This doesn't
        actually do anything very interesting other than set the timing
        and num_cells property"""

        self = cls()
        self.file_path = file_path

        footer = _extract_footer(file_path)

        self.num_vessels = len(footer["VesselNames"])

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
def _read_name(file: str, id: int, file_type: str) -> str:
    """return the name of a cell"""
    footer = _extract_footer(file)
    return footer[f"{file_type}Names"][id]


@beartype
def _read_cell_trace(cell_set_file: str, cell_id: int):
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


def _read_vessel_trace(vessel_set_file: str, vessel_id: int):
    """
    Structure of vessel set file:
    - Projection image [spacingInfo.numPixels.x * spacingInfo.numPixels.y * 4 bytes]
    - Vessels
        - Contour [4*8] if diameter, [8*8] if velocity
        - Trace [4 * timingInfo.numTimes]
        (repeated for each vessel)
    - Footer
    """
    footer = _extract_footer(vessel_set_file)
    n_frames = footer["timingInfo"]["numTimes"]

    # get frame dimensions
    size_x = footer["spacingInfo"]["numPixels"]["x"]
    size_y = footer["spacingInfo"]["numPixels"]["y"]
    n_pixels = size_y * size_x

    # Project image
    n_bytes_per_vessel = 4 * (n_pixels + n_frames)
    # Contour TODO:
    vessel_set_type = footer["extraProperties"]["idps"]["vesselset"]["type"]
    if vessel_set_type == "red blood cell velocity":
        num_to_unpack = 8
    elif vessel_set_type == "vessel diameter":
        num_to_unpack = 4
    else:
        raise ValueError(f"Unknown vessel set type: {vessel_set_type}")

    contour_len = 8 * num_to_unpack
    trace_len = 4 * n_frames

    # Projection image + previous vessels + current contour
    if footer["VesselCenterSaved"]:
        print("Vessel center saved")
        data_loc = (
            n_bytes_per_vessel
            + (vessel_id * (contour_len + (2 * trace_len)))
            + contour_len
        )
    else:
        # Projection image + previous vessels
        data_loc = (
            n_bytes_per_vessel
            + (vessel_id * (contour_len + trace_len))
            + contour_len
        )

    with open(vessel_set_file, mode="rb") as file:
        file.seek(data_loc)

        # read cell trace
        data = file.read(trace_len)
        trace = struct.unpack("f" * n_frames, data)
        trace = np.array(trace)

    return trace


def _read_vessel_contour(vessel_set_file: str, vessel_id: int):
    """
    Structure of vessel set file:
    - Projection image [spacingInfo.numPixels.x * spacingInfo.numPixels.y * 4 bytes]
    - Vessels
        - Contour [4*8] if diameter, [8*8] if velocity
        - Trace [4 * timingInfo.numTimes]
        (repeated for each vessel)
    - Footer
    """
    footer = _extract_footer(vessel_set_file)
    n_frames = footer["timingInfo"]["numTimes"]

    # get frame dimensions
    size_x = footer["spacingInfo"]["numPixels"]["x"]
    size_y = footer["spacingInfo"]["numPixels"]["y"]
    n_pixels = size_y * size_x

    # Project image
    n_bytes_per_vessel = 4 * (n_pixels + n_frames)

    vessel_set_type = footer["extraProperties"]["idps"]["vesselset"]["type"]

    if vessel_set_type == "red blood cell velocity":
        num_to_unpack = 8
    elif vessel_set_type == "vessel diameter":
        num_to_unpack = 4
    else:
        raise ValueError(f"Unknown vessel set type: {vessel_set_type}")

    contour_len = 4 * num_to_unpack
    # Trace
    trace_len = 4 * n_frames
    if footer["VesselCenterSaved"]:
        print("Vessel center saved")
        data_loc = n_bytes_per_vessel + (
            vessel_id * (contour_len + (2 * trace_len))
        )
    else:
        # Projection image + previous vessels
        data_loc = n_bytes_per_vessel + (vessel_id * (contour_len + trace_len))

    with open(vessel_set_file, mode="rb") as file:
        file.seek(data_loc)

        # read cell trace
        data = file.read(16)
        contour = struct.unpack("i" * 4, data)
        contour2 = struct.unpack("I" * 4, data)  # fix prefix to be int
        contour = np.array(contour)
        contour2 = np.array(contour2)
    print(f"i: {contour}")
    print(f"I: {contour2}")

    return contour


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


def _show_footer(file: str):
    """print the footer of a given ISXD file"""

    footer = _extract_footer(file)
    # just show top level keys
    print(json.dumps(footer))


@beartype
def _read_status(file: str, id: int, file_type: str) -> str:
    """standalone function to read the status of a given cell
    from a cellset file, without needing the IDPS API

    """

    footer = _extract_footer(file)

    if footer[f"{file_type}Statuses"][id] == 0:
        return "accepted"
    elif footer[f"{file_type}Statuses"][id] == 1:
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


def _read_frame(movie_file: str, frame_num: int) -> np.array:
    """read a single frame from a ISXD movie file"""
    # read footer
    footer = _extract_footer(movie_file)

    if footer["dataType"] != 1:
        raise RuntimeError(
            "[UNIMPLEMENTED] dataType is not float32. Unable to read frames. "
        )

    if footer["hasFrameHeaderFooter"]:
        raise RuntimeError(
            """[UNIMPLEMENTED] Cannot extract frame from this
    movie because frames have footers and headers."""
        )

    n_frames = footer["timingInfo"]["numTimes"]

    if frame_num >= n_frames:
        raise RuntimeError(
            f"""[INVALID FRAME NUMBER] This movie has
    {n_frames}, so accessing frame number {frame_num} 
    is impossible."""
        )

    size_x = footer["spacingInfo"]["numPixels"]["x"]
    size_y = footer["spacingInfo"]["numPixels"]["y"]

    frame = np.zeros((size_x, size_y))
    n_pixels = size_y * size_x

    n_bytes_per_frame = size_y * size_x * 4

    with open(movie_file, mode="rb") as file:
        file.seek(frame_num * n_bytes_per_frame)
        for i in np.arange(n_pixels):
            data = file.read(4)  # read 4 bytes at a time, float 32
            x, y = np.unravel_index(i, frame.shape)
            frame[x, y] = struct.unpack("f", data)[0]

    return frame


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
