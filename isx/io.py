"""
The io module deals with data input and output.

This includes reading from and writing to supported file formats for
movies, images, cell sets and event sets.
"""

import os
import ctypes
import textwrap
import tifffile
from enum import Enum
import warnings

import numpy as np
import pandas as pd

import PIL.Image

import isx._internal
import isx.core


class Movie(object):
    """
    A movie contains a number of frames with timing and spacing information.

    It is always backed by a file, which can be read or written using this class.
    See :ref:`importMovie` for details on what formats are supported for read.
    Only the native `.isxd` format is supported for write.

    Examples
    --------
    Read an existing movie and get its first frame as a numpy array.

    >>> movie = isx.Movie.read('recording_20160613_105808-PP-PP.isxd')
    >>> frame_data = movie.get_frame_data(0)

    Write a 400x300 movie with 200 random frames of float32 values.

    >>> timing = isx.Timing(num_samples=200)
    >>> spacing = isx.Spacing(num_pixels=(300, 400))
    >>> movie = isx.Movie.write('movie-400x300x200.isxd', timing, spacing, numpy.float32)
    >>> for i in range(timing.num_samples):
    >>>     movie.set_frame_data(i, numpy.random.random(spacing.num_pixels).astype(numpy.float32))
    >>> movie.flush()

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    timing : :class:`isx.Timing`
        The timing of the frames.
    spacing : :class:`isx.Spacing`
        The spacing of the pixels in each frame.
    data_type : {numpy.uint16, numpy.float32}
        The data type of each pixel.
    """

    def __init__(self):
        self._ptr = isx._internal.IsxMoviePtr()

    @property
    def file_path(self):
        return self._ptr.contents.file_path.decode() if self._ptr else None

    @property
    def mode(self):
        return isx._internal.get_mode_from_read_only(self._ptr.contents.read_only) if self._ptr else None

    @property
    def timing(self):
        return isx.core.Timing._from_impl(self._ptr.contents.timing) if self._ptr else None

    @property
    def spacing(self):
        return isx.core.Spacing._from_impl(self._ptr.contents.spacing) if self._ptr else None

    @property
    def data_type(self):
        return isx._internal.DATA_TYPE_TO_NUMPY[self._ptr.contents.data_type] if self._ptr else None

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
        movie = cls()
        isx._internal.c_api.isx_read_movie(file_path.encode('utf-8'), ctypes.byref(movie._ptr))
        return movie

    @classmethod
    def write(cls, file_path, timing, spacing, data_type):
        """
        Open a new movie to a file for writing.

        This is a light weight operation. It does not write any frame data immediately.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        timing : :class:`isx.Timing`
            The timing of the movie to write.
        spacing : :class:`isx.Spacing`
            The spacing of the movie to write.
        data_type : {numpy.uint16, numpy.float32}
            The data type of each pixel.

        Returns
        -------
        :class:`isx.Movie`
            The empty movie that was written.
            Frame data must be written with :func:`isx.Movie.set_frame_data`.
        """
        movie = cls()
        data_type_int = isx._internal.lookup_enum('data_type', isx._internal.DATA_TYPE_FROM_NUMPY, data_type)
        isx._internal.c_api.isx_write_movie(file_path.encode('utf-8'), timing._impl, spacing._impl, data_type_int, False, ctypes.byref(movie._ptr))
        return movie

    def get_frame_data(self, index):
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
        isx._internal.validate_ptr(self._ptr)

        shape = self.spacing.num_pixels
        f = np.zeros([np.prod(shape)], dtype=self.data_type)

        if self.data_type == np.uint16:
            f_p = f.ctypes.data_as(isx._internal.UInt16Ptr)
            isx._internal.c_api.isx_movie_get_frame_data_u16(self._ptr, index, f_p)
        elif self.data_type == np.float32:
            f_p = f.ctypes.data_as(isx._internal.FloatPtr)
            isx._internal.c_api.isx_movie_get_frame_data_f32(self._ptr, index, f_p)
        elif self.data_type == np.uint8:
            f_p = f.ctypes.data_as(isx._internal.UInt8Ptr)
            isx._internal.c_api.isx_movie_get_frame_data_u8(self._ptr, index, f_p)
        else:
            raise RuntimeError('Cannot read from movie with datatype: {}'.format(str(self.data_type)))

        return f.reshape(shape)

    def get_frame_timestamp(self, index):
        """
        Get a frame timestamp from the movie by index.

        The timestamps are in units of microseconds.
        This is a TSC (time stamp counter) value which is saved during acquisition.
        These values come from a hardware counter on a particular acquisition box.
        As a result, they can only be used to compare to other values that originate from the same hardware counter (e.g., paired recordings).

        To get timestamps relative to the start of the movie,
        simply subtract each timestamp with the timestamp of the first frame in the movie.
        To get timestamps relative to Unix epoch time, add the timestamps computed relative
        to the start of the movie with the Unix epoch start timestamp of the movie, accessible through the `timing` member of this class.
        Alternatively, the timestamps of a movie can be exported relative to the start of the movie,
        or relative to the Unix epoch time, using the function `isx.export_movie_timestamps_to_csv`.

        Arguments
        ---------
        index : int >= 0
            The index of the frame. If this is out of range, this should error.

        Returns
        -------
        int
            The retreived frame timestamp.
            If the movie has no frame timestamps, the function will throw an error,
        """
        isx._internal.validate_ptr(self._ptr)
        timestamp = ctypes.c_uint64(0)
        timestamp_ptr = ctypes.pointer(timestamp)
        isx._internal.c_api.isx_movie_get_frame_timestamp(self._ptr, index, timestamp_ptr)
        return timestamp.value

    def set_frame_data(self, index, frame):
        """
        Set frame data in a writable movie.

        Frames must be set in increasing order, otherwise this will error.

        Arguments
        ---------
        index : int >= 0
            The index of the frame.
        frame : :class:`numpy.ndarray`
            The frame data.
        """
        isx._internal.validate_ptr(self._ptr)

        if self.mode != 'w':
            raise ValueError('Cannot set frame data if movie is read-only.')

        if not isinstance(frame, np.ndarray):
            raise TypeError('Frame must be a numpy array')

        if frame.shape != self.spacing.num_pixels:
            raise ValueError('Cannot set frame with different shape than movie')

        f_flat = isx._internal.ndarray_as_type(frame, np.dtype(self.data_type)).ravel()

        if self.data_type == np.uint16:
            FrameType = ctypes.c_uint16 * np.prod(frame.shape)
            c_frame = FrameType(*f_flat)
            isx._internal.c_api.isx_movie_write_frame_u16(self._ptr, index, c_frame)
        elif self.data_type == np.float32:
            FrameType = ctypes.c_float * np.prod(frame.shape)
            c_frame = FrameType(*f_flat)
            isx._internal.c_api.isx_movie_write_frame_f32(self._ptr, index, c_frame)
        else:
            raise RuntimeError('Cannot write frames for movie with datatype: {}'.format(str(self.data_type)))

    def flush(self):
        """
        Flush all meta-data and frame data to the file.

        This should be called after setting all frames of a movie opened with :func:`isx.Movie.write`.
        """
        isx._internal.validate_ptr(self._ptr)
        isx._internal.c_api.isx_movie_flush(self._ptr)

    def get_acquisition_info(self):
        """
        Get information about acquisition that may be stored in some files,
        such as nVista 3 movies and data derived from those.

        Returns
        -------
        dict
            A dictionary likely parsed from JSON that maps from string keys to variant values.
        """
        return isx._internal.get_acquisition_info(
                self._ptr,
                isx._internal.c_api.isx_movie_get_acquisition_info,
                isx._internal.c_api.isx_movie_get_acquisition_info_size);

    def __del__(self):
        if self._ptr:
            isx._internal.c_api.isx_movie_delete(self._ptr)

    def __str__(self):
        return textwrap.dedent("""\
        Movie
            file_path: {}
            mode: {}
            timing: {}
            spacing: {}
            data_type: {}\
        """.format(self.file_path, self.mode, self.timing, self.spacing, self.data_type))


class Image(object):
    """
    An image is effectively a movie with one frame and no timing.

    It is always backed by a file, which can be read or written using this class.
    See :ref:`importMovie` for details on what formats are supported for read.
    Only the native `.isxd` format is supported for write.

    Examples
    --------
    Read an existing image and get its data.

    >>> image = isx.Image.read('recording_20160613_105808-PP-PP-BP-Mean Image.isxd')
    >>> image_data = image.get_data()

    Calculate the minimum image from an existing movie and write it.

    >>> movie = isx.Movie.read('recording_20160613_105808-PP-PP.isxd')
    >>> min_image = 4095 * numpy.ones(movie.spacing.num_pixels, dtype=movie.data_type)
    >>> for i in range(movie.timing.num_samples):
    >>>     min_image = numpy.minimum(min_image, movie.get_frame_data(i))
    >>> isx.Image.write('recording_20160613_105808-PP-PP-min.isxd', movie.spacing, movie.data_type, min_image)

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    spacing : :class:`isx.Spacing`
        The spacing of the pixels in the image.
    data_type : {numpy.uint16, numpy.float32}
        The data type of each pixel.
    """

    def __init__(self):
        self._impl = isx.Movie()
        self._data = None

    @property
    def file_path(self):
        return self._impl.file_path

    @property
    def mode(self):
        return self._impl.mode

    @property
    def spacing(self):
        return self._impl.spacing

    @property
    def data_type(self):
        return self._impl.data_type

    @classmethod
    def read(cls, file_path):
        """
        Read an existing image from a file.

        Arguments
        ---------
        file_path : str
            The path of the image file to read.

        Returns
        -------
        :class:`isx.Image`
            The image that was read.
        """
        self = cls()
        self._impl = isx.Movie.read(file_path)
        if self._impl.timing.num_samples > 1:
            raise AttributeError('File has more than one frame. Use isx.Movie.read instead.')
        self._data = self._impl.get_frame_data(0)
        return self

    @classmethod
    def write(cls, file_path, spacing, data_type, data):
        """
        Write an image to a file.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        spacing : :class:`isx.Spacing`
            The spacing of the image to write.
        data_type : {numpy.uint16, numpy.float32}
            The data type of each pixel.
        data : :class:`numpy.array`
            The 2D array of data to write.

        Returns
        -------
        :class:`isx.Image`
            The image that was written.
        """
        self = cls()
        self._impl = isx.Movie.write(file_path, isx.Timing(num_samples=1), spacing, data_type)
        self._data = isx._internal.ndarray_as_type(data, np.dtype(data_type))
        self._impl.set_frame_data(0, self._data)
        self._impl.flush()
        return self

    def get_data(self):
        """
        Get the data stored in the image.

        Returns
        -------
        :class:`numpy.ndarray`
            The image data.
        """
        return self._data

    def __str__(self):
        return textwrap.dedent("""\
        Image
            file_path: {}
            mode: {}
            spacing: {}
            data_type: {}\
        """.format(self.file_path, self.mode, self.spacing, self.data_type))


class CellSet(object):
    """
    A cell set contains the image and trace data associated with components in
    a movie, such as cells or regions of interest.

    It is always backed by a file in the native `.isxd` format.

    Examples
    --------
    Read an existing cell set from a file and get the image and trace data of
    the first cell.

    >>> cell_set = isx.CellSet.read('recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA.isxd')
    >>> image_0 = cell_set.get_cell_image_data(0)
    >>> trace_0 = cell_set.get_cell_trace_data(0)

    Write a new cell set to a file with the same timing and spacing as an
    existing movie, with 3 random cell images and traces.

    >>> movie = isx.Movie.read('recording_20160613_105808-PP-PP.isxd')
    >>> cell_set = isx.CellSet.write('cell_set.isxd', movie.timing, movie.spacing)
    >>> for i in range(3):
    >>>     image = numpy.random.random(cell_set.spacing.num_pixels).astype(numpy.float32)
    >>>     trace = numpy.random.random(cell_set.timing.num_samples).astype(numpy.float32)
    >>>     cell_set.set_cell_data(i, image, trace, 'C{}'.format(i))
    >>> cell_set.flush()

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    timing : :class:`isx.Timing`
        The timing of the samples in each cell trace.
    spacing : :class:`isx.Spacing`
        The spacing of the pixels in each cell image.
    num_cells : int
        The number of cells or components.
    """

    _MAX_CELL_NAME_SIZE = 256

    def __init__(self):
        self._ptr = isx._internal.IsxCellSetPtr()

    @property
    def file_path(self):
        return self._ptr.contents.file_path.decode() if self._ptr else None

    @property
    def mode(self):
        return isx._internal.get_mode_from_read_only(self._ptr.contents.read_only) if self._ptr else None

    @property
    def timing(self):
        return isx.core.Timing._from_impl(self._ptr.contents.timing) if self._ptr else None

    @property
    def spacing(self):
        return isx.core.Spacing._from_impl(self._ptr.contents.spacing) if self._ptr else None

    @property
    def num_cells(self):
        return self._ptr.contents.num_cells if self._ptr else None

    @classmethod
    def read(cls, file_path, read_only=True):
        """
        Open an existing cell set from a file for reading.

        This is a light weight operation that simply reads the meta-data from the cell set,
        and does not read any image or trace data.

        Arguments
        ---------
        file_path : str
            The path of the file to read.
        read_only : bool
            If true, only allow meta-data and data to be read, otherwise allow some meta-data
            to be written (e.g. cell status).

        Returns
        -------
        :class:`isx.CellSet`
            The cell set that was read. Meta-data is immediately available.
            Image and trace data must be read using :func:`isx.CellSet.get_cell_image_data`
            and :func:`isx.CellSet.get_cell_trace_data` respectively.
        """
        cell_set = cls()
        isx._internal.c_api.isx_read_cell_set(file_path.encode('utf-8'), read_only, ctypes.byref(cell_set._ptr))
        return cell_set

    @classmethod
    def write(cls, file_path, timing, spacing):
        """
        Open a new cell set to a file for writing.

        This is a light weight operation. It does not write any image or trace data immediately.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        timing : :class:`isx.Timing`
            The timing of the cell set to write. Typically this comes from the movie this
            is derived from.
        spacing : :class:`isx.Spacing`
            The spacing of the movie to write. Typically this comes from the movie this is
            derived from.

        Returns
        -------
        :class:`isx.CellSet`
            The empty cell set that was written.
            Image and trace data must be written with :func:`isx.CellSet.set_cell_data`.
        """
        if not isinstance(timing, isx.core.Timing):
            raise TypeError('timing must be a Timing object')

        if not isinstance(spacing, isx.core.Spacing):
            raise ValueError('spacing must be a Spacing object')

        cell_set = cls()
        isx._internal.c_api.isx_write_cell_set(
                file_path.encode('utf-8'), timing._impl, spacing._impl, False, ctypes.byref(cell_set._ptr))
        return cell_set

    def get_cell_name(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        str
            The name of the indexed cell.
        """
        isx._internal.validate_ptr(self._ptr)
        result = ctypes.create_string_buffer(CellSet._MAX_CELL_NAME_SIZE)
        isx._internal.c_api.isx_cell_set_get_name(self._ptr, index, CellSet._MAX_CELL_NAME_SIZE, result)
        return result.value.decode('utf-8')

    def get_cell_status(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        {'accepted', 'undecided', 'rejected'}
            The status of the indexed cell as a string.
        """
        isx._internal.validate_ptr(self._ptr)
        status_int = ctypes.c_int(0)
        isx._internal.c_api.isx_cell_set_get_status(self._ptr, index, ctypes.byref(status_int))
        return isx._internal.CELL_STATUS_TO_STRING[status_int.value]

    def set_cell_status(self, index, status):
        """
        Set the status of cell. This will also flush the file.

        .. warning:: As this flushes the file, only use this after all cells have been
                     written using :func:`isx.CellSet.set_cell_data`.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.
        status : {'accepted', 'undecided', 'rejected'}
            The desired status of the indexed cell.
        """
        isx._internal.validate_ptr(self._ptr)
        if self.mode != 'w':
            raise RuntimeError('Cannot set cell status in read-only mode')
        status_int = isx._internal.lookup_enum('cell_status', isx._internal.CELL_STATUS_FROM_STRING, status)
        isx._internal.c_api.isx_cell_set_set_status(self._ptr, index, status_int)

    def get_cell_trace_data(self, index):
        """
        Get the trace data associated with a cell.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        :class:`numpy.ndarray`
            The trace data in a 1D array.
        """
        isx._internal.validate_ptr(self._ptr)
        trace = np.zeros([self.timing.num_samples], dtype=np.float32)
        trace_p = trace.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_cell_set_get_trace(self._ptr, index, trace_p)
        return trace

    def get_cell_image_data(self, index):
        """
        Get the image data associated with a cell.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        :class:`numpy.ndarray`
            The image data in a 2D array.
        """
        isx._internal.validate_ptr(self._ptr)
        f = np.zeros([np.prod(self.spacing.num_pixels)], dtype=np.float32)
        f_p = f.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_cell_set_get_image(self._ptr, index, f_p)
        return f.reshape(self.spacing.num_pixels)

    def set_cell_data(self, index, image, trace, name):
        """
        Set the image and trace data of a cell.

        Cells must be set in increasing order, otherwise this will error.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.
        image : :class:`numpy.ndarray`
            The image data in a 2D array.
        trace : :class:`numpy.ndarray`
            The trace data in a 1D array.
        name : str
            The name of the cell.
        """
        isx._internal.validate_ptr(self._ptr)

        if self.mode != 'w':
            raise RuntimeError('Cannot set cell data in read-only mode')

        if name is None:
            name = 'C{}'.format(index)

        im = isx._internal.ndarray_as_type(image.reshape(np.prod(self.spacing.num_pixels)), np.dtype(np.float32))
        im_p = im.ctypes.data_as(isx._internal.FloatPtr)
        tr = isx._internal.ndarray_as_type(trace, np.dtype(np.float32))
        tr_p = tr.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_cell_set_write_image_trace(self._ptr, index, im_p, tr_p, name.encode('utf-8'))

    def flush(self):
        """
        Flush all meta-data and cell data to the file.

        This should be called after setting all cell data of a cell set opened with :func:`isx.CellSet.write`.
        """
        isx._internal.validate_ptr(self._ptr)
        isx._internal.c_api.isx_cell_set_flush(self._ptr)

    def get_acquisition_info(self):
        """
        Get information about acquisition that may be stored in some files,
        such as nVista 3 movies and data derived from those.

        Returns
        -------
        dict
            A dictionary likely parsed from JSON that maps from string keys to variant values.
        """
        return isx._internal.get_acquisition_info(
                self._ptr,
                isx._internal.c_api.isx_cell_set_get_acquisition_info,
                isx._internal.c_api.isx_cell_set_get_acquisition_info_size);

    def __del__(self):
        if self._ptr:
            isx._internal.c_api.isx_cell_set_delete(self._ptr)

    def __str__(self):
        return textwrap.dedent("""\
        CellSet
            file_path: {}
            mode: {}
            timing: {}
            spacing: {}
            num_cells: {}\
        """.format(self.file_path, self.mode, self.timing, self.spacing, self.num_cells))


class EventSet(object):
    """
    An event set contains the event data of a number of components or cells.

    It is typically derived from a cell set after applying an event detection
    algorithm.
    Each event of a cell is comprised of a time stamp offset and a value or amplitude.

    Examples
    --------
    Read an existing event set from a file and get the event data associated with the
    first cell.

    >>> event_set = isx.EventSet.read('recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA-ED.isxd')
    >>> [offsets, amplitudes] = event_set.get_cell_data(0)

    Write a new event set to a file by applying a threshold to the traces of an existing
    cell set.

    >>> cell_set = isx.CellSet.read('recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA.isxd')
    >>> cell_names = ['C{}'.format(c) for c in range(cell_set.num_cells)]
    >>> event_set = isx.EventSet.write('recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA-custom_ED.isxd', cell_set.timing, cell_names)
    >>> offsets = numpy.array([x.to_usecs() for x in cell_set.timing.get_offsets_since_start()], numpy.uint64)
    >>> for c in range(cell_set.num_cells):
    >>>     trace = cell_set.get_cell_trace_data(c)
    >>>     above_thresh = trace > 500
    >>>     event_set.set_cell_data(c, offsets[above_thresh], trace[above_thresh])
    >>> event_set.flush()

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    timing : :class:`isx.Timing`
        The timing of the samples in each event trace.
    num_cells : int
        The number of cells or components.
    cell_dict : dict
        Dictionary mapping cell names to cell indices
    """

    def __init__(self):
        self._ptr = isx._internal.IsxEventsPtr()
        self._cell_dict = dict()

    @property
    def file_path(self):
        return self._ptr.contents.file_path.decode() if self._ptr else None

    @property
    def mode(self):
        return isx._internal.get_mode_from_read_only(self._ptr.contents.read_only) if self._ptr else None

    @property
    def timing(self):
        return isx.core.Timing._from_impl(self._ptr.contents.timing) if self._ptr else None

    @property
    def num_cells(self):
        return self._ptr.contents.num_cells if self._ptr else None

    @property
    def cell_dict(self):
        return self._cell_dict

    @classmethod
    def read(cls, file_path):
        """
        Open an existing event set from a file for reading.

        This is a light weight operation that simply reads the meta-data from the event set,
        and does not read any event data.

        Arguments
        ---------
        file_path : str
            The path of the file to read.

        Returns
        -------
        :class:`isx.EventSet`
            The event set that was read. Meta-data is immediately available.
            Event data must be read using :func:`isx.EventSet.get_cell_data`.
        """
        event_set = cls()
        isx._internal.c_api.isx_read_events(file_path.encode('utf-8'), ctypes.byref(event_set._ptr))

        # Populate cell -> index dict
        for i in range(event_set.num_cells):
            event_set._cell_dict[event_set.get_cell_name(i)] = i

        return event_set

    @classmethod
    def write(cls, file_path, timing, cell_names):
        """
        Open a new event set to a file for writing.

        This is a light weight operation. It does not write any event data immediately.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        timing : isx.Timing
            The timing of the event set to write. Typically this comes from the cell set this
            is derived from.
        cell_names : list<str>
            The names of the cells that will be written. Typically these come from the cell set
            this is derived from.

        Returns
        -------
        :class:`isx.EventSet`
            The empty event set that was written.
            Image and trace data must be written with :func:`isx.EventSet.set_cell_data`.
        """
        if not isinstance(timing, isx.core.Timing):
            raise TypeError('timing must be a Timing object')

        num_cells = len(cell_names)
        if num_cells <= 0:
            raise ValueError('cell_names must not be empty')

        cell_names_c = isx._internal.list_to_ctypes_array(cell_names, ctypes.c_char_p)
        event_set = cls()
        isx._internal.c_api.isx_write_events(file_path.encode('utf-8'), timing._impl, cell_names_c, num_cells, ctypes.byref(event_set._ptr))

        # Populate cell -> index dict
        event_set._cell_dict = { name : index for index, name in enumerate(cell_names) }

        return event_set

    def get_cell_name(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        str
            The name of the indexed cell.
        """
        isx._internal.validate_ptr(self._ptr)
        result = ctypes.create_string_buffer(CellSet._MAX_CELL_NAME_SIZE)
        isx._internal.c_api.isx_events_get_cell_name(self._ptr, index, CellSet._MAX_CELL_NAME_SIZE, result)
        return result.value.decode('utf-8')

    def get_cell_index(self, name):
        """
        Arguments
        ---------
        name : int >= 0
            The name of a cell.

        Returns
        -------
        str
            The index of the named cell.
        """
        try:
            return self._cell_dict[name]
        except KeyError:
            raise KeyError(f"Cell with name \"{name}\" does not exist.")

    def get_cell_data(self, index):
        """
        Get the event data associated with a cell.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        offsets : :class:`numpy.ndarray`
            The 1D array of time stamps offsets from the start in microseconds.
        amplitudes : :class:`numpy.ndarray`
            The 1D array of event amplitudes.
        """
        isx._internal.validate_ptr(self._ptr)

        cell_name = self.get_cell_name(index)

        num_events = ctypes.c_size_t(0)
        isx._internal.c_api.isx_events_get_cell_count(self._ptr, cell_name.encode('utf-8'), ctypes.byref(num_events))
        num_events = num_events.value

        f = np.zeros([np.prod(num_events)], dtype=np.float32)
        f_p = f.ctypes.data_as(isx._internal.FloatPtr)

        usecs = np.zeros([np.prod(num_events)], dtype=np.uint64)
        usecs_p = usecs.ctypes.data_as(isx._internal.UInt64Ptr)

        isx._internal.c_api.isx_events_get_cell(self._ptr, cell_name.encode('utf-8'), usecs_p, f_p)

        return usecs, f

    def set_cell_data(self, index, offsets, amplitudes):
        """
        Set the event data of a cell.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.
        offsets : :class:`numpy.ndarray`
            The 1D array of time stamps offsets from the start in microseconds.
        amplitudes : :class:`numpy.ndarray`
            The 1D array of event amplitudes.
        """
        isx._internal.validate_ptr(self._ptr)

        if len(offsets) != len(amplitudes):
            raise TypeError("Number of events must be the same as the number of timestamps.")

        amps = isx._internal.ndarray_as_type(amplitudes, np.dtype(np.float32))
        offs = isx._internal.ndarray_as_type(offsets, np.dtype(np.uint64))
        f_p = amps.ctypes.data_as(isx._internal.FloatPtr)
        usecs_p = offs.ctypes.data_as(isx._internal.UInt64Ptr)
        isx._internal.c_api.isx_events_write_cell(self._ptr, index, len(offs), usecs_p, f_p)

    def flush(self):
        """
        Flush all meta-data and cell data to the file.

        This should be called after setting all cell data of an event set opened with :func:`isx.EventSet.write`.
        """
        isx._internal.validate_ptr(self._ptr)
        isx._internal.c_api.isx_events_flush(self._ptr)

    def get_acquisition_info(self):
        """
        Get information about acquisition that may be stored in some files,
        such as nVista 3 movies and data derived from those.

        Returns
        -------
        dict
            A dictionary likely parsed from JSON that maps from string keys to variant values.
        """
        return isx._internal.get_acquisition_info(
                self._ptr,
                isx._internal.c_api.isx_events_get_acquisition_info,
                isx._internal.c_api.isx_events_get_acquisition_info_size);

    def __del__(self):
        if self._ptr:
            isx._internal.c_api.isx_events_delete(self._ptr)

    def __str__(self):
        return textwrap.dedent("""\
        EventSet
            file_path: {}
            mode: {}
            timing: {}
            num_cells: {}\
        """.format(self.file_path, self.mode, self.timing, self.num_cells))


class GpioSet(object):
    """
    A GPIO set contains the data recorded across a number of channels.

    Each data point is comprised of a time stamp offset and a value or amplitude.

    Examples
    --------
    Read an existing gpio set from a file and get the data associated with the first channel.

    >>> gpio_set = isx.GpioSet.read('2020-05-20-10-33-22_video.gpio')
    >>> [offsets, amplitudes] = gpio_set.get_channel_data(0)

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    timing : :class:`isx.Timing`
        The timing of the samples.
    num_channels : int
        The number of channels.
    channel_dict : dict
        Dictionary mapping channel names to channel indices
    """

    _MAX_CHANNEL_NAME_SIZE = 256

    def __init__(self):
        self._ptr = isx._internal.IsxGpioPtr()
        self._is_imu = False
        self._channel_dict = dict()

    @property
    def file_path(self):
        return self._ptr.contents.file_path.decode() if self._ptr else None

    @property
    def mode(self):
        return isx._internal.get_mode_from_read_only(self._ptr.contents.read_only) if self._ptr else None

    @property
    def timing(self):
        return isx.core.Timing._from_impl(self._ptr.contents.timing) if self._ptr else None

    @property
    def num_channels(self):
        return self._ptr.contents.num_channels if self._ptr else None

    @property
    def channel_dict(self):
        return self._channel_dict

    @classmethod
    def read(cls, file_path):
        """
        Open an existing GPIO set from a file for reading.

        This is a light weight operation that simply reads the meta-data from the GPIO set,
        and does not read any GPIO data.

        Arguments
        ---------
        file_path : str
            The path of the file to read.

        Returns
        -------
        :class:`isx.GpioSet`
            The GPIO set that was read. Meta-data is immediately available.
            GPIO data must be read using :func:`isx.GpioSet.get_channel_data`.
        """
        gpio = cls()
        isx._internal.c_api.isx_read_gpio(file_path.encode('utf-8'), ctypes.byref(gpio._ptr))

        if file_path.lower().endswith('.imu'):
            gpio._is_imu = True

        # Populate channel -> index dict
        for i in range(gpio.num_channels):
            gpio._channel_dict[gpio.get_channel_name(i)] = i

        return gpio

    def get_channel_name(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a channel.

        Returns
        -------
        str
            The name of the indexed channel.
        """
        isx._internal.validate_ptr(self._ptr)
        result = ctypes.create_string_buffer(GpioSet._MAX_CHANNEL_NAME_SIZE)
        isx._internal.c_api.isx_gpio_get_channel_name(self._ptr, index, GpioSet._MAX_CHANNEL_NAME_SIZE, result)
        return result.value.decode('utf-8')

    def get_channel_index(self, name):
        """
        Arguments
        ---------
        name : int >= 0
            The name of a channel.

        Returns
        -------
        str
            The index of the named channel.
        """
        try:
            return self._channel_dict[name]
        except KeyError:
            raise KeyError(f"Channel with name \"{name}\" does not exist.")

    def get_channel_data(self, index):
        """
        Get the data associated with a channel.

        Arguments
        ---------
        index : int >= 0
            The index of a channel.

        Returns
        -------
        offsets : :class:`numpy.ndarray`
            The 1D array of time stamps offsets from the start in microseconds.
        amplitudes : :class:`numpy.ndarray`
            The 1D array of amplitudes.
        """
        isx._internal.validate_ptr(self._ptr)

        channel_name = self.get_channel_name(index)

        num_channels = ctypes.c_size_t(0)
        isx._internal.c_api.isx_gpio_get_channel_count(self._ptr, channel_name.encode('utf-8'), ctypes.byref(num_channels))
        num_channels = num_channels.value

        f = np.zeros([np.prod(num_channels)], dtype=np.float32)
        f_p = f.ctypes.data_as(isx._internal.FloatPtr)

        usecs = np.zeros([np.prod(num_channels)], dtype=np.uint64)
        usecs_p = usecs.ctypes.data_as(isx._internal.UInt64Ptr)

        isx._internal.c_api.isx_gpio_get_channel(self._ptr, channel_name.encode('utf-8'), usecs_p, f_p)

        # Convert acceleration (first 3 channels) units to g for IMU data
        if self._is_imu and index < 3:
            f /= 16384.

        return usecs, f

    def get_acquisition_info(self):
        """
        Get information about acquisition that may be stored in some files,
        such as nVista 3 movies and data derived from those.

        Returns
        -------
        dict
            A dictionary likely parsed from JSON that maps from string keys to variant values.
        """
        return isx._internal.get_acquisition_info(
                self._ptr,
                isx._internal.c_api.isx_gpio_get_acquisition_info,
                isx._internal.c_api.isx_gpio_get_acquisition_info_size)

    def __del__(self):
        if self._ptr:
            isx._internal.c_api.isx_gpio_delete(self._ptr)

    def __str__(self):
        return textwrap.dedent("""\
        GPIO Set
            file_path: {}
            mode: {}
            timing: {}
            num_channels: {}\
        """.format(self.file_path, self.mode, self.timing, self.num_channels))


class VesselSet(object):
    """
    A vessel set contains the image, line and trace data associated with components in
    a movie, such as vessels or regions of interest.

    It is always backed by a file in the native `.isxd` format.

    A vessel set can represent two types of data: vessel diameter and rbc velocity.
    Depending on the vessel type, different information will be stored to disk.

    Note: Since blood flow algorithms apply a sliding window over input movies,
    the timing of a vessel trace is different from the timing of its input movie.
    Each frame of a vessel trace represents a measurement for a particular window sampled from its input movie.
    Relative to the input movie, each frame maps to the start of the corresponding window sampled.
    The duration of a frame is equal to the time increment of the sliding window.

    The following examples will demonstrate the types of data accessible for both types of vessel sets.

    Examples
    --------
    **Vessel Diameter**
    Read an existing vessel set from a file and get the image, line and diameter trace data of
    the first vessel.

    >>> vessel_set = isx.VesselSet.read('bloodflow_movie_10s-VD.isxd')
    >>> image_0 = vessel_set.get_vessel_image_data(0)
    >>> line_0 = vessel_set.get_vessel_line_data(0)
    >>> trace_0 = vessel_set.get_vessel_trace_data(0)
    >>> center_trace_0 = vessel_set.get_vessel_center_trace_data(0)

    Write a new vessel set to a file with the same timing and spacing as an
    existing movie, with 3 random vessel images, lines and diameter traces.

    >>> movie = isx.Movie.read('recording_20160613_105808-PP-PP.isxd')
    >>> vessel_set = isx.VesselSet.write('vessel_set.isxd', movie.timing, movie.spacing, 'vessel diameter')
    >>> for i in range(3):
    >>>     image = numpy.random.random(vessel_set.spacing.num_pixels).astype(numpy.float32)
    >>>     lines = numpy.random.randint(0, min(spacing.num_pixels), (num_vessels, 2, 2))
    >>>     trace = numpy.random.random(vessel_set.timing.num_samples).astype(numpy.float32)
    >>>     center_trace = numpy.random.random(vessel_set.timing.num_samples).astype(numpy.float32)
    >>>     vessel_set.set_vessel_diameter_data(i, image, lines, trace, center_trace, 'V{}'.format(i))
    >>> vessel_set.flush()

    **RBC Velocity**
    Read an existing vessel set from a file and get the image, line and rbc velocity trace data of
    the first vessel.

    >>> vessel_set = isx.VesselSet.read('bloodflow_movie_10s-RBCV.isxd')
    >>> image_0 = vessel_set.get_vessel_image_data(0)
    >>> line_0 = vessel_set.get_vessel_line_data(0)
    >>> trace_0 = vessel_set.get_vessel_trace_data(0)
    >>> direction_trace_0 = vessel_set.get_vessel_direction_trace_data(0)
    >>> corr_0 = vessel_set.get_vessel_correlations_data(0, 0) # look at first frame of correlation data for the first vessel

    Write a new vessel set to a file with the same timing and spacing as an
    existing movie, with 3 random vessel images, lines and rbc velocity traces.

    >>> movie = isx.Movie.read('recording_20160613_105808-PP-PP.isxd')
    >>> vessel_set = isx.VesselSet.write('vessel_set.isxd', movie.timing, movie.spacing, 'rbc velocity')
    >>> for i in range(3):
    >>>     image = numpy.random.random(vessel_set.spacing.num_pixels).astype(numpy.float32)
    >>>     lines = numpy.random.randint(0, min(spacing.num_pixels), (num_vessels, 2, 2))
    >>>     trace = numpy.random.random(vessel_set.timing.num_samples).astype(numpy.float32)
    >>>     direction_trace = numpy.random.random(vessel_set.timing.num_samples).astype(numpy.float32)
    >>>     correlation_size = np.random.randint(2, 5, size=(2,))
    >>>     correlations_trace = numpy.random.random([vessel_set.timing.num_samples, 3, correlation_size[0], correlation_size[1]]).astype(numpy.float32)
    >>>     vessel_set.set_rbc_velocity_data(i, image, lines, trace, direction_trace, correlations_trace, 'V{}'.format(i))
    >>> vessel_set.flush()

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    timing : :class:`isx.Timing`
        The timing of the samples in each vessel trace.
    spacing : :class:`isx.Spacing`
        The spacing of the pixels in each vessel image.
    num_vessels : int
        The number of vessels or components.
    """

    _MAX_VESSEL_NAME_SIZE = 256

    class VesselSetType(Enum):
        VESSEL_DIAMETER = 0
        RBC_VELOCITY = 1

        @classmethod
        def from_str(cls, type_str):
            if type_str == 'rbc velocity':
                return cls.RBC_VELOCITY
            else:
                return cls.VESSEL_DIAMETER
                
    def __init__(self):
        self._ptr = isx._internal.IsxVesselSetPtr()

    @property
    def file_path(self):
        return self._ptr.contents.file_path.decode() if self._ptr else None

    @property
    def mode(self):
        return isx._internal.get_mode_from_read_only(self._ptr.contents.read_only) if self._ptr else None

    @property
    def timing(self):
        return isx.core.Timing._from_impl(self._ptr.contents.timing) if self._ptr else None

    @property
    def spacing(self):
        return isx.core.Spacing._from_impl(self._ptr.contents.spacing) if self._ptr else None

    @property
    def num_vessels(self):
        return self._ptr.contents.num_vessels if self._ptr else None

    @classmethod
    def read(cls, file_path, read_only=True):
        """
        Open an existing vessel set from a file for reading.

        This is a light weight operation that simply reads the meta-data from the vessel set,
        and does not read any image or trace data.

        Arguments
        ---------
        file_path : str
            The path of the file to read.
        read_only : bool
            If true, only allow meta-data and data to be read, otherwise allow some meta-data
            to be written (e.g. vessel status).

        Returns
        -------
        :class:`isx.VesselSet`
            The vessel set that was read. Meta-data is immediately available.
            Image and trace data must be read using :func:`isx.VesselSet.get_vessel_image_data`
            and :func:`isx.VesselSet.get_vessel_trace_data` respectively.
        """
        vessel_set = cls()
        isx._internal.c_api.isx_read_vessel_set(file_path.encode('utf-8'), read_only, ctypes.byref(vessel_set._ptr))
        return vessel_set

    @classmethod
    def write(cls, file_path, timing, spacing, vessel_type):
        """
        Open a new vessel set to a file for writing.

        This is a light weight operation. It does not write any image or trace data immediately.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        timing : :class:`isx.Timing`
            The timing of the vessel set to write. Typically this comes from the movie this
            is derived from.
        spacing : :class:`isx.Spacing`
            The spacing of the movie to write. Typically this comes from the movie this is
            derived from.
        vessel_type : str
            The type of metric to store in the vessel set. Either 'vessel diameter' or 'rbc velocity'.

        Returns
        -------
        :class:`isx.VesselSet`
            The empty vessel set that was written.
            Image and trace data must be written with :func:`isx.VesselSet.set_vessel_data`.
        """
        if not isinstance(timing, isx.core.Timing):
            raise TypeError('timing must be a Timing object')

        if not isinstance(spacing, isx.core.Spacing):
            raise ValueError('spacing must be a Spacing object')

        vessel_set = cls()
        isx._internal.c_api.isx_write_vessel_set(
                file_path.encode('utf-8'), timing._impl, spacing._impl, cls.VesselSetType.from_str(vessel_type).value, ctypes.byref(vessel_set._ptr))
        return vessel_set

    def get_vessel_name(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a vessel.

        Returns
        -------
        str
            The name of the indexed vessel.
        """
        isx._internal.validate_ptr(self._ptr)
        result = ctypes.create_string_buffer(VesselSet._MAX_VESSEL_NAME_SIZE)
        isx._internal.c_api.isx_vessel_set_get_name(self._ptr, index, VesselSet._MAX_VESSEL_NAME_SIZE, result)
        return result.value.decode('utf-8')

    def get_vessel_status(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a vessel.

        Returns
        -------
        {'accepted', 'undecided', 'rejected'}
            The status of the indexed vessel as a string.
        """
        isx._internal.validate_ptr(self._ptr)
        status_int = ctypes.c_int(0)
        isx._internal.c_api.isx_vessel_set_get_status(self._ptr, index, ctypes.byref(status_int))
        return isx._internal.VESSEL_STATUS_TO_STRING[status_int.value]

    def set_vessel_status(self, index, status):
        """
        Set the status of vessel. This will also flush the file.

        .. warning:: As this flushes the file, only use this after all vessels have been
                     written using :func:`isx.VesselSet.set_vessel_data`.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.
        status : {'accepted', 'undecided', 'rejected'}
            The desired status of the indexed vessel.
        """
        isx._internal.validate_ptr(self._ptr)
        if self.mode != 'w':
            raise RuntimeError('Cannot set vessel status in read-only mode')
        status_int = isx._internal.lookup_enum('vessel_status', isx._internal.VESSEL_STATUS_FROM_STRING, status)
        isx._internal.c_api.isx_vessel_set_set_status(self._ptr, index, status_int)

    def get_vessel_trace_data(self, index):
        """
        Get the trace data associated with a vessel.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.

        Returns
        -------
        :class:`numpy.ndarray`
            The trace data in a 1D array.
        """
        isx._internal.validate_ptr(self._ptr)
        trace = np.zeros([self.timing.num_samples], dtype=np.float32)
        trace_p = trace.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_vessel_set_get_trace(self._ptr, index, trace_p)
        return trace

    def get_vessel_image_data(self, index):
        """
        Get the image data associated with a vessel.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.

        Returns
        -------
        :class:`numpy.ndarray`
            The image data in a 2D array.
        """
        isx._internal.validate_ptr(self._ptr)
        image = np.zeros([np.prod(self.spacing.num_pixels)], dtype=np.float32)
        image_p = image.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_vessel_set_get_image(self._ptr, index, image_p)
        return image.reshape(self.spacing.num_pixels)
    
    def get_vessel_line_data(self, index):
        """
        Get the line data associated with a vessel.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.

        Returns
        -------
        :class:`numpy.ndarray`
            The line data in a 2D array.
        """
        isx._internal.validate_ptr(self._ptr)
        # Handle vessel diameter vessel set
        if (self.get_vessel_set_type() == self.VesselSetType.VESSEL_DIAMETER):
            line = np.zeros(4, dtype=np.int64)
            line_p = line.ctypes.data_as(isx._internal.Int64Ptr)
            isx._internal.c_api.isx_vessel_set_get_line_endpoints(self._ptr, index, line_p)
            return line.reshape(2,2)
        # Handle rbc velocity vessel set
        elif (self.get_vessel_set_type() == self.VesselSetType.RBC_VELOCITY):
            line = np.zeros(8, dtype=np.int64)
            line_p = line.ctypes.data_as(isx._internal.Int64Ptr)
            isx._internal.c_api.isx_vessel_set_get_line_endpoints(self._ptr, index, line_p)
            return line.reshape(4,2)

    def get_vessel_set_type(self):
        """
        Get the type of the vessel set.

        Returns
        -------
        VesselSetType Enum
            An enum representation of the different vessel set types.
            either VesselSetType.VESSEL_DIAMETER or VesselSetType.RBC_VELOCITY
        """
        isx._internal.validate_ptr(self._ptr)
        type_int = ctypes.c_int(-1)
        isx._internal.c_api.isx_vessel_set_get_type(self._ptr, ctypes.byref(type_int))
        return self.VesselSetType(type_int.value)

    def get_vessel_center_trace_data(self, index):
        """
        Get the center trace data associated with a vessel.
        This represents an index on the user drawn line where the center of the diamter was estimated to be.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.

        Returns
        -------
        :class:`numpy.ndarray`
            The center trace data in a 1D array.
            If no center traces are stored in the file, the function will throw an error
        """
        isx._internal.validate_ptr(self._ptr)
        trace = np.zeros([self.timing.num_samples], dtype=np.float32)
        trace_p = trace.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_vessel_set_get_center_trace(self._ptr, index, trace_p)
        return trace

    def get_vessel_direction_trace_data(self, index):
        """
        Get the direction trace data associated with a vessel
        This is the direction component of each velocity measurement reported in degrees relative to positive x-axis.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.

        Returns
        -------
        :class:`numpy.ndarray`
            The direction trace data in a 1D array.
            If no direction traces are stored in the file, the function will throw an error
        """
        isx._internal.validate_ptr(self._ptr)
        trace = np.zeros([self.timing.num_samples], dtype=np.float32)
        trace_p = trace.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_vessel_set_get_direction_trace(self._ptr, index, trace_p)
        return trace

    def has_correlation_heatmaps(self):
        """
        If true, cross-correlation heatmaps for rbc velocity measurements are stored in this file.

        Returns
        -------
        bool
            Flag indicating whether heatmaps were saved.
        """
        isx._internal.validate_ptr(self._ptr)
        saved_ptr = ctypes.c_int(0)
        isx._internal.c_api.isx_vessel_set_is_correlation_saved(self._ptr, ctypes.byref(saved_ptr))
        return bool(saved_ptr.value)

    def get_vessel_correlations_data(self, index, frame):
        """
        Get the correlation trace data associated with a vessel at a certain frame.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.
        frame : int >= 0
            The frame index of the time-series trace.

        Returns
        -------
        :class:`numpy.ndarray`
            The correlations data in a 3D array of size (3, width, height)
            Each slice of the ndarray is a correlation heatmap for a particular temporal offset
            The mapping of temporal offsets is (slice 0 -> t = -1, slice 1 -> t = 0, slice 2 = t = 1)
            If no correlations are stored in the file, the function will throw an error
        """
        isx._internal.validate_ptr(self._ptr)

        correlation_size = np.zeros([2], dtype=np.uint64)
        correlation_size_p = correlation_size.ctypes.data_as(isx._internal.SizeTPtr)
        isx._internal.c_api.isx_vessel_set_get_correlation_size(self._ptr, index, correlation_size_p);
        
        height, width = correlation_size[0], correlation_size[1]
        correlations = np.zeros([int(3 * height *width)], dtype=np.float32)
        correlations_p = correlations.ctypes.data_as(isx._internal.FloatPtr)

        isx._internal.c_api.isx_vessel_set_get_correlations(self._ptr, index, frame, correlations_p)

        return correlations.reshape((3, height, width))

    def set_vessel_diameter_data(self, index, image, line, trace, center_trace, name=None):
        """
        Set the image, line and diameter trace data of a vessel

        Vessels must be set in increasing order, otherwise this will error.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.
        image : :class:`numpy.ndarray`
            The image data in a 2D array.
        line : :class:`numpy.ndarray`
            The line endpoint data in a 2D array.
        trace : :class:`numpy.ndarray`
            The trace data in a 1D array.
        center_trace : :class:`numpy.ndarray`
            The center trace data in a 1D array.
        name : str
            The name of the vessel.
        """
        isx._internal.validate_ptr(self._ptr)

        if self.mode != 'w':
            raise RuntimeError('Cannot set vessel data in read-only mode')

        if name is None:
            name = 'V{}'.format(index)

        im = isx._internal.ndarray_as_type(image.reshape(np.prod(self.spacing.num_pixels)), np.dtype(np.float32))
        im_p = im.ctypes.data_as(isx._internal.FloatPtr)

        ln = isx._internal.ndarray_as_type(line, np.dtype(np.int64)) 
        ln_p = ln.ctypes.data_as(isx._internal.Int64Ptr)

        tr = isx._internal.ndarray_as_type(trace, np.dtype(np.float32))
        tr_p = tr.ctypes.data_as(isx._internal.FloatPtr)

        cen_tr = isx._internal.ndarray_as_type(center_trace, np.dtype(np.float32))
        cen_tr_p = cen_tr.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_vessel_set_write_vessel_diameter_data(self._ptr, index, im_p, ln_p, tr_p, cen_tr_p, name.encode('utf-8'))

    def set_rbc_velocity_data(self, index, image, line, trace, direction_trace, correlations_trace=None, name=None):
        """
        Set the image, line and rbc velocity trace data of a vessel.

        Vessels must be set in increasing order, otherwise this will error.

        Arguments
        ---------
        index : int >= 0
            The index of a vessel.
        image : :class:`numpy.ndarray`
            The image data in a 2D array.
        line : :class:`numpy.ndarray`
            The line endpoint data in a 2D array.
        trace : :class:`numpy.ndarray`
            The trace data in a 1D array.
        direction_trace : :class:`numpy.ndarray`
            The direction trace data in a 1D array.
        correlations_trace : :class:`numpy.ndarray`
            The correlations trace data in a 4D array T x 3 x W x H
            Where T is the number of time samples
            W is the width of the correlation heatmap
            H is the width of the correlation heatmap
            There are three heatmaps for each time sample representing the three temporal offsets (-1, 0, 1)
        name : str
            The name of the vessel.
        """
        isx._internal.validate_ptr(self._ptr)

        if self.mode != 'w':
            raise RuntimeError('Cannot set vessel data in read-only mode')

        if name is None:
            name = 'V{}'.format(index)

        im = isx._internal.ndarray_as_type(image.reshape(np.prod(self.spacing.num_pixels)), np.dtype(np.float32))
        im_p = im.ctypes.data_as(isx._internal.FloatPtr)

        ln = isx._internal.ndarray_as_type(line, np.dtype(np.int64)) 
        ln_p = ln.ctypes.data_as(isx._internal.Int64Ptr)

        tr = isx._internal.ndarray_as_type(trace, np.dtype(np.float32))
        tr_p = tr.ctypes.data_as(isx._internal.FloatPtr)

        dir_tr = isx._internal.ndarray_as_type(direction_trace, np.dtype(np.float32))
        dir_tr_p = dir_tr.ctypes.data_as(isx._internal.FloatPtr)

        corr_tr, corr_tr_p = None, None
        corr_size = [0, 0]
        if correlations_trace is not None:
            corr_tr = isx._internal.ndarray_as_type(correlations_trace, np.dtype(np.float32))
            corr_tr_p = corr_tr.ctypes.data_as(isx._internal.FloatPtr)
            corr_size = [correlations_trace.shape[3], correlations_trace.shape[2]]

        isx._internal.c_api.isx_vessel_set_write_rbc_velocity_data(self._ptr, index, im_p, ln_p, tr_p, dir_tr_p, corr_size[0], corr_size[1], corr_tr_p, name.encode('utf-8'))

    def flush(self):
        """
        Flush all meta-data and vessel data to the file.

        This should be called after setting all vessel data of a vessel set opened with :func:`isx.VesselSet.write`.
        """
        isx._internal.validate_ptr(self._ptr)
        isx._internal.c_api.isx_vessel_set_flush(self._ptr)

    def get_acquisition_info(self):
        """
        Get information about acquisition that may be stored in some files,
        such as nVista 3 movies and data derived from those.

        Returns
        -------
        dict
            A dictionary likely parsed from JSON that maps from string keys to variant values.
        """
        return isx._internal.get_acquisition_info(
                self._ptr,
                isx._internal.c_api.isx_vessel_set_get_acquisition_info,
                isx._internal.c_api.isx_vessel_set_get_acquisition_info_size);

    def __del__(self):
        if self._ptr:
            isx._internal.c_api.isx_vessel_set_delete(self._ptr)

    def __str__(self):
        return textwrap.dedent("""\
        VesselSet
            file_path: {}
            mode: {}
            timing: {}
            spacing: {}
            num_vessels: {}\
        """.format(self.file_path, self.mode, self.timing, self.spacing, self.num_vessels))

def convert_type_numpy_array(array, dtype=np.uint16, keep_0_to_1=False):
    """ Convert a numpy array to a different data type by normalizing and mapping."""

    if np.nanmax(array) - np.nanmin(array) != 0:
        scaled_array = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
    else:
        scaled_array = array.copy()

    if np.issubdtype(dtype, np.integer):
        scaled_array *= np.iinfo(dtype).max
    elif not keep_0_to_1:
        scaled_array *= np.finfo(dtype).max

    # if converting from complex to real type, drop imaginary component
    if np.iscomplexobj(array) and not np.issubdtype(dtype, np.dtype(complex)):
        scaled_array = np.real(scaled_array)

    return scaled_array.astype(dtype)


def export_image_to_tiff(image, tiff_file, write_rgb=False):
    """Save an image as a uint16 tiff."""
    if write_rgb:
        image_data = convert_type_numpy_array(image, np.uint8)
        image_out = PIL.Image.fromarray(image_data)
    else:
        image_data = convert_type_numpy_array(image, np.uint16)
        image_out = PIL.Image.fromarray(image_data)
    image_out.save(tiff_file)


def _normalize_image(cell_image):
    """Map an image to the scale [0, 1]. """

    lower_limit = np.nanpercentile(cell_image, 0)
    higher_limit = np.nanpercentile(cell_image, 100)

    cell_image_f32 = cell_image.astype('float32')
    limits = np.array((lower_limit, higher_limit)).astype('float32')

    norm_image = (cell_image_f32 - limits[0]) / np.diff(limits)
    #norm_image[norm_image < 0] = 0
    #norm_image[norm_image > 1] = 1
    norm_image[np.less(norm_image, 0, where=~np.isnan(norm_image))] = 0
    norm_image[np.greater(norm_image, 1, where=~np.isnan(norm_image))] = 1

    return norm_image


def _get_footprint_list(cellset_file, selected_cell_statuses=['accepted']):
    """ Get a list of valid footprint images from a cellset."""
    cellset = isx.io.CellSet.read(cellset_file)

    # cells with non-NaN values are valid
    valid_indices = [index for index in range(cellset.num_cells) if not all(np.isnan(cellset.get_cell_trace_data(index)))]
    # select cells that have the selected status
    selected_indices = [index for index in valid_indices if cellset.get_cell_status(index) in selected_cell_statuses]

    cell_images = []
    n_cells = len(selected_indices)
    for i in range(n_cells):
        cell_index = selected_indices[i]
        cell_image = cellset.get_cell_image_data(cell_index)
        cell_name = cellset.get_cell_name(cell_index)
        cell_images.append(cell_image)

    return cell_images


def _isxd_cell_set_to_cell_map(cell_set, selected_cell_statuses=['accepted'],
                               cell_normalization=True, cell_thresh=None, binary=False,
                               footprint_min_var=1e-4):
    """Generate cell map as a max projection of the cell footprints of an isxd cell set.

    Arguments
    ---------
    cell_set : str
        Path to .isxd cell set file.
    selected_cell_statuses : list<str>
        a list of cell statuses for decision criteria to keep cell footprints.
        Possible list values are 'accepted', 'undecided', 'rejected'.
    cell_normalization : Bool
        If true, each cell will be normalized to [0, 1].
    cell_thresh : float [0, 1]
        Pixels with values lower than cell_thresh will be set to 0.
    binary : Bool
        If true, pixels with values above cell_thresh are set to 1. (Pixels with values
        below cell_thresh are set to 0.)
    footprint_min_var: float
        Minimum variance of an individual cell footprint. Footprints with variance
        below this threshold are not included in the cell map.

    Return Type
    -----------
    numpy.ndarray
        cell map as a max projection of the cell footprints.
    """
    footprint_list = _get_footprint_list(cell_set, selected_cell_statuses=selected_cell_statuses)

    if not footprint_list:
        raise ValueError('There are no cells to create a cell map with! Only selected cells will be used for the cell map.')

    cell_footprints = np.stack(footprint_list, axis=2)
    cell_footprints = np.moveaxis(cell_footprints, 2, 0)

    n_cells = cell_footprints.shape[0]
    cell_map = None
    for index in range(n_cells):
        cell_image = cell_footprints[index, :, :].copy()

        if cell_normalization:
            cell_image = _normalize_image(cell_image)

        # skip if footprint pixel values lack variability - indicates absence of cell in image
        if np.var(cell_image) < footprint_min_var:
            continue

        if cell_thresh is not None:
            cell_image_thresh = cell_thresh * np.nanmax(cell_image)
            cell_image[cell_image < cell_image_thresh] = 0.

        if binary:
            cell_image = (cell_image >= cell_image_thresh).astype(np.float32)

        if cell_map is None: # first index
            cell_map = cell_image
        else:
            np.fmax(cell_map, cell_image, out=cell_map)

    return cell_map


def export_image_to_isxd_tiff(image, isxd_file, tiff_file, rgb=None):
    """Save an image as isxd and tiff.
    If rgb is one of 'red', 'green', 'blue', the images will be colored.
    """
    spacing = isx.core.Spacing(num_pixels=image.shape)

    if image.dtype == np.float32:
        dtype = np.float32
    elif image.dtype == np.uint16:
        dtype = np.uint16
    elif image.dtype == np.float64:
        image = convert_type_numpy_array(image, np.float32)
        dtype = np.float32
    elif image.dtype == bool:
        image = image.astype(np.uint16)
        dtype = np.uint16
    else:
        image = convert_type_numpy_array(image, np.uint16)
        dtype = np.uint16

    if isxd_file is not None:
        Image.write(isxd_file, spacing, dtype, image)

    if tiff_file is not None:
        if rgb is not None:
            color_array = ['red', 'green', 'blue']
            if rgb not in color_array:
                raise ValueError('Value {} for rgb not one of red, green, blue'.format(rgb))

            color = color_array.index(rgb)
            image = convert_type_numpy_array(image, np.uint16)
            image_rgb = np.full([image.shape[0], image.shape[1], 3], 0, dtype=np.uint16)
            image_rgb[:, :, color] = image

            export_image_to_tiff(image_rgb, tiff_file, write_rgb=True)
        else:
            export_image_to_tiff(image, tiff_file)


def export_isxd_image_to_tiff(input_isxd_image, output_tiff_file):
    """ Convert an ISXD image file to a tiff image. 
    WARNING: does not export tiff with alignment metadata, use align_image instead

    Arguments
    ---------
    input_isxd_image : str
        Path to the input ISXD image file.
    output_tiff_file : str
        Path to the output tiff file.
    """

    if not os.path.exists(input_isxd_image):
        raise FileNotFoundError('ISXD image not found: {}'.format(input_isxd_image))

    image_data = isx.Image.read(input_isxd_image).get_data()
    PIL.Image.fromarray(isx.convert_type_numpy_array(image_data)).save(output_tiff_file)


def export_tiff_to_isxd_image(input_tiff_file, output_isxd_file):
    """ Convert an tiff image file to an ISXD image. 
    WARNING: does not export tiff with alignment metadata, use align_image instead

    Arguments
    ---------
    input_tiff_file : str
        Path to the input tiff file.
    output_isxd_file : str
        Path to the output ISXD image file.

    """
    if not os.path.exists(input_tiff_file):
        raise FileNotFoundError('TIFF file not found: {}'.format(input_tiff_file))

    tiff_data = PIL.Image.open(input_tiff_file)
    image = np.array(tiff_data)
    spacing = isx.core.Spacing(num_pixels=image.shape)

    if image.dtype == np.float32:
        dtype = np.float32
    elif image.dtype == np.uint16:
        dtype = np.uint16
    elif image.dtype == np.float64:
        image = isx.convert_type_numpy_array(image, np.float32)
        dtype = np.float32
    elif image.dtype == bool:
        image = image.astype(np.uint16)
        dtype = np.uint16
    else:
        image = isx.convert_type_numpy_array(image, np.uint16)
        dtype = np.uint16

    isx.Image.write(output_isxd_file, spacing, dtype, image)  


def _overlay_isxd_images(*isxd_images):
    """Max project the isxd images and return a numpy array."""
    max_image = None
    for isxd_image in isxd_images:
        image = Image.read(isxd_image).get_data()
        if max_image is None:
            max_image = image
        else:
            max_image = np.maximum(max_image, image)

    return max_image


def create_cell_map(isxd_cellset_file, selected_cell_statuses=['accepted'],
                    cell_thresh=0.3, binary=False, rgb=None,
                    output_isxd_cell_map_file=None, output_tiff_cell_map_file=None):
    """Generate cell map from an .isxd cellset file, saving the image as isxd and tiff.

    Arguments
    ---------
    isxd_cellset_file : str
        Path to an .isxd cellset file.
    selected_cell_statuses : list<str>
        a list of cell statuses for decision criteria to keep cell footprints.
        Possible list values are 'accepted', 'undecided', 'rejected'.
    cell_thresh : float [0, 1]
        Pixels with values lower than cell_thresh will be set to 0.
    binary : Bool
        If true, pixels with values above cell_thresh are set to 1. (Pixels with values
        below cell_thresh are set to 0.)
    rgb : one of "red", "blue", or "green"
        Color for the cell map.
    output_isxd_cell_map_file : str
        Path to the output isxd cell map image. If not given, will not be generated.
    output_tiff_cell_map_file : str
        Path to the output tiff cell map image. If not given, will not be generated.
    """
    if not os.path.exists(isxd_cellset_file):
        raise FileNotFoundError('ISXD cellset not found: {}'.format(isxd_cellset_file))

    for file_name in [output_isxd_cell_map_file, output_tiff_cell_map_file]:
        if file_name and os.path.exists(file_name):
            raise FileExistsError('Output file already exists: {}'.format(file_name))

    cell_map = _isxd_cell_set_to_cell_map(isxd_cellset_file,
                                          selected_cell_statuses=selected_cell_statuses,
                                          cell_thresh=cell_thresh, binary=binary)
    cell_map = convert_type_numpy_array(cell_map, np.uint16)
    export_image_to_isxd_tiff(cell_map, output_isxd_cell_map_file, output_tiff_cell_map_file, rgb=rgb)


def overlay_cellmaps(first_tiff_cellmap_file, second_tiff_cellmap_file, overlayed_tiff_cellmap_file, 
                     background_color='#000000', first_color='#00ff00', second_color='#ff00ff', cell_thresh=0.5):
    """ Overlay two cellmaps using different colors to show overlap.

    Arguments
    ---------
    first_tiff_cellmap_file : str
        Path to the first tiff cellmap image.
    second_tiff_cellmap_file : str
        Path to the second tiff cellmap image.
    overlayed_tiff_cellmap_file : str
        Path to the output tiff cellmap.
    background_color : str
        Hex color code for background. Format: #RRGGBB
    first_color : str
        Hex color code for cells in first cellmap. Format: #RRGGBB
    second_color : str
        Hex color code for cells in second cellmap. Format: #RRGGBB
    cell_thresh : float [0, 1]
        Pixel values less than cell_thresh will be considered as the background. 
    """
    for input_file in [first_tiff_cellmap_file, second_tiff_cellmap_file]:
        if not os.path.exists(input_file):
            raise FileNotFoundError("Input file not found: {}".format(input_file))

    first_image = PIL.Image.open(first_tiff_cellmap_file)
    second_image = PIL.Image.open(second_tiff_cellmap_file)

    # convert images to 8 bit grayscale
    first_image = PIL.Image.fromarray(isx.convert_type_numpy_array(first_image, dtype=np.uint8)).convert('L')
    second_image = PIL.Image.fromarray(isx.convert_type_numpy_array(second_image, dtype=np.uint8)).convert('L')

    if first_image.size != second_image.size:
        raise ValueError('The two images do not have the same size: {} vs {}'.format(first_image.size, second_image.size))

    # get rgb tuples of selected colors
    pil_bg_color     = PIL.ImageColor.getcolor(background_color, "RGB")
    pil_first_color  = PIL.ImageColor.getcolor(first_color, "RGB")
    pil_second_color = PIL.ImageColor.getcolor(second_color, "RGB")

    # create boolean mask - True for lighter colors (cells), False for darker colors (no cell)
    first_mask = np.array(first_image) >= 255 * cell_thresh
    second_mask = np.array(second_image) >= 255 * cell_thresh

    overlayed_arr = np.empty(first_mask.shape + (3,), dtype=np.uint8)

    # assign appropriate colors depending on where cells are and intersect
    overlayed_arr[np.logical_and(np.logical_not(first_mask), np.logical_not(second_mask))] = pil_bg_color   # no cell
    overlayed_arr[np.logical_and(first_mask, np.logical_not(second_mask))] = pil_first_color                # first cell map only
    overlayed_arr[np.logical_and(np.logical_not(first_mask), second_mask)] = pil_second_color               # second cell map only
    overlayed_arr[np.logical_and(first_mask, second_mask)] = np.maximum(pil_first_color, pil_second_color)  # both cell maps
    
    isx.export_image_to_tiff(overlayed_arr, overlayed_tiff_cellmap_file, write_rgb=True)


def overlay_cell_map_on_image(input_isxd_cell_map_file, input_isxd_image_file, output_tiff_image_file):
    """Overlay a cellmap onto an image, and save as isxd. Can save tiff as well.

    Arguments
    ---------
    input_isxd_cell_map_file : str
        Path to an .isxd cell map image file.
    input_isxd_image_file : str
        Path to an .isxd image file to overlay on.
    output_tiff_image_file : str
        Path to the output isxd cell map image.
    """
    if not os.path.exists(input_isxd_cell_map_file):
        raise FileNotFoundError('ISXD cell map not found: {}'.format(input_isxd_cell_map_file))
    if not os.path.exists(input_isxd_image_file):
        raise FileNotFoundError('ISXD image not found: {}'.format(input_isxd_image_file))

    if os.path.exists(output_tiff_image_file):
        raise FileExistsError('Output file already exists: {}'.format(output_tiff_image_file))

    overlayed_image = _overlay_isxd_images(input_isxd_image_file, input_isxd_cell_map_file)
    export_image_to_isxd_tiff(overlayed_image, None, output_tiff_image_file)


def export_movie_to_tiff(input_movie_files, output_tiff_file, write_invalid_frames=False):
    """
    Export movies to a TIFF file.

    For more details see :ref:`exportMovie`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to be exported.
    output_tiff_file : str
        The path of the TIFF file to be written.
    write_invalid_frames : bool
        If True, write invalid (dropped, cropped, and blank) frames as zero,
        otherwise, do not write them at all.
    """
    num_movies, in_movie_arr = isx._internal.check_input_files(input_movie_files)
    isx._internal.c_api.isx_export_movie_tiff(num_movies, in_movie_arr, output_tiff_file.encode('utf-8'), write_invalid_frames)


def export_movie_to_nwb(
        input_movie_files, output_nwb_file,
        identifier='', session_description='', comments='',
        description='', experiment_description='', experimenter='',
        institution='', lab='', session_id=''):
    """
    Export movies to an HDF5-based neurodata without borders (NWB) file.

    For more details see :ref:`exportMovie`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to be exported.
    output_nwb_file : str
        The path of the NWB file to be written.
    identifier : str
        An identifier for the file according to the NWB spec.
    session_description : str
        A session description for the file according to the NWB spec.
    comments : str
        Comments on the recording session.
    description : str
        Description for the file according to the NWB spec.
    experiment_description : str
        Details about the experiment.
    experimenter : str
        The person who recorded the data.
    institution : str
        The place where the recording was performed.
    lab : str
        The lab where the recording was performed.
    session_id : str
        A unique session identifier for the recording.
    """
    num_movies, in_movie_arr = isx._internal.check_input_files(input_movie_files)
    isx._internal.c_api.isx_export_movie_nwb(
            num_movies, in_movie_arr, output_nwb_file.encode('utf-8'),
            identifier.encode('utf-8'), session_description.encode('utf-8'),
            comments.encode('utf-8'), description.encode('utf-8'),
            experiment_description.encode('utf-8'), experimenter.encode('utf-8'),
            institution.encode('utf-8'), lab.encode('utf-8'), session_id.encode('utf-8'))


def export_movie_to_mp4(
    input_movie_files,
    output_mp4_file,
    compression_quality=0.1,
    write_invalid_frames=False,
    frame_rate_format="float",
    draw_bounding_box=True,
    draw_bounding_box_center=True,
    draw_zones=True
):
    """
    Export movies to an MP4 file.

    For more details see :ref:`exportMovie`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to be exported.
    output_mp4_file : str
        The path of the MP4 file to be written.
    compression_quality : float
        A value between 0.001 and 1 that controls the quality and size of the output file for the MP4 format.
        The larger the value, the better the quality of the movie, but the larger the size of the file.
        The default value of 0.1 typically produces a good quality movie where the file is at most 10% of the uncompressed original file size,
        but may be smaller if the movie content can be efficiently encoded.
        More formally, this represents a rough maximum on the output file size as a fraction of the original file size.
    write_invalid_frames : bool
        If True, write invalid (dropped, cropped, and blank) frames as zero,
        otherwise, do not write them at all.
    frame_rate_format : {"float", "int"}
        Format to encode the frame rate in the output mp4 file.
        If float, the frame rate will be exported as a precise estimate of the input movie sampling rate.
        If int, the frame rate will be rounded to the nearest integer.
    draw_bounding_box : bool
        Only used for nVision `.isxb` movies.
        If there is nVision tracking data in the `.isxb` movie, and this flag is enabled,
        then draw the bounding box estimate on each frame of the exported mp4 movie.
    draw_bounding_box_center : bool
        Only used for nVision `.isxb` movies.
        If there is nVision tracking data in the `.isxb` movie, and this flag is enabled,
        then draw the center of the bounding box estimate on each frame of the exported mp4 movie.
    draw_zones : bool
        Only used for nVision `.isxb` movies.
        If there is nVision tracking data in the `.isxb` movie, and this flag is enabled,
        then draw the zones of the tracking area on each frame of the exported mp4 movie.
    """
    num_movies, in_movie_arr = isx._internal.check_input_files(input_movie_files)

    frame_rate_format_map = {'float' : 0, 'int' : 1}
    if not frame_rate_format in frame_rate_format_map.keys():
        raise ValueError('Invalid frame rate format. Valid frame rate formats include: {}'.format(*frame_rate_format_map.keys()))
    
    isx._internal.c_api.isx_export_movie_mp4(
        num_movies,
        in_movie_arr,
        output_mp4_file.encode('utf-8'),
        compression_quality,
        write_invalid_frames,
        frame_rate_format_map[frame_rate_format],
        draw_bounding_box,
        draw_bounding_box_center,
        draw_zones
    )

def export_movie_timestamps_to_csv(input_movie_files, output_csv_file, time_ref='start'):
    """
    Export movie frame timestamps to a csv file.
    This operation is supported for .isxd and .isxb movies.

    For more details see :ref:`exportMovie`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to export frame timestamps from.
    output_csv_file : str
        The path of the csv file to be written.
    time_ref : {'start', 'unix', 'tsc'}
        The time reference for the CSV time stamps.
        If 'start' is used, the timestamps represent the seconds since the start of the movie.
        If 'unix' is used, the timestamps represent the seconds since the Unix epoch.
        If 'tsc' is used, the timestamps represent the hardware counter value on the acquisition box when each frame was captured.
    """
    num_movies, in_movie_arr = isx._internal.check_input_files(input_movie_files)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)
    isx._internal.c_api.isx_export_movie_timestamps_to_csv(
        num_movies, in_movie_arr, output_csv_file.encode('utf-8'), time_ref_int)

def export_nvision_movie_tracking_frame_data_to_csv(input_movie_files, output_csv_file, time_ref='start'):
    """
    Export frame tracking metadata from an nVision movie.
    This operation is supported for .isxb movies.


    The frame tracking metadata is generated by the nVision tracking model, and includes the following columns:
        * Global Frame Number: The frame number in the input movie series.
        * Movie Number: The movie number in the series.
        * Local Frame Number: The frame number in the individual movie.
        * Frame Timestamp: The frame timestamp. In units of seconds or microseconds based on the input `time_ref` parameter.
        * Bounding Box Left: X coordinate of the top left corner of the bounding box. In units of pixels.
        * Bounding Box Top: Y coordinate of the top left corner of the bounding box. In units of pixels.
        * Bounding Box Right: X coordinate of the bottom right corner of the bounding box. In units of pixels.
        * Bounding Box Bottom: Y coordinate of the bottom right corner of the bounding box. In units of pixels.
        * Bounding Box Center X: X coordinate of the center point of the bounding box. In units of pixels.
        * Bounding Box Center Y: Y coordinate of the center point of the bounding box. In units of pixels.
        * Confidence: The nVision tracking model confidence of the bounding box estimate. In units of %.
        * Zone ID: If the nVision tracking model detected the mouse was inside a zone, this column contains the id of that zone.
        * Zone Name: If the nVision tracking model detected the mouse was inside a zone, this column contains the name of that zone.
    
    For more details see :ref:`exportMovie`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to export frame tracking metadata from.
    output_csv_file : str
        The path of the csv file to be written.
    time_ref : {'start', 'unix', 'tsc'}
        The time reference for the CSV time stamps.
        If 'start' is used, the timestamps represent the seconds since the start of the movie.
        If 'unix' is used, the timestamps represent the seconds since the Unix epoch.
        If 'tsc' is used, the timestamps represent the hardware counter value on the acquisition box when each frame was captured.
    """

    num_movies, in_movie_arr = isx._internal.check_input_files(input_movie_files)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)
    isx._internal.c_api.isx_export_nvision_movie_tracking_frame_data_to_csv(
        num_movies, in_movie_arr, output_csv_file.encode('utf-8'), time_ref_int)


def export_nvision_movie_tracking_zone_data_to_csv(input_movie_files, output_csv_file):
    """
    Export zone tracking metadata from an nVision movie.
    This operation is supported for .isxb movies.

    The zone tracking metadata used by the nVision tracking model, and includes the following columns:
        * ID: Unique identifier for the zone.
        * Enabled: Flag indicating whether the zone is enabled for tracking.
        * Name: User friendly name for the zone.
        * Description: Optional description for the zone
        * Type: The shape of the zone. Can be either rectangle, polygon, or ellipse.
        * X `i`: The ith X coordinate of the zone.
            Note: Since zones can have different shapes, they can have a different number of coordinates.
            The csv output will contain `n` columns for the X coordinate, where `n` is the maxmimum
            number of coordinates across all zones in the metadata.
        * Y `i`: The ith Y coordinate of the zone.
            Note: Since zones can have different shapes, they can have a different number of coordinates.
            The csv output will contain `n` columns for the Y coordinate, where `n` is the maxmimum
            number of coordinates across all zones in the metadata.
        * Major Axis: only outputted for ellipse shaped zones. Length of the major axis.
        * Minor Axis: only outputted for ellipse shaped zones. Length of the minor axis.
        * Angle: only outputted for ellipse shaped zones. Ellipse rotation angle in degrees.

    For more details see :ref:`exportMovie`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to export frame tracking metadata from.
    output_csv_file : str
        The path of the csv file to be written.
    """

    num_movies, in_movie_arr = isx._internal.check_input_files(input_movie_files)
    isx._internal.c_api.isx_export_nvision_movie_tracking_zone_data_to_csv(
        num_movies, in_movie_arr, output_csv_file.encode('utf-8'))

def export_cell_set_to_csv_tiff(input_cell_set_files, output_csv_file, output_tiff_file, time_ref='start', output_props_file=''):
    """
    Export cell sets to a CSV file with trace data and TIFF files with image data.

    For more details see :ref:`exportCellsAndStuff`.

    Unlike the desktop application, this will only produce a TIFF cell map image file
    and not a PNG file too.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell sets to export.
    output_csv_file : str
        The path of the CSV file to write.
    output_tiff_file : str
        The base name of the TIFF files to write.
    time_ref : {'start', 'unix'}
        The time reference for the CSV time stamps.
        If 'start' is used, the time stamps represent the seconds since the start of the cell set.
        If 'unix' is used, the time stamps represents the second since the Unix epoch.
    output_props_file : str
        The path of the properties CSV file to write.
    """
    num_cell_sets, in_cell_sets = isx._internal.check_input_files(input_cell_set_files)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)
    isx._internal.c_api.isx_export_cell_set(
            num_cell_sets, in_cell_sets, output_csv_file.encode('utf-8'),
            output_tiff_file.encode('utf-8'), time_ref_int, False, output_props_file.encode('utf-8'))


def export_vessel_set_to_csv_tiff(input_vessel_set_files, output_trace_csv_file='', output_line_csv_file='', output_map_tiff_file='', output_heatmaps_tiff_dir='', time_ref='start'):
    """
    Export vessel sets to a CSV file with trace data and TIFF files with image data.

    Arguments
    ---------
    input_vessel_set_files : list<str>
        The file paths of the vessel sets to export.
    output_trace_csv_file : str
        The path of the trace CSV file to write.
    output_line_csv_file : str
        The path of the line CSV file to write.
    output_map_tiff_file : str
        The name of the vessel map TIFF file to write.
    output_heatmaps_tiff_dir : str
        The name of the directory to write correlation heatmaps as TIFF stacks.
    time_ref : {'start', 'unix'}
        The time reference for the CSV time stamps.
        If 'start' is used, the time stamps represent the seconds since the start of the vessel set.
        If 'unix' is used, the time stamps represents the second since the Unix epoch.
    """
    num_vessel_sets, in_vessel_sets = isx._internal.check_input_files(input_vessel_set_files)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)

    if all([not f for f in [output_trace_csv_file, output_line_csv_file, output_map_tiff_file, output_heatmaps_tiff_dir]]):
        raise ValueError('Must provide at least one output file path.')

    isx._internal.c_api.isx_export_vessel_set(
            num_vessel_sets, in_vessel_sets,
            output_trace_csv_file.encode('utf-8'), output_line_csv_file.encode('utf-8'), output_map_tiff_file.encode('utf-8'), output_heatmaps_tiff_dir.encode('utf-8'),
            time_ref_int)


def export_event_set_to_csv(input_event_set_files, output_csv_file, time_ref='start', output_props_file='', 
                            sparse_output=True, write_amplitude=True):
    """
    Export event sets to a CSV file.

    For more details see :ref:`exportCellsAndStuff`.

    Arguments
    ---------
    input_event_set_files : list<str>
        The file paths of the cell sets to export.
    output_csv_file : str
        The path of the CSV file to write.
    time_ref : {'start', 'unix'}
        The time reference for the CSV time stamps.
        If 'start' is used, the time stamps represent the seconds since the start of the cell set.
        If 'unix' is used, the time stamps represents the second since the Unix epoch.
    output_props_file : str
        The path of the properties CSV file to write.
    sparse_output: bool
        If true, output events in sparse format showing all time points,
        otherwise, output events in dense format showing only timepoints with events.
    write_amplitude: bool
        Only relevant when sparse_output is True.
        If true, write amplitudes of each event,
        otherwise, writes 1 where events occur and 0 elsewhere.
    """
    num_event_sets, in_event_sets = isx._internal.check_input_files(input_event_set_files)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)
    isx._internal.c_api.isx_export_event_set(
            num_event_sets, in_event_sets, output_csv_file.encode('utf-8'), time_ref_int,
            output_props_file.encode('utf-8'), sparse_output, write_amplitude)


def export_gpio_to_isxd(input_gpio_file, output_isxd_dir):
    """
    Export GPIO file (.gpio, .raw, .hdf5, .imu) to an Inscopix Data File (.isxd).
    
    Output file will have the name <input_file_name>_gpio.isxd

    Arguments
    ---------
    input_gpio_file : list<str>
        The file path of the gpio file to be exported.
    output_isxd_dir : str
        The path of the directory to write isxd file.
    """
    isx._internal.c_api.isx_export_gpio_isxd(input_gpio_file.encode('utf-8'), output_isxd_dir.encode('utf-8'))


def export_gpio_set_to_csv(input_gpio_set_files, output_csv_file, inter_isxd_file_dir='/tmp', time_ref='start'):
    """
    Export gpio sets to a CSV file. 

    If exporting more than one file, for correct formatting, files should either be all non .imu files or all .imu files.

    For more details see :ref:`exportCellsAndStuff`.

    Arguments
    ---------
    input_gpio_set_files : list<str>
        The file paths of the cell sets to export.
    output_csv_file : str
        The path of the CSV file to write.
    inter_isxd_file_dir : str
        The path of the directory to put intermediate .isxd file
        The default path for Mac & Linux is /tmp. The default path for Windows is the directory containing the input gpio file.
    time_ref : {'start', 'unix'}
        The time reference for the CSV time stamps.
        If 'start' is used, the time stamps represent the seconds since the start of the cell set.
        If 'unix' is used, the time stamps represents the second since the Unix epoch.
    """
    num_gpio_sets, in_gpio_sets = isx._internal.check_input_files(input_gpio_set_files)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)
    isx._internal.c_api.isx_export_gpio_set(
            num_gpio_sets, in_gpio_sets, output_csv_file.encode('utf-8'), inter_isxd_file_dir.encode('utf-8'), 
            time_ref_int)

def align_start_times(input_ref_file, input_align_files):
    """
    Align the epoch start times of files originating from the same paired and synchronized start-stop recording session.
    The epoch start time stored in the input align files are modified in-place
    so that they are aligned relative to the epoch start time of the input timing reference file
    
    For each input align file, the epoch start time is recomputed using the following formula:
        align_epoch_start_ms = ref_epoch_start_ms + ((align_first_tsc_us - ref_first_tsc_us) / 1e3)
    
    In the event that the first sample of an input align file is dropped, the tsc value of the first sample is inferred using the following formula:
        align_first_tsc_us = align_first_valid_tsc_us - (align_first_valid_idx * align_sample_period_us)

    Arguments
    ---------
        input_ref_file : str
            The path of the file to use as the timing reference to align with the other input files.
            This can be either a .gpio file, .isxd movie, or .isxb movie, otherwise the function will throw an error.
            If the timing reference is a movie, the movie must contain frame timestamps, otherwise this function will throw an error.
        input_align_files : list<str>
            The path of the files to align to the epoch start time of the input timing reference file.
            These files can either be an .isxd movie or .isxb movie, otherwise the function will throw an error.
            The movies must contain frame timestamps, otherwise this function will throw an error.
    """
    num_align_files, in_align_files = isx._internal.check_input_files(input_align_files)
    isx._internal.c_api.isx_align_start_times(
        input_ref_file.encode('utf-8'),
        num_align_files, in_align_files
    )

def export_aligned_timestamps(input_ref_file, input_align_files, input_ref_name, input_align_names, output_csv_file, time_ref='start'):
    """
    Export timestamps from files which originate from the same paired and synced start-stop recording session to a .csv file.
    Timestamps are aligned to a single start time which is defined as the start time of the specified timing reference file.

    Arguments
    ---------
    input_ref_file : str
        The path of the file to use as the timing reference to align with the other input files.
        Timestamps are exported relative to the start time of this file.
        This can be either a .gpio file, .isxd movie, or .isxb movie, otherwise the function will throw an error.
        If the timing reference is a movie, the movie must contain frame timestamps, otherwise this function will throw an error.
    input_align_files : list<str>
        The path of the files to align to the epoch start time of the input timing reference file.
        These files can either be a .gpio file, .isxd movie, or .isxb movie, otherwise the function will throw an error.
        The movies must contain frame timestamps, otherwise this function will throw an error.
    input_ref_name : str
        The name of the reference data set to use in the output csv.
    input_align_names : list<str>
        The names of the align data sets to use in the output csv.
    output_csv_file : str
        The path of the csv file to be written.
    time_ref : {'start', 'unix', 'tsc'}
        The time reference for the CSV time stamps.
        If 'start' is used, the timestamps represent the seconds since the start of the movie.
        If 'unix' is used, the timestamps represent the seconds since the Unix epoch.
        If 'tsc' is used, the timestamps represent the hardware counter value on the acquisition box when each frame was captured.
    """
    num_align_files, in_align_files = isx._internal.check_input_files(input_align_files)
    _, in_align_names = isx._internal.check_input_files(input_align_names)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)
    isx._internal.c_api.isx_export_aligned_timestamps(
        input_ref_file.encode('utf-8'),
        num_align_files, in_align_files,
        input_ref_name.encode('utf-8'), in_align_names,
        output_csv_file.encode('utf-8'),
        time_ref_int
    )

def _get_ethovision_header_size(ethovision_file):
    """
    Open an Ethovision file to get the header length.
    
    Arguments
    ---------
    ethovision_file: str
        The path to the Ethovision file.
        
    Returns
    -------
    int:
        The number of rows in the header of the Ethovision file.
    """
    # Check ethovision file format, use appropriate Pandas import method
    if ethovision_file.lower().endswith('.csv'):
        raw_ethovision_file = pd.read_csv(ethovision_file, header=None).set_index(0)
    elif ethovision_file.lower().endswith('.xlsx'):
        raw_ethovision_file = pd.read_excel(ethovision_file, header=None).set_index(0)
    else:
        raise ValueError("Only .csv or .xlsx file formats are accepted for Ethovision files.")
        
    # Extract the number of header lines from the ethovision file
    number_of_header_lines = int(raw_ethovision_file.loc['Number of header lines:'][1])
    return number_of_header_lines

def _load_ethovision_data(ethovision_file):
    """
    Load the data from the Ethovision file, ignoring the header.
    Header length will be read from the file to ensure that the correct number of rows are ignored.
    Note that Ethovision convention is to encode NaN data with the '-' character. This function
        replaces all '-' with NaN.
    
    Arguments
    ---------
    ethovision_file: str
        The path to the Ethovision file.
        
    Returns
    -------
    Pandas.DataFrame:
        Dataframe with all columns from Ethovision file returned.
    """
    # get number of header lines
    number_of_header_lines = _get_ethovision_header_size(ethovision_file)
    
    # load the data using either `pd.read_csv` or `pd.read_excel`, depending on extension
    if ethovision_file.endswith('.csv'):
        ethovision_data = pd.read_csv(
            ethovision_file, 
            header=number_of_header_lines-2, 
            skiprows=[number_of_header_lines-1], 
            low_memory=False
        ).replace('-', np.nan)
    elif ethovision_file.endswith('.xlsx'):
        ethovision_data = pd.read_excel(
            ethovision_file, 
            header=number_of_header_lines-2, 
            skiprows=[number_of_header_lines-1]
        ).replace('-', np.nan)
    else:
        raise ValueError("Only .csv or .xlsx file formats are accepted for Ethovision files.")
        
    return ethovision_data

def export_ethovision_data_with_isxb_timestamps(
    input_ethovision_file,
    input_isxb_file,
    output_csv_file,
    input_ref_file=None,
    time_ref='start'
):
    """
    Given paths to an Ethovision file and a nVision (.isxb) movie file,
        writes a csv file with Ethovision data and a dedicated column,
        either `isxb Frame Timestamp (s)` or `isxb Frame Timestamp (us)`
        depending on the format of the timestamps,
        that contains aligned timestamps from the nVision movie.
        
    It is important to note that internal Inscopix testing has shown that the Ethovision output file
        can have a length that is one less than the number of frames in the input movie. When this is true,
        the missing data row appears to be at the beginning.
        This function applies .isxb timestamps to the Ethovision table while ignoring the first timestamp
        to compensate for the length mismatch if the mismatch exists. More recent versions of Ethovision
        should prevent this from occuring.
        
    Arguments
    ---------
    input_ethovision_file: str
        The path to the Ethovision file (.csv, .xlsx).
    input_isxb_file: str
        The path to the nVision (.isxb) movie file.
    output_csv_file: str
        The path to the output csv file.
    input_ref_file: str | None
        The path to the reference file (.isxd, .gpio, .isxb).
        Timestamps from the .isxb file are aligned to the start time of this file.
        Generally, this reference is a file from a miniscope recording which was synchronized with this .isxb file.
        This argument is required if the `time_ref` is selected as 'start' or 'unix'.
        If the .isxb file is a standalone behaviour recording that is not synchronized to any miniscope file,
        simply provide the .isxb file as the reference (if it's required).
    time_ref : {'start', 'unix', 'tsc'}
        The time reference for the nVision (.isxb) timestamps.
        If 'tsc' is used, the timestamps represent the hardware counter value in microseconds on the acquisition box when each frame was captured.
        If 'start' is used, the timestamps represent the seconds since the start of the experiment.
            Note: `input_ref_file` must be specified if this option is used, otherwise an exception is thrown.
            Timestamps are exported relative to the start time of the reference file.
        If 'unix' is used, the timestamps represent the seconds since the Unix epoch.
            Note: `input_ref_file` must be specified if this option is used, otherwise an exception is thrown.
            The `input_ref_file` ensures that `isx.align_start_times` has been called
            on the .isxb file with the corresponding, synchronized miniscope file.
    """
    if os.path.exists(output_csv_file):
        raise FileExistsError('File already exists: {}'.format(output_csv_file))

    if (time_ref == 'start' or time_ref == 'unix') and input_ref_file is None:
        raise ValueError("An input reference file is required for time_ref = 'start' or time_ref = 'unix'.")

    # read timestamps for isxb movie
    if time_ref == 'start':
        isx.export_aligned_timestamps(
            input_ref_file=input_ref_file,
            input_align_files=[input_isxb_file],
            input_ref_name='ref',
            input_align_names=['isxb'],
            output_csv_file=output_csv_file,
            time_ref='start'
        )
        assert os.path.exists(output_csv_file)
        isxb_timestamps_df = pd.read_csv(output_csv_file)
        isxb_timestamps = isxb_timestamps_df['isxb Timestamp (s)'].tolist()
        timestamp_column_name = 'isxb Frame Timestamp (s)'
    else:
        if time_ref == 'unix':
            isx.align_start_times(
                input_ref_file=input_ref_file,
                input_align_files=[input_isxb_file]
            )
        
        isx.export_movie_timestamps_to_csv(
            input_movie_files=input_isxb_file,
            output_csv_file=output_csv_file,
            time_ref=time_ref
        )
        assert os.path.exists(output_csv_file)
        timestamp_column_index = 3
        isxb_timestamps_df = pd.read_csv(output_csv_file)
        timestamp_column_name = isxb_timestamps_df.columns[timestamp_column_index]
        isxb_timestamps = isxb_timestamps_df[timestamp_column_name].tolist()
        timestamp_column_name = "isxb " + timestamp_column_name
    
    os.remove(output_csv_file)

    # get ethovision data
    ethovision_data = _load_ethovision_data(input_ethovision_file)
    if len(ethovision_data) == len(isxb_timestamps) - 1:
        # deal with off-by-one issue
        warning_text = (
            "Your Ethovision data file is one element shorter "
            "than the number of frames in your movie. "
            "This is a known issue in earlier releases of Ethovision. "
            "To correct for it, the first timestamp in your array "
            "of timestamps is being dropped. Please consider updating to "
            "the newest version of Ethovision to avoid this behavior."
        )
        warnings.warn(warning_text)
        
        isxb_timestamps_to_use = isxb_timestamps[1:]
        
    elif len(ethovision_data) == len(isxb_timestamps):
        # equal lengths are ideal
        isxb_timestamps_to_use = isxb_timestamps
    else:
        # else, throw a ValueError
        raise ValueError(
            "Length of timestamps array "
            f"({len(isxb_timestamps)}) "
            "is not the same as (or within one) of the ethovision table "
            f"({len(ethovision_data)})"
        )
        
    # get ethovision columns
    ethovision_columns = ethovision_data.columns
    
    # add a column with `isxb Frame Timestamp (s)/(us)`` to the ethovision data, ignoring the first to deal with mismatch
    ethovision_data[timestamp_column_name] = isxb_timestamps_to_use
    
    # return ethovision data with `isxb Frame Timestamp (s)/(us)` column in leftmost position
    aligned_data_table = ethovision_data[[timestamp_column_name] + ethovision_columns.tolist()]
    aligned_data_table.to_csv(output_csv_file, index=False)
