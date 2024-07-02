"""
The core module contains functionality that is used by most other
modules frequently.
"""

import ctypes
import datetime
from fractions import Fraction

import numpy as np

import isx._internal

__version__ = isx._internal.get_version_string()

class Duration(object):
    """
    A duration of time.

    Examples
    --------
    Make a period of 50 milliseconds

    >>> period = isx.Duration.from_msecs(50)

    Attributes
    ----------
    secs_float : float
        The duration in seconds as a floating point number.
    """

    def __init__(self):
        self._impl = isx._internal.IsxRatio(0, 1)

    @property
    def secs_float(self):
        return float(self._impl.num) / float(self._impl.den)

    def to_secs(self):
        """ Convert to an integer number of whole seconds.
        """
        return int(self.secs_float)

    def to_msecs(self):
        """ Convert to an integer number of whole milliseconds.
        """
        return int(self.secs_float * 1e3)

    def to_usecs(self):
        """ Convert to an integer number of whole microseconds.
        """
        return round(self.secs_float * 1e6)

    @classmethod
    def from_secs(cls, secs):
        """ Make a duration from a number of seconds.
        """
        return cls._from_num_den(int(secs), int(1))

    @classmethod
    def from_msecs(cls, msecs):
        """ Make a duration from a number of milliseconds.
        """
        return cls._from_num_den(int(msecs), int(1e3))

    @classmethod
    def from_usecs(cls, usecs):
        """ Make a duration from a number of microseconds.
        """
        return cls._from_num_den(int(usecs), int(1e6))

    @classmethod
    def _from_num_den(cls, num, den):
        return cls._from_impl(isx._internal.IsxRatio(num, den))

    @classmethod
    def _from_secs_float(cls, flt, max_denominator=1000000000):
        frac = Fraction(flt).limit_denominator(max_denominator)
        return cls._from_impl(isx._internal.IsxRatio(frac.numerator, frac.denominator))

    @classmethod
    def _from_impl(cls, impl):
        self = cls()
        self._impl = impl
        return self

    def __eq__(self, other):
        return self._impl == other._impl

    def __str__(self):
        return '{}s'.format(self.secs_float)


class Time(object):
    """
    A time stamp that defines a calendar date and wall time.
    """

    def __init__(self):
        self._impl = isx._internal.IsxTime(isx._internal.IsxRatio(0, 1), 0)

    def to_datetime(self):
        """
        Returns
        -------
        :class:`datetime.datetime`
            The nearest Python datetime.
        """
        secs_since_epoch = self._to_secs_since_epoch().secs_float + float(self._impl.utc_offset)
        return datetime.datetime.utcfromtimestamp(secs_since_epoch)

    def _to_secs_since_epoch(self):
        return isx.Duration._from_impl(self._impl.secs_since_epoch)

    @classmethod
    def _from_secs_since_epoch(cls, secs_since_epoch, utc_offset=0):
        self = cls()
        self._impl.secs_since_epoch = secs_since_epoch._impl
        self._impl.utc_offset = utc_offset
        return self

    @classmethod
    def _from_impl(cls, impl):
        self = cls()
        self._impl = impl
        return self

    def __eq__(self, other):
        return self._impl == other._impl

    def __str__(self):
        return str(self.to_datetime())


class Timing(object):
    """
    The timing associated with a set of samples, such as the frames of a movie or the
    values of a trace.

    Some samples are described as invalid, meaning that the sample is missing.
    These include dropped samples, which could arise due to an error at acquisition time,
    and cropped samples, which are likely due to processing.

    Examples
    --------
    Make timing for 9 samples, every 10 milliseconds.

    >>> timing = isx.Timing(num_samples=9, period=isx.Duration.from_msecs(10))

    Attributes
    ----------
    num_samples : int >= 0
        The number of samples, including invalid (dropped, cropped, or blank) ones.
    period : :class:`isx.Duration`
        The period or duration of one sample.
    start : :class:`isx.Time`
        The time stamp associated with the start of the first sample.
    dropped : list<int>
        The indices of the dropped samples.
    cropped : list<2-tuple>
        The index ranges of the cropped samples.
        Each element specifies the inclusive lower and upper bounds of a range
        of indices.
    blank : list<int>
        The indices of the blank samples.
    """

    def __init__(self, num_samples=0, period=Duration.from_msecs(50), start=Time(), dropped=[], cropped=[], blank=[]):
        """
        __init__(self, num_samples=0, period=``isx.Duration.from_msecs(50)``, start=``isx.Time()``, dropped=[], cropped=[], blank=[]):

        Make a timing object.

        Arguments
        ---------
        num_samples : int >= 0
            The number of samples, including invalid (dropped, cropped, or blank) ones.
        period : :class:`isx.Duration`
            The period or duration of one sample.
        start : :class:`isx.Time`
            The time stamp associated with the start of the first sample.
        dropped : 1D array-like
            The indices of the dropped samples.
        cropped : Nx2 array-like
            The index ranges of the cropped samples.
            Each 2-tuple or row specifies the inclusive lower and upper bounds of a
            range of indices.
        blank : list<int>
            The indices of the blank samples.
        """

        if num_samples < 0:
            raise ValueError('num_samples must be non-negative')

        if not isinstance(period, Duration):
            raise TypeError('period must be an isx.Duration object')

        if not isinstance(start, Time):
            raise ValueError('start_time must be an isx.Time object')

        self._impl = isx._internal.IsxTimingInfo()
        self._impl.num_samples = num_samples
        self._impl.step = period._impl
        self._impl.start = start._impl

        dropped = isx._internal.convert_to_1d_numpy_array(dropped, np.uint64, 'dropped')
        self._impl.num_dropped = dropped.size
        self._impl.dropped = isx._internal.numpy_array_to_ctypes_array(dropped, ctypes.c_uint64)

        cropped = isx._internal.convert_to_nx2_numpy_array(cropped, np.uint64, 'cropped')
        self._impl.num_cropped = cropped.shape[0]
        self._impl.cropped_first = isx._internal.numpy_array_to_ctypes_array(cropped[:, 0], ctypes.c_uint64)
        self._impl.cropped_last = isx._internal.numpy_array_to_ctypes_array(cropped[:, 1], ctypes.c_uint64)

        blank = isx._internal.convert_to_1d_numpy_array(blank, np.uint64, 'blank')
        self._impl.num_blank = blank.size
        self._impl.blank = isx._internal.numpy_array_to_ctypes_array(blank, ctypes.c_uint64)

    @property
    def start(self):
        return isx.Time._from_impl(self._impl.start)

    @property
    def period(self):
        return isx.Duration._from_impl(self._impl.step)

    @property
    def num_samples(self):
        return self._impl.num_samples

    @property
    def dropped(self):
        return isx._internal.ctypes_ptr_to_list(self._impl.dropped, self._impl.num_dropped)

    @property
    def cropped(self):
        cropped_first = isx._internal.ctypes_ptr_to_list(self._impl.cropped_first, self._impl.num_cropped)
        cropped_last = isx._internal.ctypes_ptr_to_list(self._impl.cropped_last, self._impl.num_cropped)
        return [(first, last) for first, last in zip(cropped_first, cropped_last)]

    @property
    def blank(self):
        return isx._internal.ctypes_ptr_to_list(self._impl.blank, self._impl.num_blank)

    def get_offsets_since_start(self):
        """
        Get the offsets from the start of the timing.

        Returns
        -------
        list<:class:`isx.Duration`>
            Each element is the offset from the start to a sample.
        """
        OffsetsType = ctypes.c_int64 * self.num_samples
        offsets_num = OffsetsType()
        offsets_den = OffsetsType()
        isx._internal.c_api.isx_timing_info_get_secs_since_start(ctypes.byref(self._impl), offsets_num, offsets_den)
        durations = []
        for i in range(self.num_samples):
            durations.append(isx.Duration._from_num_den(offsets_num[i], offsets_den[i]))
        return durations

    def get_valid_samples_mask(self):
        """
        Get a 1D array mask indicating whether each sample is valid.

        Returns
        -------
        :class:`numpy.ndarray`
            Each element indicates whether the corresponding sample is valid.
        """
        mask = (ctypes.c_uint8 * self.num_samples)()
        isx._internal.c_api.isx_timing_info_get_valid_sample_mask(ctypes.byref(self._impl), mask)
        return np.array(mask, dtype=bool)

    def get_valid_samples(self):
        """
        Returns
        -------
        :class:`numpy.ndarray`
            The valid sample indices.
        """
        return np.flatnonzero(self.get_valid_samples_mask())

    @classmethod
    def _from_impl(cls, impl):
        self = cls()
        self._impl = impl
        return self

    def __eq__(self, other):
        return ((self.start == other.start) and
                (self.period == other.period) and
                (self.num_samples == other.num_samples) and
                (self.dropped == other.dropped) and
                (self.cropped == self.cropped) and
                (self.blank == other.blank))


    def __str__(self):
        return 'Timing(num_samples={}, period={}, start={}, dropped={}, cropped={}, blank={})'.format(
                self.num_samples, self.period, self.start, self.dropped, self.cropped, self.blank)


class Spacing(object):
    """
    The spacing associated with a set of pixels.

    Examples
    --------
    Make spacing for a 1440x1080 image.

    >>> spacing = isx.Spacing(num_pixels=(1080, 1440))

    Attributes
    ----------
    num_pixels : 2-tuple<int>
        The number of pixels as (num_rows, num_cols).
    """

    def __init__(self, num_pixels=(0, 0)):
        if len(num_pixels) != 2:
            raise ValueError('num_pixels must be specified as a two element list/tuple/array (num_rows, num_cols)')
        self._impl = isx._internal.IsxSpacingInfo.from_num_pixels(num_pixels)

    @property
    def num_pixels(self):
        return (self._impl.num_rows, self._impl.num_cols)

    @property
    def _pixel_coordinates(self):
        pixel_width = self._impl.pixel_width._as_float()
        pixel_height = self._impl.pixel_height._as_float()
        left = self._impl.left._as_float() / pixel_width    # convert micron to pixel
        top = self._impl.top._as_float() / pixel_height
        numX = self._impl.num_cols
        numY = self._impl.num_rows

        return np.array([round(left), round(top), numX, numY]).astype(int)

    @classmethod
    def _from_impl(cls, impl):
        self = cls()
        self._impl = impl
        return self

    def __eq__(self, other):
        return self._impl == other._impl

    def __str__(self):
        return 'Spacing(num_pixels={})'.format(self.num_pixels)
