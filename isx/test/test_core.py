# This tests some core functionality related to timing and spacing.

import datetime
import numpy as np
import pytest

import isx

class TestCore:
    def test_version(self):
        assert isx.__version__

    @pytest.mark.duration
    def test_duration_from_to_secs(self):
        secs = 79
        duration = isx.Duration.from_secs(secs)
        assert duration.to_secs() == secs
        assert duration._impl.den == 1

    @pytest.mark.duration
    def test_duration_from_to_msecs(self):
        msecs = 5237
        duration = isx.Duration.from_msecs(msecs)
        assert duration.to_msecs() == msecs
        assert duration._impl.den == 1e3

    @pytest.mark.duration
    def test_duration_from_to_usecs(self):
        usecs = 123920
        duration = isx.Duration.from_usecs(usecs)
        assert duration.to_usecs() == usecs
        assert duration._impl.den == 1e6

    @pytest.mark.duration
    def test_duration_from_secs_float(self):
        secs = 0.09364790469408035
        duration = isx.Duration._from_secs_float(secs, max_denominator=1000000000)
        assert duration._impl.num == 12569209
        assert duration._impl.den == 134217728

        simplified_duration = isx.Duration._from_secs_float(secs, max_denominator=100)
        assert simplified_duration._impl.num == 3
        assert simplified_duration._impl.den == 32

    @pytest.mark.duration
    def test_duration_str_valid(self):
        duration = isx.Duration.from_msecs(50)
        assert isinstance(str(duration), str)

    @pytest.mark.duration
    def test_duration_str_invalid(self):
        duration = isx.Duration()
        assert isinstance(str(duration), str)

    @pytest.mark.time
    def test_time_from_to_secs_since_epoch(self):
        secs_since_epoch = isx.Duration.from_secs(1523469679)
        time = isx.Time._from_secs_since_epoch(secs_since_epoch)
        assert time._to_secs_since_epoch() == secs_since_epoch
        exp_datetime = datetime.datetime(2018, 4, 11, 18, 1, 19)
        assert time.to_datetime() == exp_datetime

    @pytest.mark.time
    def test_time_from_msecs_since_epoch(self):
        secs_since_epoch = isx.Duration.from_msecs(1523469679261)
        time = isx.Time._from_secs_since_epoch(secs_since_epoch)
        assert time._to_secs_since_epoch() == secs_since_epoch
        exp_datetime = datetime.datetime(2018, 4, 11, 18, 1, 19, 261000)
        assert time.to_datetime() == exp_datetime

    @pytest.mark.time
    def test_time_str_valid(self):
        time = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1523469679261))
        assert isinstance(str(time), str)

    @pytest.mark.time
    def test_time_str_valid(self):
        time = isx.Time()
        assert isinstance(str(time), str)

    @pytest.mark.spacing
    def test_spacing_get_num_pixels(self):
        num_pixels = (4, 7)
        spacing = isx.Spacing(num_pixels=num_pixels)
        assert spacing.num_pixels == num_pixels

    @pytest.mark.spacing
    def test_spacing_str_valid(self):
        spacing = isx.Spacing(num_pixels=(1080, 1440))
        assert isinstance(str(spacing), str)

    @pytest.mark.spacing
    def test_spacing_str_invalid(self):
        spacing = isx.Spacing()
        assert isinstance(str(spacing), str)

    @pytest.mark.timing
    def test_timing_get_offsets_since_start(self):
        num_samples = 10
        period = isx.Duration.from_msecs(26)
        timing = isx.Timing(num_samples=num_samples, period=period)

        # Note that the first offset is 0/1, so if we fill it with 0/1000
        # our strict equality check will fail.
        exp_offsets = [isx.Duration()]
        for i in range(1, num_samples):
            exp_offsets.append(isx.Duration.from_msecs(i * 26))

        act_offsets = timing.get_offsets_since_start()
        for i in range(num_samples):
            assert act_offsets[i] == exp_offsets[i]

    @pytest.mark.timing
    def test_timing_get_valid_samples(self):
        num_samples = 10
        timing = isx.Timing(num_samples=num_samples)
        np.testing.assert_array_equal(timing.get_valid_samples(), range(num_samples))

    @pytest.mark.timing
    def test_timing_get_valid_samples_with_dropped(self):
        num_samples = 10
        dropped = [2, 4, 5, 6]
        timing = isx.Timing(num_samples=num_samples, dropped=dropped)
        np.testing.assert_array_equal(timing.dropped, dropped)
        np.testing.assert_array_equal(timing.get_valid_samples(), [0, 1, 3, 7, 8, 9])

    @pytest.mark.timing
    def test_timing_get_valid_samples_with_cropped(self):
        num_samples = 10
        cropped = [[2, 3], [6, 8]]
        timing = isx.Timing(num_samples=num_samples, cropped=cropped)
        np.testing.assert_array_equal(timing.cropped, cropped)
        np.testing.assert_array_equal(timing.get_valid_samples(), [0, 1, 4, 5, 9])

    @pytest.mark.timing
    def test_timing_get_valid_samples_with_dropped_and_cropped(self):
        num_samples = 10
        dropped = [2, 4]
        cropped = [[6, 8]]
        timing = isx.Timing(num_samples=num_samples, dropped=dropped, cropped=cropped)
        np.testing.assert_array_equal(timing.dropped, dropped)
        np.testing.assert_array_equal(timing.cropped, cropped)
        np.testing.assert_array_equal(timing.get_valid_samples(), [0, 1, 3, 5, 9])

    @pytest.mark.timing
    def test_timing_str_valid(self):
        timing = isx.Timing(num_samples=18, dropped=[2, 4], cropped=[[5, 6], [9, 11]])
        assert isinstance(str(timing), str)

    @pytest.mark.timing
    def test_timing_str_invalid(self):
        timing = isx.Timing()
        assert isinstance(str(timing), str)
