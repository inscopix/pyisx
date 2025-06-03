from .utilities.setup import delete_files_silently, test_data_path, is_file
import shutil, os, platform

import h5py
import numpy as np
import pandas as pd
import pytest
import warnings as w

import isx

from .utilities.create_sample_data import write_sample_cellset, write_sample_vessel_diameter_set, write_sample_rbc_velocity_set
from .asserts import assert_tiff_files_equal_by_path, \
    compare_h5_groups, assert_csv_events_are_equal_by_path, \
    assert_csv_files_are_equal_by_path, assert_isxd_images_are_close_by_path_nan_zero

data_types = ('float16', 'float32', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'int16', 'int32', 'int64')

class TestFileIO:
    def test_ReadCellSet(self):
        input_file = test_data_path + '/unit_test/eventDetectionCellSet.isxd'
        cellset = isx.CellSet.read(input_file)

        assert cellset.get_cell_name(0) == 'C0'
        assert cellset.get_cell_name(1) == 'C1'
        assert cellset.get_cell_name(2) == 'C2'

        assert cellset.num_cells == 3

        exp_period = isx.Duration._from_num_den(1, 10)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_secs(20))
        exp_spacing = isx.Spacing(num_pixels=(200,200))
        exp_spacing._impl.pixel_width = isx._internal.IsxRatio(22, 10)
        exp_spacing._impl.pixel_height = isx._internal.IsxRatio(22, 10)

        assert cellset.timing == isx.Timing(num_samples=50, period=exp_period, start=exp_start)
        assert cellset.spacing == exp_spacing

    def test_ReadCellSetName(self):
        input_file = test_data_path + '/unit_test/eventDetectionCellSet.isxd'
        cellset = isx.CellSet.read(input_file)

        name = cellset.get_cell_name(1)

        assert name == 'C1'

    def test_read_cell_set_with_dropped_cropped(self):
        cellset = isx.CellSet.read(test_data_path + '/unit_test/cropped/Trimmed-ROI.isxd')

        exp_period = isx.Duration._from_num_den(100, 1500)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1478271283662))
        exp_timing = isx.Timing(num_samples=40, period=exp_period, start=exp_start,
                dropped=[10], cropped=[[2, 6], [12, 12], [19, 34]])

        assert cellset.timing == exp_timing

    def test_ReadWriteCellSet(self):
        cs_out_file = test_data_path + '/unit_test/output/test_readwrite_cellset.isxd'
        delete_files_silently([cs_out_file])

        # create sample data that will be used to make the cell set
        cell_props = write_sample_cellset(cs_out_file)

        # read the created cell set file and confirm the correct values have been written
        cs_in = isx.CellSet.read(cs_out_file)
        assert cs_in.num_cells == cell_props['num_cells']
        assert cs_in.spacing == cell_props['spacing']
        assert cs_in.timing == cell_props['timing']
        valid_samples_mask = cs_in.timing.get_valid_samples_mask()
        for k in range(cs_in.num_cells):
            np.testing.assert_array_equal(cs_in.get_cell_image_data(k), cell_props['images'][k, :, :])
            trace = cs_in.get_cell_trace_data(k)
            exp_trace = np.where(valid_samples_mask, cell_props['traces'][k, :], np.nan)
            np.testing.assert_array_equal(trace, exp_trace)
            assert cs_in.get_cell_name(k) == cell_props['names'][k]
            assert cs_in.get_cell_status(k) == 'undecided'

        del cs_in
        delete_files_silently([cs_out_file])


    @pytest.mark.cellset
    @pytest.mark.parametrize('image_type', data_types)
    @pytest.mark.parametrize('traces_type', data_types)
    def test_ReadWriteCellSetOtherTypeToFloat32(self, traces_type, image_type):
        cs_out_file = test_data_path + '/unit_test/output/test_readwrite_cellset.isxd'
        delete_files_silently([cs_out_file])

        num_cells = 2
        num_pixels = (4, 3)
        num_samples = 5
        timing = isx.Timing(num_samples=num_samples)
        spacing = isx.Spacing(num_pixels=num_pixels)

        cs_out = isx.CellSet.write(cs_out_file, timing, spacing)
        images = np.random.rand(num_cells, *num_pixels).astype(image_type)
        traces = np.random.rand(num_cells, num_samples).astype(traces_type)
        for c in range(num_cells):

            if image_type != 'float32':
                if traces_type != 'float32':
                    with pytest.warns(UserWarning) as warnings:
                        cs_out.set_cell_data(c, images[c, :, :], traces[c, :], 'C{:02d}'.format(c))
                        assert 'Converting from {} to float32.'.format(image_type) in [str(x.message) for x in warnings]
                        assert 'Converting from {} to float32.'.format(traces_type) in [str(x.message) for x in warnings]
                else:
                    with pytest.warns(UserWarning) as warnings:
                        cs_out.set_cell_data(c, images[c, :, :], traces[c, :], 'C{:02d}'.format(c))
                        assert 'Converting from {} to float32.'.format(image_type) in [str(x.message) for x in warnings]
            else:
                if traces_type != 'float32':
                    with pytest.warns(UserWarning) as warnings:
                        cs_out.set_cell_data(c, images[c, :, :], traces[c, :], 'C{:02d}'.format(c))
                        assert 'Converting from {} to float32.'.format(traces_type) in [str(x.message) for x in warnings]
                else:
                    with w.catch_warnings():
                        w.simplefilter("error")
                        cs_out.set_cell_data(c, images[c, :, :], traces[c, :], 'C{:02d}'.format(c))

        cs_out.flush()

        # read the created cell set file and confirm the correct values have been written
        cs_in = isx.CellSet.read(cs_out_file)
        for c in range(cs_in.num_cells):
            np.testing.assert_array_almost_equal(cs_in.get_cell_image_data(c), images[c, :, :])
            np.testing.assert_array_almost_equal(cs_in.get_cell_trace_data(c), traces[c, :])

        del cs_in
        del cs_out
        delete_files_silently([cs_out_file])

    def test_ReadWriteCellSetStatus(self):
        """ Test writing a cell set, then reading it and setting the status, then reading it again. """

        cs_out_file = test_data_path + '/unit_test/output/test_readwrite_cellset.isxd'
        delete_files_silently([cs_out_file])

        # create sample data that will be used to make the cell set
        cell_props = write_sample_cellset(cs_out_file)

        # read the created cell set file and set the new status values
        cs_in = isx.CellSet.read(cs_out_file, read_only=False)

        statuses = ['accepted', 'rejected']
        written_status = list()
        for k in range(cs_in.num_cells):
            new_stat = statuses[int(np.random.rand() > 0.50)]
            cs_in.set_cell_status(k, new_stat)
            written_status.append(new_stat)

        # re-read the created cell set file and check the status values
        cs_in = isx.CellSet.read(cs_out_file)
        for k in range(cs_in.num_cells):
            assert cs_in.get_cell_status(k) == written_status[k], "Cell status does not match"

        del cs_in
        delete_files_silently([cs_out_file])

    def test_GetCell(self):
        input_file = test_data_path + '/unit_test/eventDetectionCellSet.isxd'
        cellset = isx.CellSet.read(input_file)

        trace = cellset.get_cell_trace_data(0)
        image = cellset.get_cell_image_data(0)
        status = cellset.get_cell_status(0)

        assert trace.shape == (cellset.timing.num_samples,)
        assert image.shape == (200, 200)
        assert status == 'undecided'


    def test_CellSetStrValid(self):
        cell_set = isx.CellSet.read(test_data_path + '/unit_test/eventDetectionCellSet.isxd')
        assert isinstance(str(cell_set), str)


    def test_CellSetStrInvalid(self):
        cell_set = isx.CellSet()
        assert isinstance(str(cell_set), str)


    def test_ReadImage(self):
        file_path = test_data_path + '/unit_test/single_10x10_frameMovie.isxd'
        image = isx.Image.read(file_path)
        assert image.mode == 'r'
        assert image.data_type == np.uint16
        exp_spacing = isx.Spacing(num_pixels=(10, 10))
        exp_spacing._impl.pixel_width = isx._internal.IsxRatio(22, 10)
        exp_spacing._impl.pixel_height = isx._internal.IsxRatio(22, 10)
        assert image.spacing == exp_spacing
        assert image.file_path == file_path
        image_data = image.get_data()
        assert image_data.dtype == image.data_type
        assert image_data.shape == image.spacing.num_pixels
        mask = np.zeros(image_data.shape, dtype=bool)
        mask[3:7, 3:7] = True
        np.testing.assert_array_equal(image_data, np.where(mask, 1, 0))

    def test_WriteImageU16(self):
        file_path = test_data_path + '/unit_test/output/image-u16.isxd'
        delete_files_silently([file_path])
        spacing = isx.Spacing(num_pixels=(5, 10))
        data_type = np.uint16
        data = np.random.randint(low=0, high=4095, size=spacing.num_pixels, dtype=data_type)
        image = isx.Image.write(file_path, spacing, data_type, data)
        assert image.mode == 'w'
        assert image.data_type == data_type
        assert image.spacing == spacing
        assert image.file_path == file_path
        np.testing.assert_array_equal(image.get_data(), data)

        image = isx.Image.read(file_path)
        assert image.mode == 'r'
        assert image.data_type == data_type
        assert image.spacing == spacing
        assert image.file_path == file_path
        np.testing.assert_array_equal(image.get_data(), data)

        del image
        delete_files_silently([file_path])

    def test_WriteImageF32(self):
        file_path = test_data_path + '/unit_test/output/image-f32.isxd'
        spacing = isx.Spacing(num_pixels=(6, 7))
        data_type = np.float32
        data = np.random.random(spacing.num_pixels).astype(data_type)

        delete_files_silently([file_path])
        image = isx.Image.write(file_path, spacing, data_type, data)
        assert image.mode == 'w'
        assert image.data_type == data_type
        assert image.spacing == spacing
        assert image.file_path == file_path
        np.testing.assert_array_equal(image.get_data(), data)

        image = isx.Image.read(file_path)
        assert image.mode == 'r'
        assert image.data_type == data_type
        assert image.spacing == spacing
        assert image.file_path == file_path
        np.testing.assert_array_equal(image.get_data(), data)

        del image
        delete_files_silently([file_path])


    @pytest.mark.image
    @pytest.mark.parametrize('image_data_type', data_types)
    @pytest.mark.parametrize('container_data_type', ('uint16', 'float32'))
    def test_ReadWriteImageOtherTypeToFloat32(self, image_data_type, container_data_type):
        image_file = test_data_path + '/unit_test/output/test_readwrite_image.isxd'
        spacing = isx.Spacing(num_pixels=(4, 5))
        data = np.random.random(spacing.num_pixels).astype(image_data_type)

        delete_files_silently([image_file])
        if image_data_type != container_data_type:
            with pytest.warns(UserWarning) as warnings:
                image = isx.Image.write(image_file, spacing, np.__getattribute__(container_data_type), data)
                assert 'Converting from {0} to {1}.'.format(image_data_type, container_data_type) in [str(x.message) for x in warnings]
        else:
            with w.catch_warnings():
                w.simplefilter("error")
                image = isx.Image.write(image_file, spacing, np.__getattribute__(container_data_type), data)

        np.testing.assert_array_equal(image.get_data(), data.astype(container_data_type))

        # read the created image and confirm the correct values have been written
        image = isx.Image.read(image_file)
        np.testing.assert_array_equal(image.get_data(), data.astype(container_data_type))

        del image
        delete_files_silently([image_file])


    def test_ImageStrValid(self):
        image = isx.Image.read(test_data_path + '/unit_test/single_10x10_frameMovie.isxd')
        assert isinstance(str(image), str)


    def test_ImageStrInvalid(self):
        image = isx.Image()
        assert isinstance(str(image), str)


    def test_ReadMovie(self):
        input_path = test_data_path + '/unit_test/recording_20160426_145041.xml'

        movie = isx.Movie.read(input_path)

        assert movie.spacing == isx.Spacing(num_pixels=(500, 500))
        exp_period = isx.Duration._from_num_den(1000, 10020)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1461682241930))
        assert movie.timing == isx.Timing(num_samples=33, period=exp_period, start=exp_start)

    def test_ReadMovieF32(self):
        input_path = test_data_path + '/unit_test/guilded/exp_mosaicReadMovieF32.isxd'

        movie = isx.Movie.read(input_path)

        exp_spacing = isx.Spacing(num_pixels=(3, 4))
        exp_spacing._impl.pixel_width = isx._internal.IsxRatio(22, 10)
        exp_spacing._impl.pixel_height = isx._internal.IsxRatio(22, 10)
        assert movie.spacing == exp_spacing
        exp_period = isx.Duration.from_msecs(20)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration._from_num_den(1, 100))
        assert movie.timing == isx.Timing(num_samples=3, period=exp_period, start=exp_start)
        assert movie.data_type == np.float32

        f0 = movie.get_frame_data(0)
        exp_f0 = np.linspace(-1, 1, 12).reshape(movie.spacing.num_pixels)
        assert np.max(np.abs(f0 - exp_f0)) < 1e-6

        f1 = movie.get_frame_data(1)
        exp_f1 = np.linspace(-1, 1, 12)[::-1].reshape(movie.spacing.num_pixels)
        assert np.max(np.abs(f1 - exp_f1)) < 1e-6

        f2 = movie.get_frame_data(2)
        exp_f2 = np.linspace(0, 1, 12).reshape(movie.spacing.num_pixels)
        assert np.max(np.abs(f2 - exp_f2)) < 1e-6

    def test_ReadMovieU16(self):
        input_path = test_data_path + '/unit_test/guilded/exp_mosaicReadMovieU16.isxd'

        movie = isx.Movie.read(input_path)

        exp_spacing = isx.Spacing(num_pixels=(3, 4))
        exp_spacing._impl.pixel_width = isx._internal.IsxRatio(22, 10)
        exp_spacing._impl.pixel_height = isx._internal.IsxRatio(22, 10)
        assert movie.spacing == exp_spacing
        exp_period = isx.Duration.from_msecs(20)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration._from_num_den(1, 100))
        assert movie.timing == isx.Timing(num_samples=3, period=exp_period, start=exp_start)
        assert movie.data_type == np.uint16

        f0 = movie.get_frame_data(0)
        exp_f0 = np.arange(1, 13, dtype='uint16').reshape(movie.spacing.num_pixels)
        assert np.max(np.abs(f0 - exp_f0)) < 1e-6

        f1 = movie.get_frame_data(1)
        exp_f1 = np.arange(1, 13, dtype='uint16')[::-1].reshape(movie.spacing.num_pixels)
        assert np.max(np.abs(f1 - exp_f1)) < 1e-6

        f2 = movie.get_frame_data(2)
        exp_f2 = np.array([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]).reshape(movie.spacing.num_pixels)
        assert np.max(np.abs(f2 - exp_f2)) < 1e-6

    def test_ReadMovieU16XmlTiff(self):
        input_path = test_data_path + '/unit_test/recording_20161104_145443.xml';

        movie = isx.Movie.read(input_path)

        assert movie.spacing == isx.Spacing(num_pixels=(1080, 1440))
        exp_period = isx.Duration._from_num_den(1000, 15000)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1478271283662))
        assert movie.timing == isx.Timing(num_samples=40, period=exp_period, start=exp_start, dropped=[10])
        assert movie.data_type == np.uint16

        f0 = movie.get_frame_data(0)
        assert f0[0, 0] == 242
        assert f0[378, 654] == 2468
        assert f0[861, 1143] == 472


    def test_ReadMovieU16XmlHdf5(self):
        input_path = test_data_path + '/unit_test/recording_20140729_145048.xml'

        movie = isx.Movie.read(input_path)

        exp_period = isx.Duration._from_num_den(45, 1000);
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_secs(1406670648));
        exp_dropped = [20, 30, 31, 32, 33, 34]
        assert movie.timing == isx.Timing(num_samples=66, period=exp_period, start=exp_start, dropped=exp_dropped);
        assert movie.data_type == np.uint16

        frame0_data = movie.get_frame_data(0)
        np.testing.assert_array_equal(frame0_data, 4094 * np.ones(frame0_data.shape, dtype=np.uint16))


    def test_ReadMovieU16XmlHdf5s(self):
        input_path = test_data_path + '/unit_test/recording_20160706_132714.xml'

        movie = isx.Movie.read(input_path)

        exp_period = isx.Duration._from_num_den(1000, 10010);
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1467811643999));
        assert movie.timing == isx.Timing(num_samples=82, period=exp_period, start=exp_start);
        assert movie.data_type == np.uint16

        f0 = movie.get_frame_data(0)
        assert f0[0, 0] == 1416
        assert f0[182, 132] == 2123
        assert f0[396, 435] == 2283


    def test_ReadMovieWithDroppedCropped(self):
        input_path = test_data_path + '/unit_test/cropped/recording_20161104_145443-TPC.isxd';

        movie = isx.Movie.read(input_path)

        exp_period = isx.Duration._from_num_den(100, 1500)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1478271283662))
        exp_timing = isx.Timing(num_samples=40, period=exp_period, start=exp_start,
                dropped=[10], cropped=[[2, 6], [12, 12], [19, 34]])

        assert movie.timing == exp_timing

    def test_ReadMovieNoFrameTimestamps(self):
        input_path = test_data_path + '/unit_test/guilded/exp_mosaicReadMovieF32.isxd'
        
        movie = isx.Movie.read(input_path)

        with pytest.raises(Exception) as error:
            movie.get_frame_timestamp(0)

        assert 'No frame timestamps stored in movie.' in str(error.value)

    def test_ReadMovieFrameTimestamps(self):
        input_path = test_data_path + '/unit_test/baseplate/2021-06-14-13-30-29_video_green.isxd'

        movie = isx.Movie.read(input_path)

        exp_first_timestamp = 2845042412112
        exp_last_timestamp = 2845044110777

        assert movie.get_frame_timestamp(0) == exp_first_timestamp
        assert movie.get_frame_timestamp(movie.timing.num_samples - 1) == exp_last_timestamp

    def test_ReadNVisionMovie(self):
        input_path = test_data_path + '/unit_test/nVision/20220401-022845-KTM-RQEHB_10_secs.isxb'
        movie = isx.Movie.read(input_path)

        # verify file path
        assert movie.file_path == input_path

        # verify timing info
        exp_period = isx.Duration._from_num_den(9968014, 299000000)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1648798026332), 25200)
        exp_num_samples = 300
        exp_timing = isx.Timing(num_samples=exp_num_samples, period=exp_period, start=exp_start)
        assert movie.timing == exp_timing

        # verify spacing info
        exp_spacing = isx.Spacing(num_pixels=(720, 1280))
        assert movie.spacing == exp_spacing

        # verify data type
        exp_data_type = np.uint8
        assert movie.data_type == exp_data_type
    
        # verify frame data by computing sum of all frames in movie
        # Results of codec are slightly different between windows and linux/mac, but images look very similar
        if platform.system() == "Windows":
            exp_frame_sum = 11687268770
        else:
            exp_frame_sum = 11687253109
        frame_sum = 0
        for i in range(movie.timing.num_samples):
            frame = movie.get_frame_data(i).astype(np.uint64)
            frame_sum += np.sum(frame, dtype=np.uint64)
        assert frame_sum == exp_frame_sum

        # verify first and last tsc values
        exp_start_tsc = 215738669569
        exp_last_tsc = 215748637583
        assert movie.get_frame_timestamp(0) == exp_start_tsc
        assert movie.get_frame_timestamp(movie.timing.num_samples - 1) == exp_last_tsc

        # verify acquisition info
        exp_acquistion_info = {
            'Animal Date of Birth': '', 'Animal Description': '', 'Animal ID': '', 'Animal Sex': 'm', 'Animal Species': '', 'Animal Weight': 0,
            'Camera Brightness': 0, 'Camera Contrast': 32, 'Camera Gain': 0, 'Camera Name': 'camera-1', 'Camera Saturation': 64, 'Camera Serial Number': 'KTM-RQEHB',
            'Miniscope Paired': False
        }
        assert movie.get_acquisition_info() == exp_acquistion_info

    def test_ReadNVisionMovieOutOfBounds(self):
        input_path = test_data_path + '/unit_test/nVision/20220401-022845-KTM-RQEHB_10_secs.isxb'
        movie = isx.Movie.read(input_path)

        with pytest.raises(Exception) as error:
            movie.get_frame_data(movie.timing.num_samples)
        assert 'Failed to read frame from file. Index is out of bounds.' in str(error.value)

        with pytest.raises(Exception) as error:
            movie.get_frame_timestamp(movie.timing.num_samples)
        assert 'Failed to read frame timestamp from file. Index is out of bounds.' in str(error.value)
    
    def test_ReadNVisionMovieDropped(self):
        input_path = test_data_path + '/unit_test/nVision/2022-04-18-21-48-13-camera-1_dropped.isxb'
        movie = isx.Movie.read(input_path)

        # verify timing info
        exp_period = isx.Duration._from_num_den(37712004, 1131000000)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1650343693459), 25200)
        exp_num_samples = 1132
        exp_dropped = [
            186, 188, 189, 190, 192, 193, 194, 196, 197, 198, 200, 201, 202, 204, 205,
            206, 208, 209, 210, 212, 213, 214, 216, 217, 218, 220, 221, 223, 224, 226,
            227, 228, 230, 231, 232, 234, 235, 237, 238, 239, 241, 242, 244, 245, 247,
            248, 249, 251, 252, 254, 255, 256, 258, 259, 260, 262, 263, 265, 266, 267,
            269, 270, 271, 273, 274, 276, 277, 278, 280, 281, 283, 284, 286, 287, 288,
            290, 291, 293, 294, 295, 297, 298, 300, 301, 303, 304, 305, 307, 308, 309,
            311, 313, 314, 315, 317, 318, 319, 321, 322, 324, 325, 326, 328, 329, 331,
            332, 334, 335, 336, 338, 339, 340, 342, 343, 345, 346, 347, 349, 350, 352,
            353, 354, 355, 357, 358, 359, 361, 362, 364, 365, 366, 368, 369, 370, 372,
            373, 375, 376, 378, 379, 380, 382, 383, 385, 386, 387, 389, 390, 392, 393,
            394, 396, 397, 398, 400, 401, 403, 404, 405, 407, 408, 410, 411, 412, 414,
            415, 417, 418, 419, 421, 422, 423, 425, 426, 428, 429, 430, 432, 433, 435,
            436, 438, 439, 440, 442, 443, 444, 446, 447, 449, 450, 451, 453, 454, 455,
            457, 458, 459, 461, 462, 464, 465, 466, 468, 470, 471, 472, 474, 475, 477,
            478, 479, 481, 482, 484, 485, 486, 488, 489, 491, 492, 494, 495, 496, 498,
            499, 500, 502, 503, 505, 506, 507, 509, 510, 512, 513, 514, 516, 517, 519,
            520, 521, 523, 524, 526, 527, 529, 530, 531, 533, 534, 535, 537, 538
        ]
        exp_timing = isx.Timing(num_samples=exp_num_samples, period=exp_period, start=exp_start, dropped=exp_dropped)
        assert movie.timing == exp_timing

        # verify frame data by computing sum of a subset of frames in the movie
        # Results of codec are slightly different between windows and linux/mac, but images look very similar
        if platform.system() == "Windows":
            exp_frame_sum = 1100566302
        else:
            exp_frame_sum = 1100566002
        frame_sum = 0
        start_frame = 180
        num_frames = 10
        for i in range(start_frame, start_frame + num_frames):
            frame = movie.get_frame_data(i).astype(np.uint64)
            frame_sum += np.sum(frame, dtype=np.uint64)
        assert frame_sum == exp_frame_sum

        # verify tsc values and check the value is zero for dropped frames
        exp_start_tsc = 38971101006
        exp_last_tsc = 39008813010
        assert movie.get_frame_timestamp(0) == exp_start_tsc
        assert movie.get_frame_timestamp(movie.timing.num_samples - 1) == exp_last_tsc
        for i in exp_dropped:
            assert movie.get_frame_timestamp(i) == 0

    def test_WriteNVisionMovie(self):
        input_path = test_data_path + '/unit_test/nVision/test.isxb'

        with pytest.raises(Exception) as error:
            movie = isx.Movie.write(input_path, isx.Timing(), isx.Spacing(), np.float32)

        assert 'Cannot write isxb movies.' in str(error.value)

    def test_MovieStrValid(self):
        movie = isx.Movie.read(test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd')
        assert isinstance(str(movie), str)


    def test_MovieStrInvalid(self):
        movie = isx.Movie()
        assert isinstance(str(movie), str)


    def test_WriteMovieU16(self):
        output_path = test_data_path + '/unit_test/output/test_write_outputU16.isxd'
        delete_files_silently([output_path])

        start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(146776827382931))
        timing = isx.Timing(num_samples=9, period=isx.Duration.from_msecs(50), start=start, dropped=[2, 3, 5])
        spacing = isx.Spacing(num_pixels=(3, 3))
        data_type = np.uint16
        frames = np.random.randint(low=0, high=4095, size=[*spacing.num_pixels, timing.num_samples], dtype=data_type)

        movie = isx.Movie.write(output_path, timing, spacing, data_type=data_type)
        for i in timing.get_valid_samples():
            movie.set_frame_data(i, frames[:, :, i])
        movie.flush()

        movie = isx.Movie.read(output_path)

        assert movie.spacing == spacing
        assert movie.timing == timing
        assert movie.data_type == np.uint16

        for i in timing.get_valid_samples():
            np.testing.assert_array_equal(movie.get_frame_data(i), frames[:, :, i])

        del movie
        delete_files_silently([output_path])

    def test_WriteMovieF32(self):
        output_path = test_data_path + '/unit_test/output/test_write_outputF32.isxd'
        delete_files_silently([output_path])

        start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1467768443283))
        timing = isx.Timing(num_samples=12, period=isx.Duration.from_msecs(50), start=start,
                dropped=[1, 11], cropped=[[3, 5], [8, 9]])
        spacing = isx.Spacing(num_pixels=(32, 57))
        data_type = np.float32
        frames = np.random.randn(spacing.num_pixels[0], spacing.num_pixels[1], timing.num_samples).astype(data_type)

        movie = isx.Movie.write(output_path, timing, spacing, data_type=data_type)
        for k in timing.get_valid_samples():
            movie.set_frame_data(k, frames[:, :, k])
        movie.flush()

        movie = isx.Movie.read(output_path)

        assert movie.timing == timing
        assert movie.spacing == spacing
        assert movie.data_type == np.float32

        for k in timing.get_valid_samples():
            np.testing.assert_array_equal(movie.get_frame_data(k), frames[:, :, k])

        del movie
        delete_files_silently([output_path])

    @pytest.mark.movie
    @pytest.mark.parametrize('frame_data_type', data_types)
    @pytest.mark.parametrize('movie_data_type', ('uint16', 'float32'))
    def test_WriteMovieOtherTypeToFloat32(self, frame_data_type, movie_data_type):
        output_path = test_data_path + '/unit_test/output/test_write_outputF32.isxd'
        delete_files_silently([output_path])

        start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1467768443282))
        timing = isx.Timing(num_samples=5, period=isx.Duration.from_msecs(52), start=start)
        spacing = isx.Spacing(num_pixels=(5, 3))
        frames = np.random.randn(spacing.num_pixels[0], spacing.num_pixels[1],
                                 timing.num_samples).astype(frame_data_type)

        movie = isx.Movie.write(output_path, timing, spacing, data_type=np.__getattribute__(movie_data_type))
        for k in timing.get_valid_samples():
            if frame_data_type != movie_data_type:
                with pytest.warns(UserWarning) as warnings:
                    movie.set_frame_data(k, frames[:, :, k])
                    assert 'Converting from {0} to {1}.'.format(frame_data_type, movie_data_type) in [str(x.message) for x in warnings]
            else:
                with w.catch_warnings():
                    w.simplefilter("error")
                    movie.set_frame_data(k, frames[:, :, k])

        movie.flush()

        movie = isx.Movie.read(output_path)

        assert movie.timing == timing
        assert movie.spacing == spacing
        assert movie.data_type == np.dtype(movie_data_type)

        for k in timing.get_valid_samples():
            np.testing.assert_array_equal(movie.get_frame_data(k), frames[:, :, k].astype(movie_data_type))

        del movie
        delete_files_silently([output_path])


    @pytest.mark.tiff_movie
    def test_MovieTiffExporter(self):
        movie_file_path = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        tiff_file_path = test_data_path + '/unit_test/output/test_output.tiff'
        expected_file_path = test_data_path + '/unit_test/guilded/exp_mosaicMovieTiffExporter_output-v2.tiff'

        delete_files_silently([tiff_file_path])

        isx.export_movie_to_tiff(movie_file_path, tiff_file_path)

        assert_tiff_files_equal_by_path(expected_file_path, tiff_file_path)

        delete_files_silently([tiff_file_path])

    @pytest.mark.tiff_movie
    def test_MovieTiffExporterWithInvalid(self):
        input_movie_file_path = test_data_path + '/unit_test/cropped/recording_20161104_145443-TPC.isxd'
        output_movie_file_path = test_data_path + '/unit_test/output/output.tif'

        delete_files_silently([output_movie_file_path])
        isx.export_movie_to_tiff(input_movie_file_path, output_movie_file_path, write_invalid_frames=True)

        input_movie = isx.Movie.read(input_movie_file_path)
        output_movie = isx.Movie.read(output_movie_file_path)

        num_pixels = input_movie.spacing.num_pixels
        num_samples = input_movie.timing.num_samples
        valid_samples_mask = input_movie.timing.get_valid_samples_mask()
        assert num_samples == output_movie.timing.num_samples
        assert len(input_movie.timing.dropped) > 0
        assert len(input_movie.timing.cropped) > 0
        for i in range(num_samples):
            if valid_samples_mask[i]:
                np.testing.assert_array_equal(output_movie.get_frame_data(i), input_movie.get_frame_data(i))
            else:
                np.testing.assert_array_equal(output_movie.get_frame_data(i), np.zeros(num_pixels, np.uint16))

        del output_movie
        delete_files_silently([output_movie_file_path])


    @pytest.mark.nwb_movie
    def test_MovieExporter(self):
        movie_file_path = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_nwb_path = test_data_path + '/unit_test/output/test_output.nwb'
        expected_nwb_path = test_data_path + '/unit_test/guilded/exp_mosaicMovieExporter_output-v3.nwb'

        delete_files_silently([output_nwb_path])

        identifier = 'recording_20160426_145041 recording_20160426_145041 NWB-1.0.6 2018-01-03T11:57:36.590-08:00'
        description = 'Exported from Inscopix Data Processing.'

        isx.export_movie_to_nwb(movie_file_path, output_nwb_path, identifier=identifier, session_description=description)

        output_nwb = h5py.File(output_nwb_path, 'r')
        expected_nwb = h5py.File(expected_nwb_path, 'r')
        compare_h5_groups(output_nwb['/'], expected_nwb['/'], ['file_create_date'])

        output_nwb.close()
        delete_files_silently([output_nwb_path])

    @pytest.mark.mp4_movie
    def test_MovieMp4ExporterIsxd(self):
        movie_file_path = test_data_path + '/unit_test/cropped/recording_20161104_145443-TPC.isxd'
        mp4_file_path = test_data_path + '/unit_test/output/test_output.mp4'

        delete_files_silently([mp4_file_path])

        isx.export_movie_to_mp4(
            movie_file_path,
            mp4_file_path,
            compression_quality=0.1,
            write_invalid_frames=False
        )

        # verify size of file
        # Results of codec are slightly different between windows and linux/mac, but images look very similar
        if platform.system() == "Windows":
            exp_file_bytes = 1028858
        else:
            exp_file_bytes = 1028817
        assert os.path.getsize(mp4_file_path) == exp_file_bytes

        delete_files_silently([mp4_file_path])

    @pytest.mark.mp4_movie
    def test_MovieMp4ExporterIsxdWithInvalid(self):
        movie_file_path = test_data_path + '/unit_test/cropped/recording_20161104_145443-TPC.isxd'
        mp4_file_path = test_data_path + '/unit_test/output/test_output.mp4'

        delete_files_silently([mp4_file_path])

        isx.export_movie_to_mp4(
            movie_file_path,
            mp4_file_path,
            compression_quality=0.1,
            write_invalid_frames=True
        )

        # verify size of file
        # Results of codec are slightly different between windows and linux/mac, but images look very similar
        if platform.system() == "Windows":
            exp_file_bytes = 1286184
        else:
            exp_file_bytes = 1285643
        assert os.path.getsize(mp4_file_path) == exp_file_bytes

        delete_files_silently([mp4_file_path])

    @pytest.mark.mp4_movie
    def test_MovieMp4ExporterIsxb(self):
        movie_file_path = test_data_path + '/unit_test/nVision/20220412-200447-camera-100.isxb'
        mp4_file_path = test_data_path + '/unit_test/output/test_output.mp4'

        delete_files_silently([mp4_file_path])

        isx.export_movie_to_mp4(
            movie_file_path,
            mp4_file_path,
            compression_quality=0.1,
            write_invalid_frames=False
        )

        # verify size of file
        # Results of codec are slightly different between windows and linux/mac, but images look very similar
        if platform.system() == "Windows":
            exp_file_bytes = 6991088
        else:
            exp_file_bytes = 6977564
        assert os.path.getsize(mp4_file_path) == exp_file_bytes

        delete_files_silently([mp4_file_path])

    @pytest.mark.mp4_movie
    def test_MovieMp4ExporterIsxbTracking(self):
        movie_file_paths = [
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_0.isxb",
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_1.isxb"
        ]
        mp4_file_path = test_data_path + '/unit_test/output/test_output.mp4'

        delete_files_silently([mp4_file_path])

        isx.export_movie_to_mp4(
            movie_file_paths,
            mp4_file_path,
            compression_quality=0.1,
            write_invalid_frames=False,
            draw_bounding_box=True,
            draw_bounding_box_center=True,
            draw_zones=True
        )

        # verify size of file
        # Results of codec are slightly different between windows and linux/mac, but images look very similar
        if platform.system() == "Windows":
            exp_file_bytes = 992197
        else:
            exp_file_bytes = 990564
        assert os.path.getsize(mp4_file_path) == exp_file_bytes

        delete_files_silently([mp4_file_path])
    
    @pytest.mark.mp4_movie
    def test_MovieMp4ExporterIntFrameRate(self):
        movie_file_path = test_data_path + '/unit_test/nVision/20220412-200447-camera-100.isxb'
        mp4_file_path = test_data_path + '/unit_test/output/test_output.mp4'

        delete_files_silently([mp4_file_path])

        isx.export_movie_to_mp4(
            movie_file_path,
            mp4_file_path,
            compression_quality=0.1,
            write_invalid_frames=False,
            frame_rate_format='int'
        )

        # verify size of file
        # Results of codec are slightly different between windows and linux/mac, but images look very similar
        if platform.system() == "Windows":
            exp_file_bytes = 6990927
        else:
            exp_file_bytes = 6977407
        assert os.path.getsize(mp4_file_path) == exp_file_bytes

        delete_files_silently([mp4_file_path])

    @pytest.mark.mp4_movie
    def test_MovieTimestampExporterIsxdNoTimestamps(self):
        movie_file_path = test_data_path + '/unit_test/cnmfe-cpp/movie_128x128x1000.isxd'
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        with pytest.raises(Exception) as error:
            isx.export_movie_timestamps_to_csv(
                movie_file_path,
                csv_file_path,
                time_ref='tsc')
        assert 'Input movie does not have frame timestamps stored in file.' in str(error.value)

    def test_MovieTimestampExporterIsxdSeries(self):
        movie_file_paths = [
            test_data_path + '/unit_test/baseplate/2021-06-28-23-45-49_video_sched_0_probe_custom.isxd',
            test_data_path + '/unit_test/baseplate/2021-06-28-23-34-09_video_sched_0_probe_none.isxd',
        ]
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        isx.export_movie_timestamps_to_csv(
            movie_file_paths,
            csv_file_path,
            time_ref='tsc')

        df = pd.read_csv(csv_file_path)
        assert (df.iloc[0] == pd.DataFrame({
            'Global Frame Number' : [0],
            'Movie Number' : [0],
            'Local Frame Number' : [0],
            'Frame Timestamp (us)' : [4170546756640]}
        )).all(axis=None)

        assert (df.iloc[len(df.index) - 1] == pd.DataFrame({
            'Global Frame Number' : [53],
            'Movie Number' : [1],
            'Local Frame Number' : [26],
            'Frame Timestamp (us)' : [4171250265074]}
        )).all(axis=None)

        delete_files_silently([csv_file_path])

    def test_MovieTimestampExporterIsxb(self):
        movie_file_path = test_data_path + '/unit_test/nVision/20220412-200447-camera-100.isxb'
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        isx.export_movie_timestamps_to_csv(
            movie_file_path,
            csv_file_path,
            time_ref='tsc')

        df = pd.read_csv(csv_file_path)
        assert (df.iloc[0] == pd.DataFrame({
            'Global Frame Number' : [0],
            'Movie Number' : [0],
            'Local Frame Number' : [0],
            'Frame Timestamp (us)' : [115829025489]}
        )).all(axis=None)

        assert (df.iloc[len(df.index) - 1] == pd.DataFrame({
            'Global Frame Number' : [112],
            'Movie Number' : [0],
            'Local Frame Number' : [112],
            'Frame Timestamp (us)' : [115832757521]}
        )).all(axis=None)

        delete_files_silently([csv_file_path])
    
    def test_MovieTimestampExporterUnix(self):
        movie_file_path = test_data_path + '/unit_test/nVision/20220412-200447-camera-100.isxb'
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        isx.export_movie_timestamps_to_csv(
            movie_file_path,
            csv_file_path,
            time_ref='unix')

        df = pd.read_csv(csv_file_path)
        assert (df.iloc[0] == pd.DataFrame({
            'Global Frame Number' : [0],
            'Movie Number' : [0],
            'Local Frame Number' : [0],
            'Frame Timestamp (s)' : [1649819290.471000]}
        )).all(axis=None)

        assert (df.iloc[len(df.index) - 1] == pd.DataFrame({
            'Global Frame Number' : [112],
            'Movie Number' : [0],
            'Local Frame Number' : [112],
            'Frame Timestamp (s)' : [1649819294.203032]}
        )).all(axis=None)

        delete_files_silently([csv_file_path])
    
    def test_MovieTimestampExporterStart(self):
        movie_file_path = test_data_path + '/unit_test/nVision/20220412-200447-camera-100.isxb'
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        isx.export_movie_timestamps_to_csv(
            movie_file_path,
            csv_file_path,
            time_ref='start')

        df = pd.read_csv(csv_file_path)
        assert (df.iloc[0] == pd.DataFrame({
            'Global Frame Number' : [0],
            'Movie Number' : [0],
            'Local Frame Number' : [0],
            'Frame Timestamp (s)' : [0.000000]}
        )).all(axis=None)

        assert (df.iloc[len(df.index) - 1] == pd.DataFrame({
            'Global Frame Number' : [112],
            'Movie Number' : [0],
            'Local Frame Number' : [112],
            'Frame Timestamp (s)' : [3.732032]}
        )).all(axis=None)

        delete_files_silently([csv_file_path])

    @pytest.mark.mp4_movie
    def test_NVisionMovieTrackingFrameDataExporter(self):
        movie_file_paths = [
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_0.isxb",
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_1.isxb"
        ]
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        isx.export_nvision_movie_tracking_frame_data_to_csv(
            movie_file_paths,
            csv_file_path,
            time_ref='start'
        )

        with open(csv_file_path, 'r') as f:
            lines = f.read().splitlines()
        
        expected_columns = "Global Frame Number,Movie Number,Local Frame Number,Frame Timestamp (s),Bounding Box Left,Bounding Box Top,Bounding Box Right,Bounding Box Bottom,Bounding Box Center X,Bounding Box Center Y,Confidence,Zone ID,Zone Name,Zone Event,Zone Trigger"
        assert lines[0] == expected_columns

        expected_first_line = "0,0,0,0.000000,526.136230,682.003479,650.984802,908.188293,588.560547,795.095886,67.986031,,,,"
        assert lines[1] == expected_first_line

        expected_last_line = "19,1,9,0.631984,528.115173,776.796631,699.851135,912.584290,613.983154,844.690430,98.499191,4270701760,ZONE#1 rectangle,,"
        assert lines[-1] == expected_last_line

        delete_files_silently([csv_file_path])
    
    @pytest.mark.mp4_movie
    def test_NVisionMovieTrackingFrameDataExporterTsc(self):
        movie_file_paths = [
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_0.isxb",
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_1.isxb"
        ]
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        isx.export_nvision_movie_tracking_frame_data_to_csv(
            movie_file_paths,
            csv_file_path,
            time_ref='tsc'
        )

        with open(csv_file_path, 'r') as f:
            lines = f.read().splitlines()
        
        expected_columns = "Global Frame Number,Movie Number,Local Frame Number,Frame Timestamp (us),Bounding Box Left,Bounding Box Top,Bounding Box Right,Bounding Box Bottom,Bounding Box Center X,Bounding Box Center Y,Confidence,Zone ID,Zone Name,Zone Event,Zone Trigger"
        assert lines[0] == expected_columns

        expected_first_line = "0,0,0,163957519943,526.136,682.003,650.985,908.188,588.561,795.096,67.986,,,,"
        assert lines[1] == expected_first_line

        expected_last_line = "19,1,9,163958151927,528.115,776.797,699.851,912.584,613.983,844.69,98.4992,4270701760,ZONE#1 rectangle,,"
        assert lines[-1] == expected_last_line

        delete_files_silently([csv_file_path])
    
    @pytest.mark.mp4_movie
    def test_NVisionMovieTrackingFrameDataExporterUnix(self):
        movie_file_paths = [
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_0.isxb",
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_1.isxb"
        ]
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        isx.export_nvision_movie_tracking_frame_data_to_csv(
            movie_file_paths,
            csv_file_path,
            time_ref='unix'
        )
        
        with open(csv_file_path, 'r') as f:
            lines = f.read().splitlines()
        
        expected_columns = "Global Frame Number,Movie Number,Local Frame Number,Frame Timestamp (s),Bounding Box Left,Bounding Box Top,Bounding Box Right,Bounding Box Bottom,Bounding Box Center X,Bounding Box Center Y,Confidence,Zone ID,Zone Name,Zone Event,Zone Trigger"
        assert lines[0] == expected_columns

        expected_first_line = "0,0,0,1705049721.643000,526.136230,682.003479,650.984802,908.188293,588.560547,795.095886,67.986031,,,,"
        assert lines[1] == expected_first_line

        expected_last_line = "19,1,9,1705049722.274984,528.115173,776.796631,699.851135,912.584290,613.983154,844.690430,98.499191,4270701760,ZONE#1 rectangle,,"
        assert lines[-1] == expected_last_line

        delete_files_silently([csv_file_path])
    
    @pytest.mark.mp4_movie
    def test_NVisionMovieTrackingZoneDataExporter(self):
        movie_file_paths = [
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_0.isxb",
            test_data_path + "/unit_test/nVision/tracking/Group-20240111-080531_2024-01-12-08-55-21_sched_1.isxb"
        ]
        csv_file_path = test_data_path + '/unit_test/output/test_output.csv'

        delete_files_silently([csv_file_path])

        isx.export_nvision_movie_tracking_zone_data_to_csv(
            movie_file_paths,
            csv_file_path
        )

        with open(csv_file_path, 'r') as f:
            lines = f.read().splitlines()
        
        expected_columns = "ID,Enabled,Name,Description,Type,X 0,Y 0,X 1,Y 1,X 2,Y 2,X 3,Y 3,X 4,Y 4,Major Axis, Minor Axis, Angle"
        assert lines[0] == expected_columns

        expected_first_line = "1705077750976,1,ZONE#1 rectangle,,rectangle,534.135,387.9,993.203,387.9,993.203,868.86,534.135,868.86,,,,,"
        assert lines[1] == expected_first_line

        expected_last_line = "1705077943271,1,ZONE#4 Elipse,,ellipse,1273.26,241.02,,,,,,,,,293.76,98.1654,90"
        assert lines[-1] == expected_last_line

        delete_files_silently([csv_file_path])

    @pytest.mark.csv_trace
    def test_EventSetExporterDense(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_event_set = unit_test_dir + '/guilded/exp_mosaicEventDetection_output-v2.isxd'
        output_csv = unit_test_dir + '/output/output.csv'

        delete_files_silently([output_csv])

        isx.export_event_set_to_csv([input_event_set], output_csv, 'start', sparse_output=False)

        expected_csv = unit_test_dir + '/guilded/exp_mosaicEventSetExporter.csv'
        assert_csv_events_are_equal_by_path(expected_csv, output_csv)

        assert not os.path.exists(unit_test_dir + '/output/output-props.csv')

        delete_files_silently([output_csv])

    @pytest.mark.csv_trace
    def test_EventSetExporterSparse(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_event_set = unit_test_dir + '/guilded/exp_mosaicEventDetection_output-v2.isxd'
        output_csv = unit_test_dir + '/output/output.csv'

        delete_files_silently([output_csv])

        isx.export_event_set_to_csv([input_event_set], output_csv, 'start', sparse_output=True)

        expected_csv = unit_test_dir + '/guilded/exp_mosaicEventSetExporterSparse.csv'
        assert_csv_events_are_equal_by_path(expected_csv, output_csv)

        assert not os.path.exists(unit_test_dir + '/output/output-props.csv')

        delete_files_silently([output_csv])

    @pytest.mark.csv_trace
    def test_EventSetExporterSparseBinary(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_event_set = unit_test_dir + '/guilded/exp_mosaicEventDetection_output-v2.isxd'
        output_csv = unit_test_dir + '/output/output.csv'

        delete_files_silently([output_csv])

        isx.export_event_set_to_csv([input_event_set], output_csv, 'start', sparse_output=True, write_amplitude=False)

        expected_csv = unit_test_dir + '/guilded/exp_mosaicEventSetExporterSparseBinary.csv'
        assert_csv_events_are_equal_by_path(expected_csv, output_csv)

        assert not os.path.exists(unit_test_dir + '/output/output-props.csv')

        delete_files_silently([output_csv])

    @pytest.mark.csv_trace
    def test_EventSetExporterWithProps(self):
        unit_test_dir = test_data_path + '/unit_test'
        test_dir = unit_test_dir + '/events-export'
        input_event_sets = ['{}/50fr10_l{}-3cells_he-ROI-LCR-ED.isxd'.format(test_dir, i) for i in range(1, 4)]
        output_dir = unit_test_dir + '/output'
        output_csv = output_dir + '/output.csv'
        output_props = output_dir + '/props.csv'

        delete_files_silently([output_csv, output_props])

        isx.export_event_set_to_csv(input_event_sets, output_csv, 'start', output_props_file=output_props)

        exp_props = unit_test_dir + '/guilded/exp_EventSetExporterWithProps-v2.csv'
        assert_csv_files_are_equal_by_path(exp_props, output_props)

        delete_files_silently([output_csv, output_props])

    @pytest.mark.csv_trace
    def test_GpioSetExporter(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_gpio_set = unit_test_dir + '/gpio/2020-05-20-10-33-22_video.gpio'
        output_csv = unit_test_dir + '/output/output.csv'
        intermediate_isxd_path = '/tmp/2020-05-20-10-33-22_video_gpio.isxd'
        intermediate_isxd_windows_path = unit_test_dir + '/gpio/2020-05-20-10-33-22_video_gpio.isxd'

        delete_files_silently([output_csv, intermediate_isxd_path])

        isx.export_gpio_set_to_csv([input_gpio_set], output_csv, inter_isxd_file_dir='/tmp', time_ref='start')

        expected_csv = unit_test_dir + '/gpio/2020-05-20-10-33-22_video_gpio.csv'
        assert_csv_events_are_equal_by_path(expected_csv, output_csv)

        assert os.path.exists(intermediate_isxd_path) or os.path.exists(intermediate_isxd_windows_path)

        delete_files_silently([output_csv, intermediate_isxd_path])

    @pytest.mark.csv_trace
    def test_ImuSetExporter(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_gpio_set = unit_test_dir + '/gpio/2020-05-20-10-33-22_video.imu'
        output_csv = unit_test_dir + '/output/output.csv'
        intermediate_isxd_path = '/tmp/2020-05-20-10-33-22_video_imu.isxd'
        intermediate_isxd_windows_path = unit_test_dir + '/gpio/2020-05-20-10-33-22_video_imu.isxd'

        delete_files_silently([output_csv, intermediate_isxd_path])

        isx.export_gpio_set_to_csv([input_gpio_set], output_csv, inter_isxd_file_dir='/tmp', time_ref='start')

        expected_csv = unit_test_dir + '/gpio/2020-05-20-10-33-22_video_imu.csv'
        assert_csv_events_are_equal_by_path(expected_csv, output_csv)

        assert os.path.exists(intermediate_isxd_path) or os.path.exists(intermediate_isxd_windows_path)

        delete_files_silently([output_csv, intermediate_isxd_path])

    @pytest.mark.csv_trace
    def test_ImuIsxdSetExporter(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_gpio_set = unit_test_dir + '/gpio/2020-05-20-10-33-22_video_imu.isxd'
        output_csv = unit_test_dir + '/output/output.csv'

        delete_files_silently([output_csv])

        isx.export_gpio_set_to_csv([input_gpio_set], output_csv, inter_isxd_file_dir='/tmp', time_ref='start')

        expected_csv = unit_test_dir + '/gpio/2020-05-20-10-33-22_video_imu.csv'
        assert_csv_events_are_equal_by_path(expected_csv, output_csv)

        delete_files_silently([output_csv])

    @pytest.mark.isxd_events
    def test_ReadEvent(self):
        input_file = test_data_path + '/unit_test/guilded/exp_mosaicEventDetection_output-v2.isxd'
        event = isx.EventSet.read(input_file)

        assert event.num_cells == 3
        exp_period = isx.Duration._from_num_den(1, 10)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_secs(20))
        assert event.timing == isx.Timing(num_samples=50, period=exp_period, start=exp_start)

    @pytest.mark.isxd_events
    def test_GetEventData(self):
        input_file = test_data_path + '/unit_test/guilded/exp_mosaicEventDetection_output-v2.isxd'
        event = isx.EventSet.read(input_file)

        time_C0, amplitude_C0 = event.get_cell_data(0)

        assert time_C0[0] == 200000
        np.testing.assert_approx_equal(1.446234, amplitude_C0[0], significant=6)

        time_C2, amplitude_C2 = event.get_cell_data(2)

        assert time_C2[0] == 0
        np.testing.assert_approx_equal(1.238986, amplitude_C2[0], significant=6)

    @pytest.mark.isxd_events
    def test_GetEventCellIndex(self):
        input_file = test_data_path + '/unit_test/guilded/exp_mosaicEventDetection_output-v2.isxd'
        event = isx.EventSet.read(input_file)

        assert event.get_cell_index('C0') == 0
        assert event.get_cell_index('C1') == 1

    @pytest.mark.isxd_events
    def test_GetEventData_iterate_cells(self):
        # test event reading on a bigger file with more events and cells
        input_file = test_data_path + '/unit_test/event_detection/recording_20161006_111406-PP-bp-mc-dff-pcaica-events.isxd'
        exp_file = test_data_path + '/unit_test/event_detection/recording_20161006_111406-PP-bp-mc-dff-pcaica-events.csv'

        event = isx.EventSet.read(input_file)
        event_df = pd.read_csv(exp_file)
        cnames = [event.get_cell_name(k) for k in range(event.num_cells)]

        num_cells_with_zero_events = 0
        for index, cname in enumerate(cnames):
            # get expected event times and amplitudes for the cell, sort
            i = event_df[' Cell Name'] == ' {}'.format(cname)
            if i.sum() == 0:
                num_cells_with_zero_events += 1
                continue

            # compare expected event times and amplitudes with those returned from API
            event_times, event_amps = event.get_cell_data(index)

            np.testing.assert_allclose(event_times / 1e6, event_df['Time (s)'][i].values)
            np.testing.assert_allclose(event_amps, event_df[' Value'][i].values, rtol=1e-5)

        # this is just a check to make sure nothing stupid happened
        assert num_cells_with_zero_events < len(cnames)

    @pytest.mark.isxd_events
    @pytest.mark.parametrize('method', ('get_cell_data', 'get_cell_name'))
    @pytest.mark.parametrize('cell_index', (424, 500000000, 18446744073709551615))
    def test_GetEventData_bad_cell_index_int(self, method, cell_index):
        input_file = test_data_path + '/unit_test/event_detection/recording_20161006_111406-PP-bp-mc-dff-pcaica-events.isxd'

        event = isx.EventSet.read(input_file)
        with pytest.raises(Exception) as error:
            getattr(event, method)(cell_index)
        assert f'Cell index {cell_index} is too large' in str(error.value)

    @pytest.mark.isxd_events
    @pytest.mark.parametrize('method', ('get_cell_data', 'get_cell_name'))
    @pytest.mark.parametrize('cell_index', (8 + 3j, '0', '423', 'My Cell Name', 1.5, (1, 5), [3, 5], {4, 9}, {6: 5}))
    def test_GetEventData_bad_cell_index_type(self, method, cell_index):
        input_file = test_data_path + '/unit_test/event_detection/recording_20161006_111406-PP-bp-mc-dff-pcaica-events.isxd'

        event = isx.EventSet.read(input_file)
        with pytest.raises(Exception) as error:
            getattr(event, method)(cell_index)
        assert ("argument 2:" in str(error.value) and "wrong type" in str(error.value)) or "cannot be interpreted as an integer" in str(error.value)

    @pytest.mark.isxd_events
    def test_GetEventCellNames(self):
        input_file = test_data_path + '/unit_test/guilded/exp_mosaicEventDetection_output-v2.isxd'
        event = isx.EventSet.read(input_file)

        cell_1 = event.get_cell_name(0)
        cell_2 = event.get_cell_name(1)
        cell_3 = event.get_cell_name(2)

        assert cell_1 == 'C0'
        assert cell_2 == 'C1'
        assert cell_3 == 'C2'

    @pytest.mark.isxd_events
    def test_WriteEvents(self):
        output_path = test_data_path + '/unit_test/output/test_write_events.isxd'
        delete_files_silently([output_path])

        timing = isx.Timing(num_samples=5, period=isx.Duration.from_msecs(10))
        events = isx.EventSet.write(output_path, timing, ['Cell_1'])

        timestamps = np.array([1, 2, 3, 4, 5]).astype(np.uint64)
        data = np.random.rand(timing.num_samples).astype(np.float32)

        events.set_cell_data(0, timestamps, data)
        events.flush()

        events = isx.EventSet.read(output_path)

        assert events.num_cells == 1
        assert events.timing == timing

        act_timestamps, act_data = events.get_cell_data(0)

        assert act_timestamps[0] == timestamps[0]
        assert act_timestamps[1] == timestamps[1]
        np.testing.assert_approx_equal(act_data[0], data[0], significant=6)
        np.testing.assert_approx_equal(act_data[1], data[1], significant=6)

        del events
        delete_files_silently([output_path])

    def test_WriteEmptyEvents(self):
        output_path = test_data_path + '/unit_test/output/test_write_empty_events.isxd'

        cell_names = ['Cell_1']
        timing = isx.Timing(num_samples=10)

        delete_files_silently([output_path])
        events = isx.EventSet.write(output_path, timing, cell_names)

        events.set_cell_data(0, np.array([], np.uint64), np.array([], np.float32))
        events.flush()

        events = isx.EventSet.read(output_path)

        num_cells = len(cell_names)
        assert events.num_cells == num_cells
        assert events.timing == timing

        for c in range(num_cells):
            assert events.get_cell_name(c) == cell_names[c]

        [act_usecs_since_start, act_values] = events.get_cell_data(0)
        assert len(act_usecs_since_start) == 0
        assert len(act_values) == 0

        del events
        delete_files_silently([output_path])


    @pytest.mark.isxd_events
    @pytest.mark.parametrize('event_data_type', data_types)
    @pytest.mark.parametrize('time_data_type', data_types)
    def test_WriteEventsOtherTypeToFloat32(self, event_data_type, time_data_type):
        output_path = test_data_path + '/unit_test/output/test_write_events.isxd'
        delete_files_silently([output_path])

        timing = isx.Timing(num_samples=5, period=isx.Duration.from_msecs(10))
        events = isx.EventSet.write(output_path, timing, ['Cell_1'])

        timestamps = np.array([1, 2, 3, 4, 5]).astype(time_data_type)

        data = np.random.rand(timing.num_samples).astype(event_data_type)

        if time_data_type != 'uint64':
            if event_data_type != 'float32':
                with pytest.warns(UserWarning) as warnings:
                    events.set_cell_data(0, timestamps, data)
                    assert 'Converting from {} to uint64.'.format(time_data_type) in [str(x.message) for x in warnings]
                    assert 'Converting from {} to float32.'.format(event_data_type) in [str(x.message) for x in warnings]
            else:
                with pytest.warns(UserWarning) as warnings:
                    events.set_cell_data(0, timestamps, data)
                    assert 'Converting from {} to uint64.'.format(time_data_type) in [str(x.message) for x in warnings]
        else:
            if event_data_type != 'float32':
                with pytest.warns(UserWarning) as warnings:
                    events.set_cell_data(0, timestamps, data)
                    assert 'Converting from {} to float32.'.format(event_data_type) in [str(x.message) for x in warnings]
            else:
                with w.catch_warnings():
                    w.simplefilter("error")
                    events.set_cell_data(0, timestamps, data)

        events.flush()

        events = isx.EventSet.read(output_path)

        assert events.num_cells == 1
        assert events.timing == timing

        act_timestamps, act_data = events.get_cell_data(0)

        assert act_timestamps[0] == timestamps[0]
        assert act_timestamps[1] == timestamps[1]
        np.testing.assert_approx_equal(act_data[0], data[0], significant=6)
        np.testing.assert_approx_equal(act_data[1], data[1], significant=6)

        del events
        delete_files_silently([output_path])


    def test_EventSetStrValid(self):
        event_set = isx.EventSet.read(test_data_path + '/unit_test/guilded/exp_mosaicEventDetection_output-v2.isxd')
        assert isinstance(str(event_set), str)


    def test_EventSetStrInvalid(self):
        event_set = isx.EventSet()
        assert isinstance(str(event_set), str)


    def test_LosslessTimingSpacing(self):
        movie_file_path = test_data_path + '/unit_test/50fr10_l1-3cells_he-PP.isxd'
        movie = isx.Movie.read(movie_file_path)

        output_dir = test_data_path + '/python/test_lossless_timing_spacing'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        movie_proc_file_path = output_dir + '/movie_proc.isxd'
        movie_proc = isx.Movie.write(movie_proc_file_path, movie.timing, movie.spacing, movie.data_type)
        for f in movie.timing.get_valid_samples():
            movie_proc.set_frame_data(f, movie.get_frame_data(f) * 0.5)
        movie_proc.flush()

        movie_proc = isx.Movie.read(movie_proc_file_path)
        assert movie.timing == movie_proc.timing
        assert movie.spacing == movie_proc.spacing

        movie_proc_data = np.zeros((np.prod(movie.spacing.num_pixels), movie.timing.num_samples), dtype=np.uint16)
        for f in movie_proc.timing.get_valid_samples():
            movie_proc_data[:, f] = movie_proc.get_frame_data(f).flatten()

        cell_set_file_path = output_dir + '/cell_set.isxd'
        cell_set = isx.CellSet.write(cell_set_file_path, movie_proc.timing, movie_proc.spacing)
        num_cells = 3
        cell_names = ['C{}'.format(c) for c in range(num_cells)]
        images = np.zeros(list(movie.spacing.num_pixels) + [movie.timing.num_samples], dtype=np.float32)
        images[14, 9, 1] = 1;
        images[14, 74, 2] = 1;
        images[64, 74, 3] = 1;
        for c in range(num_cells):
            image = images[:, :, c]
            trace = image.flatten().dot(movie_proc_data)
            cell_set.set_cell_data(c, image, trace, cell_names[c])
        cell_set.flush()

        cell_set = isx.CellSet.read(cell_set_file_path)
        assert cell_set.timing == movie_proc.timing
        assert cell_set.spacing == movie_proc.spacing

        event_set_file_path = output_dir + '/event_set.isxd'
        event_set = isx.EventSet.write(event_set_file_path, cell_set.timing, cell_names)
        usecs_since_start = np.array([x.to_usecs() for x in cell_set.timing.get_offsets_since_start()], np.uint64)
        event_set_usecs = []
        event_set_values = []
        for c in range(num_cells):
            trace = cell_set.get_cell_trace_data(c)
            trace_above_thresh = (trace > 1500).nonzero()
            event_set_usecs.append(usecs_since_start[trace_above_thresh])
            event_set_values.append(trace[trace_above_thresh])
            event_set.set_cell_data(c, event_set_usecs[c], event_set_values[c])
        event_set.flush()

        event_set = isx.EventSet.read(event_set_file_path)
        assert event_set.timing == cell_set.timing
        for c in range(num_cells):
            [usecs, values] = event_set.get_cell_data(c)
            np.testing.assert_array_equal(usecs, event_set_usecs[c])
            np.testing.assert_array_equal(values, event_set_values[c])

        del movie
        del movie_proc
        del cell_set
        del event_set

        shutil.rmtree(output_dir)

    @pytest.mark.isxd_gpio
    def test_ReadGpio(self):
        input_file = test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video.gpio'
        gpio = isx.GpioSet.read(input_file)

        assert gpio.num_channels == 26
        exp_period = isx.Duration._from_num_den(1, 1000)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(1589970802786))
        assert gpio.timing == isx.Timing(num_samples=51280, period=exp_period, start=exp_start)

    @pytest.mark.isxd_gpio
    def test_GetGpioData(self):
        input_file = test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video.gpio'
        gpio = isx.GpioSet.read(input_file)

        time_C23, amplitude_C23 = gpio.get_channel_data(23)

        assert time_C23[1] == 76000
        np.testing.assert_approx_equal(500.0, amplitude_C23[1], significant=6)

        time_C24, amplitude_C24 = gpio.get_channel_data(24)

        assert time_C24[1] == 75000
        np.testing.assert_approx_equal(1.0, amplitude_C24[1], significant=6)

    @pytest.mark.isxd_events
    def test_GetGpioChannelIndex(self):
        input_file = test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video.gpio'
        gpio = isx.GpioSet.read(input_file)

        assert gpio.get_channel_index('e-focus') == 23
        assert gpio.get_channel_index('BNC Sync Output') == 24

    @pytest.mark.isxd_gpio
    def test_GetImuData(self):
        input_file = test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video.imu'
        imu = isx.GpioSet.read(input_file)

        time_acc, amp_acc = imu.get_channel_data(0)

        assert time_acc[1] == 20592
        np.testing.assert_approx_equal(-6.1035156e-05, amp_acc[1], significant=6)

        time_ori, amp_ori = imu.get_channel_data(3)

        assert time_ori[1] == 20592
        np.testing.assert_approx_equal(-1.5742188, amp_ori[1], significant=6)

    @pytest.mark.isxd_gpio
    def test_GetGpioData_iterate_channels(self):
        input_file = test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video_gpio.isxd'
        exp_file = test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video_gpio.csv'

        gpio = isx.GpioSet.read(input_file)
        gpio_df = pd.read_csv(exp_file)
        cnames = [gpio.get_channel_name(k) for k in range(gpio.num_channels)]

        num_channels_with_zero_data = 0
        for index, cname in enumerate(cnames):
            # get expected data points for this channel, and sort
            i = gpio_df[' Channel Name'] == ' {}'.format(cname)

            # skip if there are zero points
            if i.sum() == 0:
                num_cells_with_zero_data += 1
                continue

            # compare expected event times and amplitudes with those returned from API
            data_times, data_amps = gpio.get_channel_data(index)

            np.testing.assert_allclose(data_times / 1e6, gpio_df['Time (s)'][i].values, rtol=1e-5)
            np.testing.assert_allclose(data_amps, gpio_df[' Value'][i].values, rtol=1e-5)

        # this is just a check to make sure nothing stupid happened
        assert num_channels_with_zero_data < len(cnames)

    @pytest.mark.isxd_gpio
    @pytest.mark.parametrize('method', ('get_channel_data', 'get_channel_name'))
    @pytest.mark.parametrize('channel_index', (26, 500000000, 18446744073709551615))
    def test_GetGpioData_bad_channel_index_int(self, method, channel_index):
        input_file = test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video_gpio.isxd'

        gpio = isx.GpioSet.read(input_file)
        with pytest.raises(Exception) as error:
            getattr(gpio, method)(channel_index)
        assert f'Channel index {channel_index} is too large' in str(error.value)

    @pytest.mark.isxd_gpio
    @pytest.mark.parametrize('method', ('get_channel_data', 'get_channel_name'))
    @pytest.mark.parametrize('channel_index', (8 + 3j, '0', '423', 'My Channel Name', 1.5, (1, 5), [3, 5], {4, 9}, {6: 5}))
    def test_GetGpioData_bad_channel_index_type(self, method, channel_index):
        input_file = test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video_gpio.isxd'

        event = isx.GpioSet.read(input_file)
        with pytest.raises(Exception) as error:
            getattr(event, method)(channel_index)
        assert ("argument 2:" in str(error.value) and "wrong type" in str(error.value)) or "cannot be interpreted as an integer" in str(error.value)

    def test_GpioSetStrValid(self):
        gpio_set = isx.GpioSet.read(test_data_path + '/unit_test/gpio/2020-05-20-10-33-22_video_gpio.isxd')
        assert isinstance(str(gpio_set), str)


    def test_GpioSetStrInvalid(self):
        gpio_set = isx.GpioSet()
        assert isinstance(str(gpio_set), str)

    @staticmethod
    def assert_acquisition_info_dataset_1(info):
        assert info['Animal Sex'] == 'm'
        assert info['Animal Date of Birth'] == ''
        assert info['Animal ID'] == ''
        assert info['Animal Species'] == ''
        assert info['Animal Weight'] == 0
        assert info['Animal Description'] == ''

        assert info['Microscope Focus'] == 0
        assert info['Microscope Gain'] == 7
        assert info['Microscope EX LED Power (mw/mm^2)'] == 0
        assert info['Microscope OG LED Power (mw/mm^2)'] == 0
        assert info['Microscope Serial Number'] == 'unknown'
        assert info['Microscope Type'] == 'nVista'

        assert info['Session Name'] == 'Session 20180621-174314'

        assert info['Experimenter Name'] == 'John Doe'

        assert info['Probe Diameter (mm)'] == 0
        assert info['Probe Flip'] == 'none'
        assert info['Probe Length (mm)'] == 0
        assert info['Probe Pitch'] == 0
        assert info['Probe Rotation (degrees)'] == 0
        assert info['Probe Type'] == 'None'

        assert info['Acquisition SW Version'] == '1.1.0'

    def test_MovieGetAcquisitionInfo(self):
        movie_file_path = test_data_path + '/unit_test/acquisition_info/2018-06-21-17-51-03_video_sched_0.isxd'
        movie = isx.Movie.read(movie_file_path)
        info = movie.get_acquisition_info()
        TestFileIO.assert_acquisition_info_dataset_1(info)

    def test_CellSetGetAcquisitionInfo(self):
        cell_set_file_path = test_data_path + '/unit_test/acquisition_info/2018-06-21-17-51-03_video_sched_0-PP-ROI.isxd'
        cell_set = isx.CellSet.read(cell_set_file_path)
        info = cell_set.get_acquisition_info()
        TestFileIO.assert_acquisition_info_dataset_1(info)

    def test_EventSetGetAcquisitionInfo(self):
        event_set_file_path = test_data_path + '/unit_test/acquisition_info/2018-06-21-17-51-03_video_sched_0-PP-ROI-ED.isxd'
        event_set = isx.EventSet.read(event_set_file_path)
        info = event_set.get_acquisition_info()
        TestFileIO.assert_acquisition_info_dataset_1(info)


    def testCreateCellMapNoAcceptedCells(self):
        create_cell_map_path = test_data_path + '/unit_test/create_cell_map'
        cell_set_file_path = create_cell_map_path + "/2020-05-07-11-38-24_video_DR_1_OQ_1-decompressed-efocus_0700-PP-ROI_001.isxd"
        output_dir = test_data_path + '/unit_test/output'

        actual_isxd_cell_map_name = 'created_cell_map_no_accepted.isxd'
        actual_isxd_cell_map = output_dir + '/' + actual_isxd_cell_map_name

        delete_files_silently(actual_isxd_cell_map)

        with pytest.raises(Exception) as error:
            isx.create_cell_map(cell_set_file_path,
                                output_isxd_cell_map_file=actual_isxd_cell_map)

        assert 'There are no cells to create a cell map with! Only selected cells will be used for the cell map.' in str(error.value)

        assert not is_file(actual_isxd_cell_map)

        delete_files_silently(actual_isxd_cell_map)

    def testCreateCellMapUndecided(self):
        create_cell_map_path = test_data_path + '/unit_test/create_cell_map'
        cell_set_file_path = create_cell_map_path + "/2020-05-07-11-38-24_video_DR_1_OQ_1-decompressed-efocus_0700-PP-ROI_001.isxd"
        output_dir = test_data_path + '/unit_test/output'

        actual_isxd_cell_map_name = 'created_cell_map_undecided.isxd'
        actual_isxd_cell_map = output_dir + '/' + actual_isxd_cell_map_name

        expected_isxd_cell_map = create_cell_map_path + '/' + actual_isxd_cell_map_name

        delete_files_silently(actual_isxd_cell_map)

        isx.create_cell_map(cell_set_file_path,
                            selected_cell_statuses=['accepted','undecided'],
                            output_isxd_cell_map_file=actual_isxd_cell_map)

        assert_isxd_images_are_close_by_path_nan_zero(expected_isxd_cell_map, actual_isxd_cell_map)

        delete_files_silently(actual_isxd_cell_map)


    @pytest.mark.parametrize(('binary','cell_thresh', 'rgb',  'cell_statuses'),
                            (( False,  .99,            None,  ['accepted', 'undecided']),
                             ( True,   .99,           "green",['accepted', 'undecided']),
                             ( True,   0.8,            None,  ['rejected']),
                             ( False,  0.8,            None,  ['rejected']),
                             ( True,   0.5,            None,  ['rejected']),
                             ( False,  0.4,           "blue", ['rejected']),
                             ( True,   0.4,           "blue", ['rejected']),
                             ( False,  0.0,           "red",  ['rejected']),
                             ))
    def testCreateCellMapVarInput(self, cell_thresh, binary, rgb, cell_statuses):
        create_cell_map_path = test_data_path + '/unit_test/create_cell_map'
        
        cell_set_file_path =  create_cell_map_path + "/synth_movie-03-no-frame-nums-lots-dots_he-PCA-ICA.isxd"
        
        output_dir = test_data_path + '/unit_test/output'

        actual_isxd_cell_map_name = f'created_cell_map_b{binary}_thresh{cell_thresh}_{rgb}.isxd'
        actual_tiff_cell_map_name = f'created_cell_map_b{binary}_thresh{cell_thresh}_{rgb}.tiff'
        actual_isxd_cell_map = output_dir + '/' + actual_isxd_cell_map_name
        actual_tiff_cell_map = output_dir + '/' + actual_tiff_cell_map_name

        expected_isxd_cell_map = create_cell_map_path + '/' + actual_isxd_cell_map_name
        expected_tiff_cell_map = create_cell_map_path + '/' + actual_tiff_cell_map_name

        delete_files_silently([actual_isxd_cell_map, actual_tiff_cell_map])

        isx.create_cell_map(cell_set_file_path,
                            selected_cell_statuses=cell_statuses,
                            output_isxd_cell_map_file=actual_isxd_cell_map,
                            output_tiff_cell_map_file=actual_tiff_cell_map,
                            cell_thresh=cell_thresh,
                            binary=binary,
                            rgb=rgb)

        assert_isxd_images_are_close_by_path_nan_zero(expected_isxd_cell_map, actual_isxd_cell_map)
        assert_tiff_files_equal_by_path(expected_tiff_cell_map, actual_tiff_cell_map)

        delete_files_silently([actual_isxd_cell_map, actual_tiff_cell_map])

    @pytest.mark.skip(reason="no suitably small data")
    def testCreateCellMapNVistaCellSet(self):
        create_cell_map_path = test_data_path + '/unit_test/create_cell_map'
        cell_set_file_path = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE.isxd'
        output_dir = test_data_path + '/unit_test/output'

        actual_isxd_cell_map = 'created_cell_map.isxd'
        actual_tiff_cell_map = 'created_cell_map.tiff'
        full_actual_isxd_cell_map = output_dir + '/created_cell_map.isxd'
        full_actual_tiff_cell_map = output_dir + '/created_cell_map.tiff'

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])

        expected_isxd_cell_map = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-MAP.isxd'
        expected_tiff_cell_map = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-MAP.tif'

        isx.create_cell_map(cell_set_file_path, cell_thresh=0.5, output_isxd_cell_map_name=actual_isxd_cell_map,
                            output_tiff_cell_map_name=actual_tiff_cell_map, output_dir=output_dir)

        assert_isxd_images_are_close_by_path_nan_zero(expected_isxd_cell_map, full_actual_isxd_cell_map)
        assert_tiff_files_equal_by_path(full_actual_tiff_cell_map, expected_tiff_cell_map)

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])


    @pytest.mark.skip(reason="no suitably small data")
    def testCreateCellMapNVistaCellSetOri(self):
        create_cell_map_path = test_data_path + '/unit_test/create_cell_map'
        cell_set_file_path = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-ORI.isxd'
        output_dir = test_data_path + '/unit_test/output'

        actual_isxd_cell_map = 'created_cell_map.isxd'
        actual_tiff_cell_map = 'created_cell_map.tiff'
        full_actual_isxd_cell_map = output_dir + '/created_cell_map.isxd'
        full_actual_tiff_cell_map = output_dir + '/created_cell_map.tiff'

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])

        expected_isxd_cell_map = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-ORI-MAP.isxd'
        expected_tiff_cell_map = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-ORI-MAP.tif'

        isx.create_cell_map(cell_set_file_path, cell_thresh=0.5,
                            output_isxd_cell_map_name=actual_isxd_cell_map,
                            output_tiff_cell_map_name=actual_tiff_cell_map,
                            output_dir=output_dir)

        assert_isxd_images_are_close_by_path_nan_zero(expected_isxd_cell_map, full_actual_isxd_cell_map)
        assert_tiff_files_equal_by_path(full_actual_tiff_cell_map, expected_tiff_cell_map)

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])


    @pytest.mark.skip(reason="no suitably small data")
    def testCreateCellMapNVistaCellSetOriTrf(self):
        create_cell_map_path = test_data_path + '/unit_test/create_cell_map'
        cell_set_file_path = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-ORI-TRF.isxd'
        output_dir = test_data_path + '/unit_test/output'

        actual_isxd_cell_map = 'created_cell_map.isxd'
        actual_tiff_cell_map = 'created_cell_map.tiff'
        full_actual_isxd_cell_map = output_dir + '/created_cell_map.isxd'
        full_actual_tiff_cell_map = output_dir + '/created_cell_map.tiff'

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])

        expected_isxd_cell_map = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-ORI-TRF-MAP.isxd'
        expected_tiff_cell_map = create_cell_map_path + '/2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-ORI-TRF-MAP.tif'

        isx.create_cell_map(cell_set_file_path, cell_thresh=0.5,
                            output_isxd_cell_map=actual_isxd_cell_map,
                            output_tiff_cell_map=actual_tiff_cell_map,
                            output_dir=output_dir)

        assert_isxd_images_are_close_by_path_nan_zero(expected_isxd_cell_map, full_actual_isxd_cell_map)
        assert_tiff_files_equal_by_path(full_actual_tiff_cell_map, expected_tiff_cell_map)

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])


    @pytest.mark.skip(reason="no suitably small data")
    def testCreateCellMapOlympusCellSet(self):
        create_cell_map_path = test_data_path + '/unit_test/create_cell_map'
        cell_set_file_path = create_cell_map_path + '/movie_920_green_resonant-BP-MC-CNMFE.isxd'
        output_dir = test_data_path + '/unit_test/output'

        actual_isxd_cell_map = 'created_cell_map.isxd'
        actual_tiff_cell_map = 'created_cell_map.tiff'
        full_actual_isxd_cell_map = output_dir + '/created_cell_map.isxd'
        full_actual_tiff_cell_map = output_dir + '/created_cell_map.tiff'

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])

        expected_isxd_cell_map = create_cell_map_path + '/movie_920_green_resonant-BP-MC-CNMFE-MAP.isxd'
        expected_tiff_cell_map = create_cell_map_path + '/movie_920_green_resonant-BP-MC-CNMFE-MAP.tif'

        isx.create_cell_map(cell_set_file_path, cell_thresh=0.5, output_isxd_cell_map_name=actual_isxd_cell_map,
                            output_tiff_cell_map_name=actual_tiff_cell_map, output_dir=output_dir)

        assert_isxd_images_are_close_by_path_nan_zero(expected_isxd_cell_map, full_actual_isxd_cell_map)
        assert_tiff_files_equal_by_path(full_actual_tiff_cell_map, expected_tiff_cell_map)

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])


    @pytest.mark.skip(reason="no suitably small data")
    def testCreateCellMapOlympusCellSetALG(self):
        create_cell_map_path = test_data_path + '/unit_test/create_cell_map'
        cell_set_file_path = create_cell_map_path + '/movie_920_green_resonant-BP-MC-CNMFE-ALG.isxd'
        output_dir = test_data_path + '/unit_test/output'

        actual_isxd_cell_map = 'created_cell_map.isxd'
        actual_tiff_cell_map = 'created_cell_map.tiff'
        full_actual_isxd_cell_map = output_dir + '/created_cell_map.isxd'
        full_actual_tiff_cell_map = output_dir + '/created_cell_map.tiff'

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])

        expected_isxd_cell_map = create_cell_map_path + '/movie_920_green_resonant-BP-MC-CNMFE-ALG-MAP.isxd'
        expected_tiff_cell_map = create_cell_map_path + '/movie_920_green_resonant-BP-MC-CNMFE-ALG-MAP.tif'

        isx.create_cell_map(cell_set_file_path, cell_thresh=0.5,
                            output_isxd_cell_map_name=actual_isxd_cell_map,
                            output_tiff_cell_map_name=actual_tiff_cell_map,
                            output_dir=output_dir)

        assert_isxd_images_are_close_by_path_nan_zero(expected_isxd_cell_map, full_actual_isxd_cell_map)
        assert_tiff_files_equal_by_path(full_actual_tiff_cell_map, expected_tiff_cell_map)

        delete_files_silently([full_actual_isxd_cell_map, full_actual_tiff_cell_map])

    def testOverlayCellMapOnImage(self):
        create_cell_map_path = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        cell_map_file = os.path.join(create_cell_map_path, 'movie_920_green_resonant-BP-MC-CNMFE-ALG-MAP.isxd')
        input_image_file = os.path.join(create_cell_map_path, 'zstack_920_green_Galvano-green_26to51.isxd')

        overlayed_image_name = 'overlayed_image.tiff'
        actual_overlayed_image_file = os.path.join(output_dir, overlayed_image_name)

        expected_overlayed_image_file = os.path.join(create_cell_map_path, overlayed_image_name)

        delete_files_silently(actual_overlayed_image_file)

        isx.overlay_cell_map_on_image(cell_map_file, input_image_file, actual_overlayed_image_file)

        assert is_file(actual_overlayed_image_file)
        assert_isxd_images_are_close_by_path_nan_zero(expected_overlayed_image_file.replace('\\', '/'), actual_overlayed_image_file.replace('\\', '/'))

        delete_files_silently(actual_overlayed_image_file)
   
   
    @pytest.mark.parametrize('input_cellset_map', (
    'created_cell_map_undecided.isxd',))
    def testOverlayCellMapOnImage_not_matching_files(self, input_cellset_map):
        create_cell_map_path = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        cell_map_file = os.path.join(create_cell_map_path, input_cellset_map)
        input_image_file = os.path.join(create_cell_map_path, 'image_output_accepted-cells-map.tiff')

        overlayed_image_name = f'o_{input_cellset_map}.tiff'
        actual_overlayed_image_file = os.path.join(output_dir, overlayed_image_name)
        with pytest.raises(Exception) as error:
            isx.overlay_cell_map_on_image(cell_map_file, input_image_file, overlayed_image_name)
        assert f'operands could not be broadcast together' in str(error.value) 

        assert not is_file(actual_overlayed_image_file)


    @pytest.mark.skip(reason="expected output file needs to be updated")
    def test_overlay_cellmaps(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_first_cellmap_file = os.path.join(base_dir, '2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-ORI-TRF-MAP.tif')
        input_second_cellmap_file = os.path.join(base_dir, 'movie_920_green_resonant-BP-MC-CNMFE-ALG-MAP.tif')

        output_tiff_name = 'overlayed_cellmaps.tiff'
        actual_output_tiff_file = os.path.join(output_dir, output_tiff_name)
        expected_output_tiff_file = os.path.join(base_dir, output_tiff_name)

        delete_files_silently(actual_output_tiff_file)

        isx.overlay_cellmaps(input_first_cellmap_file, input_second_cellmap_file, actual_output_tiff_file)

        assert is_file(actual_output_tiff_file)

        assert_tiff_files_equal_by_path(actual_output_tiff_file, expected_output_tiff_file)

        delete_files_silently(actual_output_tiff_file)


    def test_overlay_cellmaps_wrong_size(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_first_cellmap_file = os.path.join(base_dir, '2019-11-30-17-47-27_video_trig_0-PP-BP-MC-CNMFE-ORI-TRF-MAP.tif')
        input_second_cellmap_file = os.path.join(base_dir, 'image_rot_flip_metadata.tiff')

        output_tiff_name = 'overlayed_cellmaps_wrong_size.tiff'
        actual_output_tiff_file = os.path.join(output_dir, output_tiff_name)

        delete_files_silently(actual_output_tiff_file)

        with pytest.raises(Exception) as error:
            isx.overlay_cellmaps(input_first_cellmap_file, input_second_cellmap_file, actual_output_tiff_file)
        assert 'The two images do not have the same size: (1024, 1024) vs (912, 786)' in str(error.value)

        assert not is_file(actual_output_tiff_file)

        delete_files_silently(actual_output_tiff_file)

    def test_ReadVesselSet(self):
        input_file = test_data_path + '/unit_test/bloodflow/bloodflow_movie_10s-VD_vesselsets.isxd'
        vessel_set = isx.VesselSet.read(input_file)

        assert vessel_set.get_vessel_name(0) == 'V0'
        assert vessel_set.get_vessel_name(1) == 'V1'
        assert vessel_set.get_vessel_name(2) == 'V2'

        assert vessel_set.num_vessels == 3

        exp_period = isx.Duration._from_num_den(50, 1000)
        exp_start = isx.Time._from_secs_since_epoch(isx.Duration.from_secs(0))
        exp_spacing = isx.Spacing(num_pixels=(250,250))
        exp_spacing._impl.pixel_width = isx._internal.IsxRatio(6, 1)
        exp_spacing._impl.pixel_height = isx._internal.IsxRatio(6, 1)

        assert vessel_set.timing == isx.Timing(num_samples=200, period=exp_period, start=exp_start)

        assert vessel_set.spacing == exp_spacing

    def test_ReadVesselSetName(self):
        input_file = test_data_path + '/unit_test/bloodflow/bloodflow_movie_10s-VD_vesselsets.isxd'
        vessel_set = isx.VesselSet.read(input_file)

        name = vessel_set.get_vessel_name(1)

        assert name == 'V1'

    def test_ReadVesselSetStatus(self):
        input_file = test_data_path + '/unit_test/bloodflow/bloodflow_movie_10s-VD_vesselsets.isxd'
        vessel_set = isx.VesselSet.read(input_file)

        assert vessel_set.get_vessel_status(0) == 'accepted'
        assert vessel_set.get_vessel_status(1) == 'undecided'
        assert vessel_set.get_vessel_status(2) == 'rejected'

    def test_ReadVesselSetLine(self):
        input_file = test_data_path + '/unit_test/bloodflow/bloodflow_movie_10s-VD_vesselsets.isxd'
        vessel_set = isx.VesselSet.read(input_file)

        np.testing.assert_array_equal(vessel_set.get_vessel_line_data(0), np.array([[90, 71], [108,  88]]))
        np.testing.assert_array_equal(vessel_set.get_vessel_line_data(1), np.array([[148, 163], [167, 179]]))
        np.testing.assert_array_equal(vessel_set.get_vessel_line_data(2), np.array([[236, 146], [213, 163]]))

    def test_ReadVesselSetRoi(self):
        input_file = test_data_path + '/unit_test/bloodflow/rbcv_movie_1-RBCV_microns.isxd'
        vessel_set = isx.VesselSet.read(input_file)
        np.testing.assert_array_equal(vessel_set.get_vessel_line_data(0), np.array([[124, 25], [153, 36], [90, 202], [61, 191]]))
        np.testing.assert_array_equal(vessel_set.get_vessel_line_data(1), np.array([[24, 42], [43, 34], [85, 148], [65, 156]]))

    def test_ReadVesselSetType(self):
        input_rbcv_file = test_data_path + '/unit_test/bloodflow/rbcv_movie_1-RBCV_microns.isxd'
        input_vessel_diameter_file = test_data_path + '/unit_test/bloodflow/bloodflow_movie_10s-VD_vesselsets.isxd'
        vessel_set = isx.VesselSet.read(input_rbcv_file)
        assert vessel_set.get_vessel_set_type() == isx.VesselSet.VesselSetType.RBC_VELOCITY
        vessel_set = isx.VesselSet.read(input_vessel_diameter_file)
        assert vessel_set.get_vessel_set_type() == isx.VesselSet.VesselSetType.VESSEL_DIAMETER

    def test_ReadVesselSetDirection(self):
        input_file = test_data_path + '/unit_test/bloodflow/rbcv_movie_1-RBCV_microns.isxd'
        vessel_set = isx.VesselSet.read(input_file)
        trace = vessel_set.get_vessel_direction_trace_data(0)
        exp_trace = [248.08775, 247.69649, 247.4231 , 247.73364, 246.72937, np.nan]
        np.testing.assert_allclose(trace, exp_trace, rtol=1e-05)

    def test_ReadVesselSetCorrelations(self):
        input_file = test_data_path + '/unit_test/bloodflow/rbcv_movie_1-RBCV_microns.isxd'
        vessel_set = isx.VesselSet.read(input_file)
        vessel_id = 0
        frame_idx = 0
        correlations = vessel_set.get_vessel_correlations_data(vessel_id, frame_idx)

        exp_shape = (3, 178, 93)
        assert correlations.shape == exp_shape

    def test_ReadWriteVesselSet(self):
        vs_out_file = test_data_path + '/unit_test/output/test_readwrite_vesselset.isxd'
        delete_files_silently([vs_out_file])

        # create sample data that will be used to make the vessel set
        vessel_props = write_sample_vessel_diameter_set(vs_out_file)

        # read the created vessel set file and confirm the correct values have been written
        vs_in = isx.VesselSet.read(vs_out_file)
        assert vs_in.num_vessels == vessel_props['num_vessels']
        assert vs_in.spacing == vessel_props['spacing']
        assert vs_in.timing == vessel_props['timing']
        valid_samples_mask = vs_in.timing.get_valid_samples_mask()
        for k in range(vs_in.num_vessels):
            assert vs_in.get_vessel_name(k) == vessel_props['names'][k]
            assert vs_in.get_vessel_status(k) == 'undecided'

            # Note: Key difference, vessel sets only have one image, that being the first image
            np.testing.assert_array_equal(vs_in.get_vessel_image_data(k), vessel_props['images'][0])
            np.testing.assert_array_equal(vs_in.get_vessel_line_data(k), vessel_props['lines'][k])
            np.testing.assert_array_equal(vs_in.get_vessel_trace_data(k), vessel_props['traces'][k])
            np.testing.assert_array_equal(vs_in.get_vessel_center_trace_data(k), vessel_props['center_traces'][k])

        del vs_in
        delete_files_silently([vs_out_file])

    def test_ReadWriteVesselSetRbcVelocity(self):
        vs_out_file = test_data_path + '/unit_test/output/test_readwrite_vesselset.isxd'
        delete_files_silently([vs_out_file])

        # create sample data that will be used to make the vessel set
        vessel_props = write_sample_rbc_velocity_set(vs_out_file)

        # read the created vessel set file and confirm the correct values have been written
        vs_in = isx.VesselSet.read(vs_out_file)
        assert vs_in.num_vessels == vessel_props['num_vessels']
        assert vs_in.spacing == vessel_props['spacing']
        assert vs_in.timing == vessel_props['timing']
        valid_samples_mask = vs_in.timing.get_valid_samples_mask()
        for k in range(vs_in.num_vessels):
            # Note: Key difference, vessel sets only have one image, that being the first image
            assert vs_in.get_vessel_name(k) == vessel_props['names'][k]
            assert vs_in.get_vessel_status(k) == 'undecided'

            np.testing.assert_array_equal(vs_in.get_vessel_image_data(k), vessel_props['images'][0])
            np.testing.assert_array_equal(vs_in.get_vessel_line_data(k), vessel_props['lines'][k])
            np.testing.assert_array_equal(vs_in.get_vessel_trace_data(k), vessel_props['traces'][k])
            np.testing.assert_array_equal(vs_in.get_vessel_direction_trace_data(k), vessel_props['direction_traces'][k])

            for t in range(vessel_props['timing'].num_samples):
                np.testing.assert_array_equal(vs_in.get_vessel_correlations_data(k, t), vessel_props['correlations_traces'][k][t, :, :, :])

        del vs_in
        delete_files_silently([vs_out_file])

    def test_VesselSetGetAcquisitionInfo(self):
        vessel_set_file_path = test_data_path + '/unit_test/bloodflow/blood_flow_movie_1-VD_window2s_increment1s.isxd'
        vessel_set = isx.VesselSet.read(vessel_set_file_path)
        info = vessel_set.get_acquisition_info()

        # Assert against known values
        assert info['Trace Units'] == 'microns'
        assert info['Vessel Set Type'] == 'vessel diameter'
        assert info['Time Increment (s)'] == 1
        assert info['Time Window (s)'] == 2

    def test_AlignStartTimesInvalidRef(self):
        test_dir = test_data_path + "/unit_test/"
        ref_file_path = test_dir + "/imu/2020-02-13-18-43-21_video.imu"
        align_file_path = test_dir + "/nVision/20220412-200447-camera-100.isxb"

        with pytest.raises(Exception) as error:    
            isx.align_start_times(
                input_ref_file=ref_file_path,
                input_align_files=align_file_path)

        assert 'Unsupported data type - only gpio files, isxd movies, and isxb movies are supported as a timing reference.' in str(error.value)

    def test_AlignStartTimesInvalidAlign(self):
        test_dir = test_data_path + "/unit_test/"
        ref_file_path = test_dir + "/gpio/2020-05-20-10-33-22_video.gpio"
        align_file_path = test_dir + "/cell_metrics/cell_metrics_movie-PCA-ICA.isxd"

        with pytest.raises(Exception) as error:    
            isx.align_start_times(
                input_ref_file=ref_file_path,
                input_align_files=align_file_path)

        assert 'Unsupported data type - only isxd movies and isxb movies are supported as input files to align to a timing reference.' in str(error.value)

    @pytest.mark.skipif(not isx._is_with_algos, reason="Cannot run algo module with minimal api")
    def test_AlignStartTimesInvalidNoFrameTimestamps(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        ref_file_path = test_dir + "/2022-06-08-23-53-41_video.gpio"
        align_file_path = test_dir + "/2022-06-08-23-53-41_video.isxd"
        processed_file_path = test_dir + "/2022-06-08-23-53-41_video-PP.isxd"

        delete_files_silently([processed_file_path])

        # temporally downsample by 2x to strip movie of timestamps
        isx.preprocess(align_file_path, processed_file_path, temporal_downsample_factor=2)

        with pytest.raises(Exception) as error:    
            isx.align_start_times(
                input_ref_file=ref_file_path,
                input_align_files=processed_file_path)

        delete_files_silently([processed_file_path])

        assert 'Cannot get first tsc from movie with no frame timestamps.' in str(error.value)
    
    def test_AlignStartTimesInvalidNoUUID(self):
        test_dir = test_data_path + "/unit_test/"
        ref_file_path = test_dir + "/gpio/2020-05-20-10-33-22_video.gpio"
        align_file_path = test_dir + "/cnmfe-cpp/movie_128x128x1000.isxd"

        with pytest.raises(Exception) as error:    
            isx.align_start_times(
                input_ref_file=ref_file_path,
                input_align_files=align_file_path)

        assert 'Cannot determine if files are paired and synchronized - no recording UUID in timing reference file metadata.' in str(error.value)

    def test_AlignStartTimesInvalidPairedUnsynced(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-unsynchronized"
        ref_file_path = test_dir + "/2022-06-08-23-57-41_video.gpio"
        align_file_path = test_dir + "/2022-06-08-23-57-43-camera-1.isxb"

        with pytest.raises(Exception) as error:    
            isx.align_start_times(
                input_ref_file=ref_file_path,
                input_align_files=align_file_path)

        assert 'Files are not paired and synchronized - recording UUID of align file (AC-00111111-l4R4GRt28o-1654732663355) does not match recording UUID of timing reference file (AC-00111111-l4R4GRt28o-1654732661796).' in str(error.value)

    def test_AlignStartTimesInvalidStandalone(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID"
        ref_file_path = test_dir + "/standalone-miniscope/2022-06-08-23-58-43_video.gpio"
        align_file_path = test_dir + "/standalone-behavior/2022-06-08-23-58-51-camera-1.isxb"

        with pytest.raises(Exception) as error:    
            isx.align_start_times(
                input_ref_file=ref_file_path,
                input_align_files=align_file_path)

        assert 'Files are not paired and synchronized - recording UUID of align file (GA-21807233-0000000000-1654732731777) does not match recording UUID of timing reference file (AC-00111111-0000000000-1654732723918).' in str(error.value)

    def test_AlignStartTimesGpioRef(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        gpio_file_path = test_dir + "/2022-06-08-23-53-41_video.gpio"
        isxd_file_path = test_dir + "/2022-06-08-23-53-41_video.isxd"
        isxb_file_path = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"

        # Copy test isxb and isxd files to modify
        isxd_file_path_copy = test_dir + "/2022-06-08-23-53-41_video-mod.isxd"
        isxb_file_path_copy = test_dir + "/2022-06-08-23-53-41_video-camera-1-mod.isxb"

        delete_files_silently([isxd_file_path_copy, isxb_file_path_copy])

        shutil.copyfile(isxd_file_path, isxd_file_path_copy)
        shutil.copyfile(isxb_file_path, isxb_file_path_copy)

        isx.align_start_times(
            input_ref_file=gpio_file_path,
            input_align_files=[isxd_file_path_copy, isxb_file_path_copy])

        # Verify the modified isxd file
        original_isxd_movie = isx.Movie.read(isxd_file_path)
        modified_isxd_movie = isx.Movie.read(isxd_file_path_copy)

        # The calculated start time of an isxd file based on a gpio reference generally equals the start time
        # stored in the original isxd file because the isxd and gpio file originate from the same hardware system.
        # However this is not necessarily guaranteed to be the case which is why the isxd start time is recomputed just in case.
        assert modified_isxd_movie.timing == original_isxd_movie.timing

        # Ensure the frame data and json metadata in the file is not corrupted due to the operation.
        for i in range(original_isxd_movie.timing.num_samples):
            original_movie_frame = original_isxd_movie.get_frame_data(i)
            modified_movie_frame = modified_isxd_movie.get_frame_data(i)
            np.testing.assert_array_equal(modified_movie_frame, original_movie_frame)
        assert modified_isxd_movie.get_acquisition_info() == original_isxd_movie.get_acquisition_info()

        # Verify the modified isxb file
        original_isxb_movie = isx.Movie.read(isxb_file_path)
        modified_isxb_movie = isx.Movie.read(isxb_file_path_copy)

        # The recomputed start time is 541 ms greater than the start time in the original isxb file
        # Generally there is a greater delay in the start of isxb recording because it's on a separate hardware system from the gpio and isxd files
        original_isxb_ti = original_isxb_movie.timing
        exp_isxb_timing = isx.Timing(
            start=isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(original_isxb_ti.start._impl.secs_since_epoch.num + 541), original_isxb_ti.start._impl.utc_offset),
            period=original_isxb_ti.period,
            num_samples=original_isxb_ti.num_samples,
            dropped=original_isxb_ti.dropped,
            cropped=original_isxb_ti.cropped,
            blank=original_isxb_ti.blank
        )
        assert modified_isxb_movie.timing == exp_isxb_timing

        # Ensure the frame data and json metadata in the file is not corrupted due to the operation.
        for i in range(original_isxb_movie.timing.num_samples):
            original_movie_frame = original_isxb_movie.get_frame_data(i)
            modified_movie_frame = modified_isxb_movie.get_frame_data(i)
            np.testing.assert_array_equal(modified_movie_frame, original_movie_frame)
        assert modified_isxb_movie.get_acquisition_info() == original_isxb_movie.get_acquisition_info()

        del modified_isxd_movie
        del modified_isxb_movie
        delete_files_silently([isxd_file_path_copy, isxb_file_path_copy])

    def test_AlignStartTimesIsxdRef(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxd_file_path = test_dir + "/2022-06-08-23-53-41_video.isxd"
        isxb_file_path = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"

        # Copy test isxb and isxd files to modify
        isxb_file_path_copy = test_dir + "/2022-06-08-23-53-41_video-camera-1-mod.isxb"

        delete_files_silently([isxb_file_path_copy])

        shutil.copyfile(isxb_file_path, isxb_file_path_copy)

        isx.align_start_times(
            input_ref_file=isxd_file_path,
            input_align_files=[isxb_file_path_copy])

        # Verify the modified isxb file
        original_isxb_movie = isx.Movie.read(isxb_file_path)
        modified_isxb_movie = isx.Movie.read(isxb_file_path_copy)

        # The recomputed start time is 541 ms greater than the start time in the original isxb file
        original_isxb_ti = original_isxb_movie.timing
        exp_isxb_timing = isx.Timing(
            start=isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(original_isxb_ti.start._impl.secs_since_epoch.num + 541), original_isxb_ti.start._impl.utc_offset),
            period=original_isxb_ti.period,
            num_samples=original_isxb_ti.num_samples,
            dropped=original_isxb_ti.dropped,
            cropped=original_isxb_ti.cropped,
            blank=original_isxb_ti.blank
        )

        del modified_isxb_movie
        delete_files_silently([isxb_file_path_copy])

    def test_AlignStartTimesSeries(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/scheduled"
        ref_file_path = test_dir + "/2022-06-09-12-33-38_video_sched_0.gpio"
        align_file_paths = [
            test_dir + "/2022-06-09-12-33-38_video_sched_0-camera-1.isxb",
            test_dir + "/2022-06-09-12-33-38_video_sched_1-camera-1.isxb",
            test_dir + "/2022-06-09-12-33-38_video_sched_2-camera-1.isxb"
        ]

        # Copy test isxb and isxd files to modify
        align_copy_file_paths = [
            test_dir + "/2022-06-09-12-33-38_video_sched_0-camera-1-mod.isxb",
            test_dir + "/2022-06-09-12-33-38_video_sched_1-camera-1-mod.isxb",
            test_dir + "/2022-06-09-12-33-38_video_sched_2-camera-1-mod.isxb"
        ]

        delete_files_silently(align_copy_file_paths)
        
        for i in range(3):
            shutil.copyfile(align_file_paths[i], align_copy_file_paths[i])

        isx.align_start_times(
            input_ref_file=ref_file_path,
            input_align_files=align_copy_file_paths)

        exp_ts_diffs = [524, 10428, 20433]

        for i in range(3):
            # Verify the modified isxb file
            original_isxb_movie = isx.Movie.read(align_file_paths[i])
            modified_isxb_movie = isx.Movie.read(align_copy_file_paths[i])

            original_isxb_ti = original_isxb_movie.timing
            exp_isxb_timing = isx.Timing(
                start=isx.Time._from_secs_since_epoch(isx.Duration.from_msecs(original_isxb_ti.start._impl.secs_since_epoch.num + exp_ts_diffs[i]), original_isxb_ti.start._impl.utc_offset),
                period=original_isxb_ti.period,
                num_samples=original_isxb_ti.num_samples,
                dropped=original_isxb_ti.dropped,
                cropped=original_isxb_ti.cropped,
                blank=original_isxb_ti.blank
            )

            del modified_isxb_movie
        
        delete_files_silently(align_copy_file_paths)
    
    def test_ExportAlignedTimestamps(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        gpio_file_path = test_dir + "/2022-06-08-23-53-41_video.gpio"
        isxd_file_path = test_dir + "/2022-06-08-23-53-41_video.isxd"
        isxb_file_path = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        output_file_path = test_dir + "/output.csv"

        delete_files_silently([output_file_path])

        isx.export_aligned_timestamps(
            input_ref_file=gpio_file_path,
            input_align_files=[isxb_file_path, isxd_file_path],
            input_ref_name="gpio",
            input_align_names=["isxb", "isxd"],
            output_csv_file=output_file_path,
            time_ref='start')

        df = pd.read_csv(output_file_path)
        assert (df.iloc[0] == pd.DataFrame({
            'gpio Timestamp (s)' : [0.000000],
            'gpio Channel' : ['Digital GPI 0'],
            'isxb Timestamp (s)' : [0.282199],
            'isxd Timestamp (s)' : [0.052248]}
        )).all(axis=None)

        df.fillna(-1, inplace=True)
        assert (df.iloc[len(df.index) - 1] == pd.DataFrame({
            'gpio Timestamp (s)' : [1.951800],
            'gpio Channel' : ['BNC Trigger Input'],
            'isxb Timestamp (s)' : [-1],
            'isxd Timestamp (s)' : [-1]}
        )).all(axis=None)

        delete_files_silently([output_file_path])
    
    def test_ExportAlignedTimestamps_Unix(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxd_file_path = test_dir + "/2022-06-08-23-53-41_video.isxd"
        isxb_file_path = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        output_file_path = test_dir + "/output.csv"

        delete_files_silently([output_file_path])

        isx.export_aligned_timestamps(
            input_ref_file=isxd_file_path,
            input_align_files=[isxb_file_path],
            input_ref_name="isxd",
            input_align_names=["isxb"],
            output_csv_file=output_file_path,
            time_ref='unix')

        df = pd.read_csv(output_file_path, dtype=str)
        assert (df.iloc[0] == pd.DataFrame({
            'isxd Timestamp (s)' : ['1654732421.888000'],
            'isxb Timestamp (s)' : ['1654732422.117951']}
        )).all(axis=None)

        df.fillna('None', inplace=True)
        assert (df.iloc[len(df.index) - 1] == pd.DataFrame({
            'isxd Timestamp (s)' : ['None'],
            'isxb Timestamp (s)' : ['1654732423.853928']}
        )).all(axis=None)

        delete_files_silently([output_file_path])
    
    def test_ExportAlignedTimestamps_Tsc(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxd_file_path = test_dir + "/2022-06-08-23-53-41_video.isxd"
        isxb_file_path = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        output_file_path = test_dir + "/output.csv"

        delete_files_silently([output_file_path])

        isx.export_aligned_timestamps(
            input_ref_file=isxd_file_path,
            input_align_files=[isxb_file_path],
            input_ref_name="isxd",
            input_align_names=["isxb"],
            output_csv_file=output_file_path,
            time_ref='tsc')

        df = pd.read_csv(output_file_path)
        assert (df.iloc[0] == pd.DataFrame({
            'isxd Timestamp (s)' : [459472532939],
            'isxb Timestamp (s)' : [459472762890]}
        )).all(axis=None)

        df.fillna(-1, inplace=True)
        assert (df.iloc[len(df.index) - 1] == pd.DataFrame({
            'isxd Timestamp (s)' : [-1],
            'isxb Timestamp (s)' : [459474498867]}
        )).all(axis=None)

        delete_files_silently([output_file_path])
    
    def test_ExportEthovisionDataWithIsxbTimestamps(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file])

        isx.export_ethovision_data_with_isxb_timestamps(
            input_ethovision_file=ethovision_file,
            input_isxb_file=isxb_file,
            output_csv_file=output_csv_file,
            time_ref='tsc'
        )

        df = pd.read_csv(output_csv_file)
        df.fillna('None', inplace=True)

        assert (df.iloc[0] == 
            pd.DataFrame({
                'isxb Frame Timestamp (us)' : [459472762890],
                'Trial time' : [0.0],
                'Recording time' : [0.0],
                'X center' : [40.7033],
                'Y center' : [71.2558],
                'Area' : [3010.72],
                'Areachange' : [0.0],
                'Elongation' : [0.553722],
                'Distance moved' : ['None'],
                'Velocity' : ['None'],
                'Activity' : ['None'],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [0.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        assert (df.iloc[-1] == 
            pd.DataFrame({
                'isxb Frame Timestamp (us)' : [459474498867],
                'Trial time' : [1.733],
                'Recording time' : [1.733],
                'X center' : [42.9174],
                'Y center' : [71.0059],
                'Area' : [3369.08],
                'Areachange' : [551.344],
                'Elongation' : [0.487716],
                'Distance moved' : [2.41289],
                'Velocity' : [72.3875],
                'Activity' : [0.250916],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [1.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        delete_files_silently([output_csv_file])
    
    def test_ExportEthovisionDataWithIsxbTimestamps_FrameOffByOne(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        original_ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial-mod.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file, ethovision_file])

        ethovision_data = pd.read_excel(original_ethovision_file)
        ethovision_data.drop(ethovision_data.index[-1], inplace=True)
        ethovision_data.to_excel(ethovision_file, index=False)

        isx.export_ethovision_data_with_isxb_timestamps(
            input_ethovision_file=ethovision_file,
            input_isxb_file=isxb_file,
            output_csv_file=output_csv_file,
            time_ref='tsc'
        )

        df = pd.read_csv(output_csv_file)
        df.fillna('None', inplace=True)

        # validate first row of csv
        assert (df.iloc[0] == 
            pd.DataFrame({
                'isxb Frame Timestamp (us)' : [459472798865],
                'Trial time' : [0.0],
                'Recording time' : [0.0],
                'X center' : [40.7033],
                'Y center' : [71.2558],
                'Area' : [3010.72],
                'Areachange' : [0.0],
                'Elongation' : [0.553722],
                'Distance moved' : ['None'],
                'Velocity' : ['None'],
                'Activity' : ['None'],
                'Activity state(Highly active)' : [0],
                'Activity state(Inactive)' : [0],
                'Result 1' : [1]
            })
        ).all(axis=None)

        # validate last row of csv
        assert (df.iloc[-1] == 
            pd.DataFrame({
                'isxb Frame Timestamp (us)' : [459474498867],
                'Trial time' : [1.7],
                'Recording time' : [1.7],
                'X center' : [40.5835],
                'Y center' : [70.3934],
                'Area' : [3172.29],
                'Areachange' : [594.905],
                'Elongation' : [0.565859],
                'Distance moved' : [1.82628],
                'Velocity' : [54.7891],
                'Activity' : [0.258729],
                'Activity state(Highly active)' : [0],
                'Activity state(Inactive)' : [1],
                'Result 1' : [1]
            })
        ).all(axis=None)

        delete_files_silently([output_csv_file, ethovision_file])
    
    def test_ExportEthovisionDataWithIsxbTimestamps_FrameOffByMoreThanOne(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        original_ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial-mod.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file, ethovision_file])

        ethovision_data = pd.read_excel(original_ethovision_file)
        ethovision_data.drop(ethovision_data.index[-1], inplace=True)
        ethovision_data.drop(ethovision_data.index[-1], inplace=True)
        ethovision_data.to_excel(ethovision_file, index=False)

        with pytest.raises(ValueError) as error:
            isx.export_ethovision_data_with_isxb_timestamps(
                input_ethovision_file=ethovision_file,
                input_isxb_file=isxb_file,
                output_csv_file=output_csv_file,
                time_ref='tsc'
            )

        assert str(error.value) == "Length of timestamps array (53) is not the same as (or within one) of the ethovision table (51)"

        delete_files_silently([output_csv_file, ethovision_file])
    
    def test_ExportEthovisionDataWithIsxbTimestampsCsv(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.csv"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file])

        isx.export_ethovision_data_with_isxb_timestamps(
            input_ethovision_file=ethovision_file,
            input_isxb_file=isxb_file,
            output_csv_file=output_csv_file,
            time_ref='tsc'
        )

        df = pd.read_csv(output_csv_file)
        df.fillna('None', inplace=True)

        assert (df.iloc[0] == 
            pd.DataFrame({
                'isxb Frame Timestamp (us)' : [459472762890],
                'Trial time' : [0.0],
                'Recording time' : [0.0],
                'X center' : ['None'],
                'Y center' : ['None'],
                'Area' : ['None'],
                'Areachange' : ['None'],
                'Elongation' : ['None'],
                'Distance moved' : ['None'],
                'Velocity' : ['None'],
                'LED start' : [0.0],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [0.0],
                'Activity' : ['None'],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        assert (df.iloc[-1] == 
            pd.DataFrame({
                'isxb Frame Timestamp (us)' : [459474498867],
                'Trial time' : [2.08],
                'Recording time' : [2.08],
                'X center' : ['None'],
                'Y center' : ['None'],
                'Area' : ['None'],
                'Areachange' : ['None'],
                'Elongation' : ['None'],
                'Distance moved' : ['None'],
                'Velocity' : ['None'],
                'LED start' : [0.0],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [1.0],
                'Activity' : [0.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        delete_files_silently([output_csv_file])
    
    def test_ExportEthovisionDataWithIsxbTimestamps_TimeRefStart(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        isxd_file = test_dir + "/2022-06-08-23-53-41_video.isxd"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file])

        isx.export_ethovision_data_with_isxb_timestamps(
            input_ethovision_file=ethovision_file,
            input_isxb_file=isxb_file,
            output_csv_file=output_csv_file,
            input_ref_file=isxd_file,
            time_ref='start'
        )

        df = pd.read_csv(output_csv_file)
        df.fillna('None', inplace=True)

        assert (df.iloc[0] == 
            pd.DataFrame({
                'isxb Frame Timestamp (s)' : [0.229951],
                'Trial time' : [0.0],
                'Recording time' : [0.0],
                'X center' : [40.7033],
                'Y center' : [71.2558],
                'Area' : [3010.72],
                'Areachange' : [0.0],
                'Elongation' : [0.553722],
                'Distance moved' : ['None'],
                'Velocity' : ['None'],
                'Activity' : ['None'],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [0.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        assert (df.iloc[-1] == 
            pd.DataFrame({
                'isxb Frame Timestamp (s)' : [1.965928],
                'Trial time' : [1.733],
                'Recording time' : [1.733],
                'X center' : [42.9174],
                'Y center' : [71.0059],
                'Area' : [3369.08],
                'Areachange' : [551.344],
                'Elongation' : [0.487716],
                'Distance moved' : [2.41289],
                'Velocity' : [72.3875],
                'Activity' : [0.250916],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [1.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        delete_files_silently([output_csv_file])
    
    def test_ExportEthovisionDataWithIsxbTimestamps_TimeRefUnix(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        original_isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1-mod.isxb"
        isxd_file = test_dir + "/2022-06-08-23-53-41_video.isxd"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file, isxb_file])
        shutil.copyfile(original_isxb_file, isxb_file)

        isx.export_ethovision_data_with_isxb_timestamps(
            input_ethovision_file=ethovision_file,
            input_isxb_file=isxb_file,
            output_csv_file=output_csv_file,
            input_ref_file=isxd_file,
            time_ref='unix'
        )

        df = pd.read_csv(output_csv_file)
        df.fillna('None', inplace=True)
        assert (df.iloc[0] == 
            pd.DataFrame({
                'isxb Frame Timestamp (s)' : [1654732422.118],
                'Trial time' : [0.0],
                'Recording time' : [0.0],
                'X center' : [40.7033],
                'Y center' : [71.2558],
                'Area' : [3010.72],
                'Areachange' : [0.0],
                'Elongation' : [0.553722],
                'Distance moved' : ['None'],
                'Velocity' : ['None'],
                'Activity' : ['None'],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [0.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        assert (df.iloc[-1] == 
            pd.DataFrame({
                'isxb Frame Timestamp (s)' : [1654732423.853977],
                'Trial time' : [1.733],
                'Recording time' : [1.733],
                'X center' : [42.9174],
                'Y center' : [71.0059],
                'Area' : [3369.08],
                'Areachange' : [551.344],
                'Elongation' : [0.487716],
                'Distance moved' : [2.41289],
                'Velocity' : [72.3875],
                'Activity' : [0.250916],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [1.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        delete_files_silently([output_csv_file, isxb_file])
    
    def test_ExportEthovisionDataWithIsxbTimestamps_TimeRefStart_RefIsxb(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file])

        isx.export_ethovision_data_with_isxb_timestamps(
            input_ethovision_file=ethovision_file,
            input_isxb_file=isxb_file,
            output_csv_file=output_csv_file,
            input_ref_file=isxb_file,
            time_ref='start'
        )

        df = pd.read_csv(output_csv_file)
        df.fillna('None', inplace=True)

        assert (df.iloc[0] == 
            pd.DataFrame({
                'isxb Frame Timestamp (s)' : [0.0],
                'Trial time' : [0.0],
                'Recording time' : [0.0],
                'X center' : [40.7033],
                'Y center' : [71.2558],
                'Area' : [3010.72],
                'Areachange' : [0.0],
                'Elongation' : [0.553722],
                'Distance moved' : ['None'],
                'Velocity' : ['None'],
                'Activity' : ['None'],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [0.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        assert (df.iloc[-1] == 
            pd.DataFrame({
                'isxb Frame Timestamp (s)' : [1.735977],
                'Trial time' : [1.733],
                'Recording time' : [1.733],
                'X center' : [42.9174],
                'Y center' : [71.0059],
                'Area' : [3369.08],
                'Areachange' : [551.344],
                'Elongation' : [0.487716],
                'Distance moved' : [2.41289],
                'Velocity' : [72.3875],
                'Activity' : [0.250916],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [1.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        delete_files_silently([output_csv_file])

    def test_ExportEthovisionDataWithIsxbTimestamps_TimeRefUnix_RefIsxb(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        original_isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1-mod.isxb"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file, isxb_file])
        shutil.copyfile(original_isxb_file, isxb_file)

        isx.export_ethovision_data_with_isxb_timestamps(
            input_ethovision_file=ethovision_file,
            input_isxb_file=isxb_file,
            output_csv_file=output_csv_file,
            input_ref_file=isxb_file,
            time_ref='unix'
        )

        df = pd.read_csv(output_csv_file)
        df.fillna('None', inplace=True)
        assert (df.iloc[0] == 
            pd.DataFrame({
                'isxb Frame Timestamp (s)' : [1654732421.577],
                'Trial time' : [0.0],
                'Recording time' : [0.0],
                'X center' : [40.7033],
                'Y center' : [71.2558],
                'Area' : [3010.72],
                'Areachange' : [0.0],
                'Elongation' : [0.553722],
                'Distance moved' : ['None'],
                'Velocity' : ['None'],
                'Activity' : ['None'],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [0.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        assert (df.iloc[-1] == 
            pd.DataFrame({
                'isxb Frame Timestamp (s)' : [1654732423.312977],
                'Trial time' : [1.733],
                'Recording time' : [1.733],
                'X center' : [42.9174],
                'Y center' : [71.0059],
                'Area' : [3369.08],
                'Areachange' : [551.344],
                'Elongation' : [0.487716],
                'Distance moved' : [2.41289],
                'Velocity' : [72.3875],
                'Activity' : [0.250916],
                'Activity state(Highly active)' : [0.0],
                'Activity state(Inactive)' : [1.0],
                'Result 1' : [1.0]
            })
        ).all(axis=None)

        delete_files_silently([output_csv_file, isxb_file])
    
    def test_ExportEthovisionDataWithIsxbTimestamps_TimeRefStart_NoRef(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file])

        with pytest.raises(ValueError) as error:
            isx.export_ethovision_data_with_isxb_timestamps(
                input_ethovision_file=ethovision_file,
                input_isxb_file=isxb_file,
                output_csv_file=output_csv_file,
                time_ref='start'
            )
        assert str(error.value) == "An input reference file is required for time_ref = 'start' or time_ref = 'unix'."
    
    def test_ExportEthovisionDataWithIsxbTimestamps_TimeRefUnix_NoRef(self):
        test_dir = test_data_path + "/unit_test/nVision/recordingUUID/paired-synchronized/manual"
        isxb_file = test_dir + "/2022-06-08-23-53-41_video-camera-1.isxb"
        ethovision_file = test_data_path + "/unit_test/nVision/ethovision/ethovision_trial.xlsx"
        output_csv_file = test_dir + "/output.csv"

        delete_files_silently([output_csv_file])

        with pytest.raises(ValueError) as error:
            isx.export_ethovision_data_with_isxb_timestamps(
                input_ethovision_file=ethovision_file,
                input_isxb_file=isxb_file,
                output_csv_file=output_csv_file,
                time_ref='unix'
            )
        assert str(error.value) == "An input reference file is required for time_ref = 'start' or time_ref = 'unix'."
