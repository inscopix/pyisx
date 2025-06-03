from .utilities.setup import delete_files_silently, delete_dirs_silently, test_data_path, is_file

import os
import numpy as np
import pandas as pd
import pytest
from shutil import copyfile

import isx

from .asserts import assert_csv_cell_metrics_are_close_by_path, assert_isxd_cellsets_are_close_by_path, \
    assert_isxd_movies_are_close, assert_isxd_movies_are_close_by_path, assert_isxd_event_sets_are_close_by_path, \
    assert_csv_files_are_equal_by_path, assert_csv_files_are_close_by_path, \
    assert_csv_pairwise_spatial_overlap_matrices_are_close_by_path, \
    assert_isxd_vesselsets_are_close_by_path, assert_csv_traces_are_close_by_path, \
    assert_json_files_equal_by_path, assert_tiff_files_equal_by_path, \
    assert_isxd_cellsets_trace_sums, \
    assert_isxd_cellsets_cell_names, \
    assert_isxd_vesselsets_vessel_names

@pytest.mark.skipif(not isx._is_with_algos, reason="Only for algo tests")
class TestAlgorithms:
    @pytest.mark.isxd_movie
    def test_DeinterleaveMovie(self):
        test_dir = test_data_path + '/unit_test/VI'
        input_movie_bases = [
            'de_interleave_15_1_2_0_simple',
            'de_interleave_15_1_2_0_complex',
            'de_interleave_15_1_2_1_simple'
        ]
        efocus_values = [599, 800, 1000]

        input_movie_files = []
        output_movie_files = []
        expected = []
        for file_base in input_movie_bases:
            input_movie_files.append(test_dir + '/' + file_base + '.isxd')
            for efocus in efocus_values:
                output_movie_files.append(test_data_path + '/unit_test/output/' + file_base + '-efocus_' + str(efocus).zfill(4) + '.isxd')
                
                # IDPS-857 Upgrade version of test files since higher precision sampling rate results
                # in a slightly different computed start time for de-interleaved movies
                # IDPS-900 Upgrade version of test files since epoch start time is calculated based on tsc values
                # IDPS-1022 Upgrade version of test files after fixing bug with start time calculation
                expected.append(test_data_path + '/unit_test/guilded/' + file_base + '-efocus_' + str(efocus).zfill(4) + '_v3.isxd')

        delete_files_silently(output_movie_files)

        isx.de_interleave(input_movie_files, output_movie_files, efocus_values)

        for o, e in zip(output_movie_files, expected):
            assert_isxd_movies_are_close_by_path(e, o)

        delete_files_silently(output_movie_files)

    @pytest.mark.isxd_movie
    def test_PreprocessMovie(self):
        input_movie_file = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_movie_file = test_data_path + '/unit_test/output/test_output_pp.isxd'
        expected = test_data_path + '/unit_test/guilded/exp_mosaicPreProcessMovie_output.isxd'

        delete_files_silently([output_movie_file])

        isx.preprocess(input_movie_file,
                       output_movie_file,
                       temporal_downsample_factor=1,
                       spatial_downsample_factor=2,
                       crop_rect=[245, 245, 254, 254],
                       fix_defective_pixels=True)

        exp_movie = isx.Movie.read(expected)
        assert exp_movie.file_path == expected
        act_movie = isx.Movie.read(output_movie_file)
        assert act_movie.file_path == output_movie_file

        exp_movie.spacing._impl.pixel_width = isx._internal.IsxRatio(6, 1)
        exp_movie.spacing._impl.pixel_height = isx._internal.IsxRatio(6, 1)
        exp_movie.spacing._impl.left = isx._internal.IsxRatio(735, 1)
        exp_movie.spacing._impl.top = isx._internal.IsxRatio(735, 1)

        assert_isxd_movies_are_close(exp_movie, act_movie)

        del exp_movie
        del act_movie
        delete_files_silently([output_movie_file])

    
    @pytest.mark.isxd_movie
    def test_PreprocessMovieCropTLWH(self):
        # Test specifying crop rect as tlwh instead of tlbr and expect the same output
        input_movie_file = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_movie_file = test_data_path + '/unit_test/output/test_output_pp.isxd'
        expected = test_data_path + '/unit_test/guilded/exp_mosaicPreProcessMovie_output.isxd'

        delete_files_silently([output_movie_file])

        isx.preprocess(input_movie_file,
                       output_movie_file,
                       temporal_downsample_factor=1,
                       spatial_downsample_factor=2,
                       crop_rect=[245, 245, 10, 10],
                       crop_rect_format="tlwh",
                       fix_defective_pixels=True)

        exp_movie = isx.Movie.read(expected)
        assert exp_movie.file_path == expected
        act_movie = isx.Movie.read(output_movie_file)
        assert act_movie.file_path == output_movie_file

        exp_movie.spacing._impl.pixel_width = isx._internal.IsxRatio(6, 1)
        exp_movie.spacing._impl.pixel_height = isx._internal.IsxRatio(6, 1)
        exp_movie.spacing._impl.left = isx._internal.IsxRatio(735, 1)
        exp_movie.spacing._impl.top = isx._internal.IsxRatio(735, 1)

        assert_isxd_movies_are_close(exp_movie, act_movie)

        del exp_movie
        del act_movie
        delete_files_silently([output_movie_file])
    
    
    @pytest.mark.isxd_movie
    def test_PreprocessMovieCropTLWHRectangle(self):
        # Test specifying crop rect with unequal sides as tlwh instead of tlbr
        input_movie_file = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_movie_file = test_data_path + '/unit_test/output/test_output_pp.isxd'

        delete_files_silently([output_movie_file])

        top_left_x, top_left_y = (245, 245)
        width, height = (10, 20)
        isx.preprocess(input_movie_file,
                       output_movie_file,
                       temporal_downsample_factor=1,
                       spatial_downsample_factor=1,
                       crop_rect=[top_left_x, top_left_y, width, height],
                       crop_rect_format="tlwh",
                       fix_defective_pixels=True)

        act_movie = isx.Movie.read(output_movie_file)
        assert act_movie.file_path == output_movie_file
        assert act_movie.spacing.num_pixels == (height, width) # num_pixels returns (num_rows, num_cols) -> (height, width)

        del act_movie
        delete_files_silently([output_movie_file])
    
    
    @pytest.mark.isxd_movie
    def test_PreprocessMovieInvalidCropRectFormat(self):
        # Test specifying invalid crop rect raises error
        input_movie_file = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_movie_file = test_data_path + '/unit_test/output/test_output_pp.isxd'

        with pytest.raises(Exception) as error:
            isx.preprocess(input_movie_file,
                output_movie_file,
                crop_rect_format="something bad"
            )
        assert "Invalid crop rect format (something bad), must be one of the following: ('tlbr', 'tlwh')" in str(error.value)


    @pytest.mark.isxd_movie
    def test_PreprocessMovieNoCropRect(self):
        input_movie_file = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_movie_file = test_data_path + '/unit_test/output/test_output_ppnorect.isxd'

        delete_files_silently([output_movie_file])

        isx.preprocess(input_movie_file,
                       output_movie_file,
                       temporal_downsample_factor=1,
                       spatial_downsample_factor=2,
                       fix_defective_pixels=True)

        mov1 = isx.Movie.read(input_movie_file)
        mov2 = isx.Movie.read(output_movie_file)

        assert np.sum(np.abs(np.array(mov1.spacing.num_pixels) - np.array(mov2.spacing.num_pixels)*2)) == 0

        del mov1
        del mov2
        delete_files_silently([output_movie_file])

    @pytest.mark.tiff_movie
    def test_PreprocessTiffMovie(self):
        # I added this test to verify that TIFF inputs can be used in processing steps,
        # but get the default timing.
        input_movie_file = test_data_path + '/unit_test/recording_20161104_145443.tif'
        output_movie_file = test_data_path + '/unit_test/output/test_output_pp_tiff.isxd'
        delete_files_silently([output_movie_file])

        isx.preprocess(input_movie_file, output_movie_file)

        movie = isx.Movie.read(output_movie_file)
        assert movie.timing.start == isx.Time()
        assert movie.timing.period == isx.Duration.from_msecs(50)

        del movie
        delete_files_silently([output_movie_file])

    def test_PreprocessTrimmedMovie(self):
        # (IDPS-1120): A bug was discovered with the preprocess function where it assumes
        # that the start time of isxd files is stored as a ratio of ms / 1000
        # but the trim movie operation changes the start time to be a ratio fo us / 1000000.
        # This assumption led to output preprocessed files with an incorrect start time far into the future.
        # This test validates that trimming and then preprocessing a movie produces
        # an output with a valid start time.

        input_movie_file = test_data_path + '/unit_test/early_frames/2019-03-18-15-56-10_video_trig_0.isxd'
        trimmed_movie_file = test_data_path + '/unit_test/output/2019-03-18-15-56-10_video_trig_0-TPC.isxd'
        output_movie_file = test_data_path + '/unit_test/output/2019-03-18-15-56-10_video_trig_0-TPC-PP.isxd'
        delete_files_silently([trimmed_movie_file, output_movie_file])

        trim_range = [(0, 84)]
        isx.trim_movie(input_movie_file, trimmed_movie_file, trim_range)

        isx.preprocess(trimmed_movie_file, output_movie_file)

        movie = isx.Movie.read(output_movie_file)

        import datetime
        assert movie.timing.start.to_datetime() == datetime.datetime(
            year=2019,
            month=3,
            day=18,
            hour=15,
            minute=56,
            second=19,
            microsecond=82155
        )

        del movie
        delete_files_silently([trimmed_movie_file, output_movie_file])

    @pytest.mark.isxd_movie
    def test_MotionCorrectMovie(self):
        input_movie  = test_data_path + '/unit_test/motionCorrection/Inscopix-ratHipp2-recording_20160707_104710-pp-trim.hdf5'
        output_movie = test_data_path + '/unit_test/output/test_output_mc.isxd'
        expected     = test_data_path + '/unit_test/guilded/exp_mosaicMotionCorrectMovie_output-v3.isxd'

        delete_files_silently([output_movie])

        lowCutoff = 0.025
        highCutoff = 0.2
        roi = [[350, 225], [370, 390], [660, 390], [660, 225]]

        isx.motion_correct(input_movie, output_movie, max_translation=50,
                             low_bandpass_cutoff=lowCutoff, high_bandpass_cutoff=highCutoff, roi=roi,
                             reference_segment_index=0, reference_frame_index=0, reference_file_name='')

        exp_movie = isx.Movie.read(expected)
        assert exp_movie.file_path == expected
        act_movie = isx.Movie.read(output_movie)
        assert act_movie.file_path == output_movie

        exp_movie.spacing._impl.pixel_width = isx._internal.IsxRatio(3, 1)
        exp_movie.spacing._impl.pixel_height = isx._internal.IsxRatio(3, 1)
        exp_movie.spacing._impl.left = isx._internal.IsxRatio(6, 1)
        exp_movie.spacing._impl.top = isx._internal.IsxRatio(39, 1)

        assert_isxd_movies_are_close(exp_movie, act_movie)

        del exp_movie
        del act_movie
        delete_files_silently([output_movie])

    @pytest.mark.isxd_movie
    def test_MotionCorrectMovieNoRoi(self):
        input_movie  = test_data_path + '/unit_test/motionCorrection/Inscopix-ratHipp2-recording_20160707_104710-pp-trim.hdf5'
        output_movie = test_data_path + '/unit_test/output/test_output_mc_nr.isxd'
        expected     = test_data_path + '/unit_test/guilded/exp_mosaicMotionCorrectMovieNoRoi_output.isxd'

        delete_files_silently([output_movie])

        isx.motion_correct(input_movie, output_movie, max_translation=50, low_bandpass_cutoff=0.025, high_bandpass_cutoff=0.2)

        exp_movie = isx.Movie.read(expected)
        assert exp_movie.file_path == expected
        act_movie = isx.Movie.read(output_movie)
        assert act_movie.file_path == output_movie

        exp_movie.spacing._impl.pixel_width = isx._internal.IsxRatio(3, 1)
        exp_movie.spacing._impl.pixel_height = isx._internal.IsxRatio(3, 1)
        exp_movie.spacing._impl.left = isx._internal.IsxRatio(3, 1)
        exp_movie.spacing._impl.top = isx._internal.IsxRatio(18, 1)

        assert_isxd_movies_are_close(exp_movie, act_movie)

        del exp_movie
        del act_movie
        delete_files_silently([output_movie])

    @pytest.mark.isxd_movie
    def test_MotionCorrectMovieHourglassRoi(self):
        input_movie  = test_data_path + '/unit_test/recording_20161104_145543-PP.isxd'
        output_movie = test_data_path + '/unit_test/output/test_output_mc_hr.isxd'
        expected = test_data_path + '/unit_test/guilded/exp_mosaicMotionCorrectMovieHourglassRoi_output-v2.isxd'
        delete_files_silently([output_movie])

        isx.motion_correct(input_movie, output_movie, roi=[[42, 54], [203, 294], [53, 301], [161, 38]])

        assert_isxd_movies_are_close_by_path(expected, output_movie)

        delete_files_silently([output_movie])

    @pytest.mark.isxd_movie
    def test_MotionCorrectMovieOutputTranslations(self):
        input_movie  = test_data_path + '/unit_test/motionCorrection/Inscopix-ratHipp2-recording_20160707_104710-pp-trim.hdf5'
        output_movie = test_data_path + '/unit_test/output/test_output_mc_ot.isxd'
        expected     = test_data_path + '/unit_test/guilded/exp_mosaicMotionCorrectMovie_output-v3.isxd'
        output_trans = test_data_path + '/unit_test/output/test_output_mc_translations.csv'

        delete_files_silently([output_movie, output_trans])

        lowCutoff = 0.025
        highCutoff = 0.2
        roi = [[350, 225], [370, 390], [660, 390], [660, 225]]

        isx.motion_correct(input_movie, output_movie, max_translation=50,
                             low_bandpass_cutoff=lowCutoff, high_bandpass_cutoff=highCutoff, roi=roi,
                             reference_segment_index=0, reference_frame_index=0, reference_file_name='',
                             output_translation_files=output_trans)

        assert os.path.exists(output_trans), "Missing expected output translations"

        df = pd.read_csv(output_trans)
        col_names = df.keys()
        assert col_names[0] == 'translationX'
        assert col_names[1] == 'translationY'

        out_mov = isx.Movie.read(output_movie)
        num_frames = out_mov.timing.num_samples
        del out_mov

        assert len(df) == num_frames, "# of output translations does not match number of frames: {} != {}".format(len(df), num_frames)

        delete_files_silently([output_movie])

    @pytest.mark.isxd_movie
    def test_MotionCorrectMovieOutputCropRect(self):
        input_movie  = test_data_path + '/unit_test/motionCorrection/Inscopix-ratHipp2-recording_20160707_104710-pp-trim.hdf5'
        output_movie = test_data_path + '/unit_test/output/test_output_mc_cr.isxd'
        expected     = test_data_path + '/unit_test/guilded/exp_mosaicMotionCorrectMovie_output-v3.isxd'
        output_rect  = test_data_path + '/unit_test/output/test_output_mc_crop_rect.csv'

        delete_files_silently([output_movie, output_rect])

        lowCutoff = 0.025
        highCutoff = 0.2
        roi = [[350, 225], [370, 390], [660, 390], [660, 225]]

        isx.motion_correct(input_movie, output_movie, max_translation=50,
                             low_bandpass_cutoff=lowCutoff, high_bandpass_cutoff=highCutoff, roi=roi,
                             reference_segment_index=0, reference_frame_index=0, reference_file_name='',
                             output_crop_rect_file=output_rect)

        assert os.path.exists(output_rect), "Missing expected output crop rectangle"

        with open(output_rect, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 1
        crop_rect = [int(x.strip()) for x in lines[0].split(',')]

        assert len(crop_rect) == 4
        x,y,width,height = crop_rect

        # verify that the output movie size matches the crop rectangle width and height
        out_mov = isx.Movie.read(output_movie)
        num_rows,num_cols = out_mov.spacing.num_pixels
        out_first_frame = out_mov.get_frame_data(0)
        del out_mov

        assert width == num_cols, "# of columns in output movie does not match width of crop rect: {} != {}".format(num_cols, width)
        assert height == num_rows, "# of rows in output movie does not match height of crop rect: {} != {}".format(num_rows, height)

        # verify that the first frame of the input and output movies match after cropping
        in_mov = isx.Movie.read(input_movie)
        num_rows,num_cols = in_mov.spacing.num_pixels
        in_first_frame = in_mov.get_frame_data(0)
        del in_mov

        in_first_frame_cropped = in_first_frame[y:(y+height), x:(x+width)]
        assert in_first_frame_cropped.shape == out_first_frame.shape

        assert np.max(np.abs(in_first_frame_cropped - out_first_frame)) < 1e-6

        delete_files_silently([output_movie])

    @pytest.mark.isxd_movie
    def test_MotionCorrectMovieWithNoPadding(self):
        input_movie_file  = test_data_path + '/unit_test/motionCorrection/Inscopix-ratHipp2-recording_20160707_104710-pp-trim.hdf5'
        output_movie_file = test_data_path + '/unit_test/output/test_output_mc_nr.isxd'

        delete_files_silently([output_movie_file])

        isx.motion_correct(input_movie_file, output_movie_file, max_translation=50, low_bandpass_cutoff=0.025, high_bandpass_cutoff=0.2, preserve_input_dimensions=False)

        input_movie = isx.Movie.read(input_movie_file)
        output_movie = isx.Movie.read(output_movie_file)

        assert output_movie.spacing.num_pixels != input_movie.spacing.num_pixels

        del input_movie
        del output_movie
        delete_files_silently([output_movie_file])


    @pytest.mark.isxd_movie
    def test_MotionCorrectMovieWithPadding(self):
        input_movie_file  = test_data_path + '/unit_test/motionCorrection/Inscopix-ratHipp2-recording_20160707_104710-pp-trim.hdf5'
        output_movie_file = test_data_path + '/unit_test/output/test_output_mc_nr.isxd'

        delete_files_silently([output_movie_file])

        isx.motion_correct(input_movie_file, output_movie_file, max_translation=50, low_bandpass_cutoff=0.025, high_bandpass_cutoff=0.2, preserve_input_dimensions=True)

        input_movie = isx.Movie.read(input_movie_file)
        output_movie = isx.Movie.read(output_movie_file)
        
        assert output_movie.spacing.num_pixels == input_movie.spacing.num_pixels

        del input_movie
        del output_movie
        delete_files_silently([output_movie_file])

    @pytest.mark.isxd_trace
    def test_PcaIcaMovie(self):
        movie_file = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        output_file     = test_data_path + '/unit_test/output/test_output_pcaica.isxd'
        expected        = test_data_path + '/unit_test/guilded/exp_mosaicPcaIcaMovie_output.isxd'

        delete_files_silently([output_file])

        converged = isx.pca_ica(movie_file, output_file, 150, 120, unmix_type='temporal', ica_temporal_weight=0,
                                 max_iterations=500, convergence_threshold=1e-5, block_size=1000,
                                auto_estimate_num_ics=False, average_cell_diameter=10)

        assert converged

        assert_isxd_cellsets_are_close_by_path(expected, output_file, relative_tolerance=1e-3)

        delete_files_silently([output_file])

    # TODO: The test below is when using SCS as opposed to OASIS. SCS is not currently exposed via the Python API.
    # @pytest.mark.isxd_movie
    # def test_CnmfeMovie(self):
    #     input_movie_files = [test_data_path + '/unit_test/cnmfe-cpp/CnmfeAlgoMovie30x30x100.isxd']
    #     output_cellset_files = [test_data_path + '/unit_test/output/test_output_cellset_cnmfe.isxd']
    #     output_memory_map_files = [test_data_path + '/unit_test/output/test_movie_mmap.bin']
    #
    #     output_events_files = [test_data_path + '/unit_test/output/test_output_events_cnmfe.isxd']
    #     expected_cellsets = [test_data_path + '/unit_test/guilded/exp_mosaicCnmfeMovie_output_CS.isxd']
    #
    #     delete_files_silently(output_cellset_files)
    #     delete_files_silently(output_memory_map_files)
    #     # delete_files_silently(output_events_files)
    #
    #     isx.run_cnmfe(input_movie_files, output_cellset_files, output_memory_map_files,
    #                   deconvolution_method='scs', processing_mode='all_in_memory')
    #
    #     # TODO: Implement event detection as a seperate module
    #     # assert_isxd_cellsets_are_close_by_path(expected_cellsets[0], output_cellset_files[0], relative_tolerance=1e-7, use_cosine=True)
    #     # assert_isxd_event_sets_are_close_by_path(expected_events[0], output_events_files[0], relative_tolerance=1e-7, use_cosine=True)
    #
    #     delete_files_silently(output_cellset_files)
    #     delete_files_silently(output_memory_map_files)
    #     # delete_files_silently(output_events_files)

    @pytest.mark.isxd_movie
    def test_CnmfeMovie_AllInMemory(self):
        input_movie_files = [test_data_path + '/unit_test/cnmfe-cpp/CnmfeAlgoMovie128x128x100.isxd']
        output_cellset_files = [test_data_path + '/unit_test/output/test_output_cellset_cnmfe.isxd']
        output_dir = test_data_path + '/unit_test/tmp-cnmfe'

        expected_cellsets = [test_data_path + '/unit_test/guilded/exp_mosaicCnmfeMovie128x128x100AllMem_output_cellset.isxd']

        delete_files_silently(output_cellset_files)
        delete_dirs_silently(output_dir)
        os.makedirs(output_dir)

        isx.run_cnmfe(
            input_movie_files, output_cellset_files, output_dir,
            cell_diameter=7, # was set to 13 before we decided to internally double the user-specified cell diameter; old results stored in exp_mosaicCnmfeMovie128x128x100AllMem_output_cellset_celldiameter13.isxd
            min_corr=0.8, min_pnr=10, bg_spatial_subsampling=2, ring_size_factor=1.4,
            gaussian_kernel_size=3, closing_kernel_size=3, merge_threshold=0.7,
            processing_mode='all_in_memory', num_threads=4, patch_size=80, patch_overlap=20,
            output_unit_type='df_over_noise')

        assert_isxd_cellsets_are_close_by_path(expected_cellsets[0], output_cellset_files[0], relative_tolerance=1e-5, use_cosine=True)

        delete_files_silently(output_cellset_files)
        delete_dirs_silently(output_dir)

    @pytest.mark.isxd_movie
    def test_CnmfeMovie_PatchMode(self):
        input_movie_files = [test_data_path + '/unit_test/cnmfe-cpp/CnmfeAlgoMovie128x128x100.isxd']
        output_cellset_files = [test_data_path + '/unit_test/output/test_output_cellset_cnmfe.isxd']
        output_dir = test_data_path + '/unit_test/tmp-cnmfe'

        # - Found a bug in patch mode where patch coordinates were reversed for rows and columns
        # - Fixing that issue results in a different order for patches which is why this is a "revised" version of the original test data
        #   /unit_test/guilded/exp_mosaicCnmfeMovie128x128x100Patch_output_cellset.isxd
        # - The revised data has been updated to take into account the cell diameter change described below.

        expected_cellsets = [test_data_path + '/unit_test/guilded/exp_mosaicCnmfeMovie128x128x100Patch_output_cellset.isxd']

        delete_files_silently(output_cellset_files)
        delete_dirs_silently(output_dir)
        os.makedirs(output_dir)

        isx.run_cnmfe(
            input_movie_files, output_cellset_files, output_dir,
            cell_diameter=7,  # was set to 13 before we decided to internally double the user-specified cell diameter; old results stored in exp_mosaicCnmfeMovie128x128x100AllMem_output_cellset_revised_celldiameter13.isxd
            min_corr=0.8, min_pnr=10, bg_spatial_subsampling=2, ring_size_factor=1.4,
            gaussian_kernel_size=0, closing_kernel_size=0, merge_threshold=0.7,
            processing_mode="parallel_patches", num_threads=4, patch_size=80, patch_overlap=20,
            output_unit_type='df')

        assert_isxd_cellsets_are_close_by_path(expected_cellsets[0], output_cellset_files[0], relative_tolerance=1e-5, use_cosine=True)

        delete_files_silently(output_cellset_files)
        delete_dirs_silently(output_dir)

    @pytest.mark.isxd_movie
    def test_EstimateIcsImage(self):
        input_file = test_data_path + '/unit_test/cell_count_est/2019-07-11-10-22-45_video-efocus_0189-PP-BP-MC-TPC-DFF_maxproj.isxd'
        expected = 304

        ic_count = isx.estimate_num_ics(input_file, average_diameter = 12)
        assert ic_count == expected
        ic_count = isx.estimate_num_ics(input_file, min_diameter = 4, max_diameter = 20)
        assert ic_count == expected

    @pytest.mark.isxd_movie
    def test_SpatialBandPassMovie(self):
        input_file  = test_data_path + '/unit_test/single_10x10_frameMovie.isxd'
        output_file = test_data_path + '/unit_test/output/test_output_sbp.isxd'
        expected    = test_data_path + '/unit_test/guilded/exp_mosaicSpatialBandPass_output-v3.isxd'

        delete_files_silently([output_file])

        isx.spatial_filter(input_file, output_file, low_cutoff=0.1, high_cutoff=0.9,
                               retain_mean=False, subtract_global_minimum=False)

        assert_isxd_movies_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])

    @pytest.mark.isxd_movie
    def test_SpatialBandPassImage(self):
        input_file  = test_data_path + '/unit_test/create_cell_map/cell_map_image.isxd'
        output_file = test_data_path + '/unit_test/output/spatial_filtered.isxd'
        expected    = test_data_path + '/unit_test/create_cell_map/spatial_filtered.isxd'

        delete_files_silently([output_file])

        isx.spatial_filter(input_file, output_file, low_cutoff=0.1, high_cutoff=0.9,
                               retain_mean=False, subtract_global_minimum=False)

        assert_isxd_movies_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])


    @pytest.mark.isxd_movie
    def test_SpatialHighPassMovie(self):
        input_file  = test_data_path + '/unit_test/single_10x10_frameMovie.isxd'
        output_file = test_data_path + '/unit_test/output/test_output_sbp.isxd'

        delete_files_silently([output_file])

        isx.spatial_filter(input_file, output_file, low_cutoff=None, high_cutoff=0.9,
                               retain_mean=False, subtract_global_minimum=False)

        assert os.path.exists(output_file)

        delete_files_silently([output_file])
    

    @pytest.mark.isxd_movie
    def test_SpatialLowPassMovie(self):
        input_file  = test_data_path + '/unit_test/single_10x10_frameMovie.isxd'
        output_file = test_data_path + '/unit_test/output/test_output_sbp.isxd'

        delete_files_silently([output_file])

        isx.spatial_filter(input_file, output_file, low_cutoff=0.1, high_cutoff=None,
                               retain_mean=False, subtract_global_minimum=False)

        assert os.path.exists(output_file)
        
        delete_files_silently([output_file])


    @pytest.mark.isxd_movie
    def test_DeltaFoverF(self):
        input_file  = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_file = test_data_path + '/unit_test/output/test_output_dff.isxd'
        expected    = test_data_path + '/unit_test/guilded/exp_mosaicDeltaFoverF_output-v2.isxd'

        delete_files_silently([output_file])

        isx.dff(input_file, output_file, f0_type='mean')

        exp_movie = isx.Movie.read(expected)
        assert exp_movie.file_path == expected
        act_movie = isx.Movie.read(output_file)
        assert act_movie.file_path == output_file

        exp_movie.spacing._impl.pixel_width = isx._internal.IsxRatio(3, 1)
        exp_movie.spacing._impl.pixel_height = isx._internal.IsxRatio(3, 1)

        assert_isxd_movies_are_close(exp_movie, act_movie)

        del exp_movie
        del act_movie
        delete_files_silently([output_file])

    @pytest.mark.isxd_movie
    def test_ProjectMovieMean(self):
        input_file  = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        output_file = test_data_path + '/unit_test/output/test_output_project_movie_mean.isxd'
        expected    = test_data_path + '/unit_test/50fr10_l1-3cells_he-Mean Image-v2.isxd'

        delete_files_silently([output_file])

        isx.project_movie(input_file, output_file, 'mean')

        assert_isxd_movies_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])

    @pytest.mark.isxd_movie
    def test_ProjectMovieMax(self):
        input_file  = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        output_file = test_data_path + '/unit_test/output/test_output_project_movie_max.isxd'
        expected    = test_data_path + '/unit_test/50fr10_l1-3cells_he-Maximum Image-v2.isxd'

        delete_files_silently([output_file])

        isx.project_movie(input_file, output_file, 'max')

        assert_isxd_movies_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])

    @pytest.mark.isxd_movie
    def test_ProjectMovieMin(self):
        input_file  = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        output_file = test_data_path + '/unit_test/output/test_output_project_movie_min.isxd'
        expected    = test_data_path + '/unit_test/50fr10_l1-3cells_he-Minimum Image-v2.isxd'

        delete_files_silently([output_file])

        isx.project_movie(input_file, output_file, 'min')

        assert_isxd_movies_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])

    @pytest.mark.isxd_movie
    def test_ProjectMovieStandardDeviation(self):
        input_file  = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        output_file = test_data_path + '/unit_test/output/test_output_project_movie_min.isxd'
        expected    = test_data_path + '/unit_test/50fr10_l1-3cells_he-Standard Deviation Image-v2.isxd'

        delete_files_silently([output_file])

        isx.project_movie(input_file, output_file, 'standard_deviation')

        assert_isxd_movies_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])    

    @pytest.mark.isxd_events
    def test_EventDetection(self):
        input_file = test_data_path + '/unit_test/eventDetectionCellSet.isxd'
        output_file = test_data_path + '/unit_test/output/event_output.isxd'
        expected = test_data_path + '/unit_test/guilded/exp_mosaicEventDetection_output-v2.isxd'

        delete_files_silently([output_file])

        isx.event_detection(input_file, output_file, threshold=0.25,
                             tau=0.500, event_time_ref='beginning',
                             ignore_negative_transients=True, accepted_cells_only=False)

        assert_isxd_event_sets_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])

    def test_EventDetection_negative_not_existing_input_file(self):
        input_file = test_data_path + '/unit_test/not_existing_input_file.isxd'
        output_file = test_data_path + '/unit_test/output/event_output_olb.isxd'

        delete_files_silently([output_file])

        with pytest.raises(Exception) as error:
            isx.event_detection(input_file, output_file, threshold=0.25,
                                 tau=0.500, event_time_ref='beginning',
                                 ignore_negative_transients=True, accepted_cells_only=False)
        assert 'File does not exist' in str(error.value)

        assert not is_file(output_file)
        delete_files_silently([output_file])

    def test_EventDetection_negative_input_file_not_cellset(self):
        input_file = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        output_file = test_data_path + '/unit_test/output/event_output_olb.isxd'

        delete_files_silently([output_file])

        with pytest.raises(Exception) as error:
            isx.event_detection(input_file, output_file, threshold=0.25,
                                tau=0.500, event_time_ref='beginning',
                                ignore_negative_transients=True, accepted_cells_only=False)
        assert 'Expected data set to be of type: Cell Set' in str(error.value)

        assert not is_file(output_file)
        delete_files_silently([output_file])

    @pytest.mark.isxd_movie
    def test_TemporalCropMovie(self):
        input_file  = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_file = test_data_path + '/unit_test/output/trim_output.isxd'
        expected    = test_data_path + '/unit_test/guilded/exp_mosaicTemporalCropMovie_output-v2.isxd'
        delete_files_silently([output_file])

        seg_indices = [(1, 5)]

        isx.trim_movie(input_file, output_file, seg_indices)

        exp_movie = isx.Movie.read(expected)
        assert exp_movie.file_path == expected
        act_movie = isx.Movie.read(output_file)
        assert act_movie.file_path == output_file

        exp_movie.spacing._impl.pixel_width = isx._internal.IsxRatio(3, 1)
        exp_movie.spacing._impl.pixel_height = isx._internal.IsxRatio(3, 1)

        assert_isxd_movies_are_close(exp_movie, act_movie)

        assert act_movie.timing.cropped == seg_indices

        del exp_movie
        del act_movie
        delete_files_silently([output_file])

    @pytest.mark.isxd_movie
    def test_TemporalCropMovieManySegments(self):
        input_file  = test_data_path + '/unit_test/recording_20160426_145041.hdf5'
        output_file = test_data_path + '/unit_test/output/trim_output_ms.isxd'
        expected = test_data_path + '/unit_test/guilded/exp_mosaicTemporalCropMovieManySegments_output.isxd'
        delete_files_silently([output_file])

        seg_indices = [(1, 3), (5, 6)]
        isx.trim_movie(input_file, output_file, seg_indices)

        exp_movie = isx.Movie.read(expected)
        assert exp_movie.file_path == expected
        act_movie = isx.Movie.read(output_file)
        assert act_movie.file_path == output_file

        exp_movie.spacing._impl.pixel_width = isx._internal.IsxRatio(3, 1)
        exp_movie.spacing._impl.pixel_height = isx._internal.IsxRatio(3, 1)

        assert_isxd_movies_are_close(exp_movie, act_movie)

        assert act_movie.timing.cropped == seg_indices

        del exp_movie
        del act_movie
        delete_files_silently([output_file])

    @pytest.mark.isxd_trace
    @pytest.mark.csv_trace
    def test_ComputeCellMetrics(self):
        cell_set_file = test_data_path + '/unit_test/cell_metrics/cell_metrics_movie-PCA-ICA.isxd'
        events_file   = test_data_path + '/unit_test/cell_metrics/cell_metrics_movie-PCA-ICA-ED.isxd'
        output_file   = test_data_path + '/unit_test/output/cell_metrics_3cells_python.csv'
        expected      = test_data_path + '/unit_test/cell_metrics/expected_cell_metrics_3cells-v3.csv'

        delete_files_silently([output_file])

        isx.cell_metrics(cell_set_file, events_file, output_file)

        assert_csv_cell_metrics_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])
    
    @pytest.mark.isxd_trace
    @pytest.mark.csv_trace
    def test_ExportCellMetrics(self):
        cell_set_file = test_data_path + '/unit_test/cell_metrics/cell_metrics_computed-CNMFe.isxd'
        events_file   = test_data_path + '/unit_test/cell_metrics/cell_metrics_computed-CNMFe-ED.isxd'
        output_file   = test_data_path + '/unit_test/output/test.csv'

        delete_files_silently([output_file])

        isx.cell_metrics(cell_set_file, events_file, output_file, recompute_metrics=False)

        df = pd.read_csv(output_file)

        # Verify first and last rows of output file
        assert (df.iloc[0] == 
            pd.DataFrame({
                'cellName' : ['C00'],
                'snr' : [48.5975],
                'mad' : [0.198209],
                'eventRate' : [0.2],
                'eventAmpMedian' : [9.63244],
                'eventAmpSD' : [0.0],
                'riseMedian' : [0.1],
                'riseSD' : [0.0],
                'decayMedian' : [0.2],
                'decaySD' : [0.0],
                'numContourComponents' : [1],
                'overallCenterInPixelsX' : [60],
                'overallCenterInPixelsY' : [56],
                'overallAreaInPixels' : [5.5],
                'overallMaxContourWidthInPixels' : [3.60555],
                'largestComponentCenterInPixelsX' : [60],
                'largestComponentCenterInPixelsY' : [56],
                'largestComponentAreaInPixels' : [5.5],
                'largestComponentMaxContourWidthInPixels' : [3.60555]
            })
        ).all(axis=None)

        assert (df.iloc[-1] == 
            pd.DataFrame({
                'cellName' : ['C19'],
                'snr' : [0.0],
                'mad' : [0.0],
                'eventRate' : [0.0],
                'eventAmpMedian' : [0.0],
                'eventAmpSD' : [0.0],
                'riseMedian' : [0.0],
                'riseSD' : [0.0],
                'decayMedian' : [0.0],
                'decaySD' : [0.0],
                'numContourComponents' : [1],
                'overallCenterInPixelsX' : [72],
                'overallCenterInPixelsY' : [67],
                'overallAreaInPixels' : [7.5],
                'overallMaxContourWidthInPixels' : [4.47214],
                'largestComponentCenterInPixelsX' : [72],
                'largestComponentCenterInPixelsY' : [67],
                'largestComponentAreaInPixels' : [7.5],
                'largestComponentMaxContourWidthInPixels' : [4.47214]
            })
        ).all(axis=None)

        delete_files_silently([output_file])

    @pytest.mark.isxd_trace
    @pytest.mark.csv_trace
    def test_ExportCellMetricsNoCellMetrics(self):
        cell_set_file = test_data_path + '/unit_test/cell_metrics/cell_metrics_movie-PCA-ICA.isxd'
        events_file   = test_data_path + '/unit_test/cell_metrics/cell_metrics_movie-PCA-ICA-ED.isxd'
        output_file   = test_data_path + '/unit_test/output/test.csv'

        delete_files_silently([output_file])

        with pytest.raises(Exception) as error:
            isx.cell_metrics(cell_set_file, events_file, output_file, recompute_metrics=False)
        assert 'Input files do not have pre-computed cell metrics stored on disk. Please compute metrics.' in str(error.value)

    @pytest.mark.isxd_trace
    def test_ApplyCellSet(self):
        input_file     = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        input_cell_set = test_data_path + '/unit_test/eventDetectionCellSet.isxd'
        output_file    = test_data_path + '/unit_test/output/test_output_applyCellSet.isxd'
        expected       = test_data_path + '/unit_test/guilded/exp_mosaicApplyCellSet_output.isxd'

        delete_files_silently([output_file])

        isx.apply_cell_set(input_file, input_cell_set, output_file, threshold=0.0)

        assert_isxd_cellsets_are_close_by_path(expected, output_file)

        delete_files_silently([output_file])

    def test_ApplyCellSet_negative_input_file_not_cellset(self):
        input_file     = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'
        input_cell_set = test_data_path + '/unit_test/50fr10_l2-3cells_he.isxd'
        output_file    = test_data_path + '/unit_test/output/test_output_applyCellSet.isxd'
        expected       = test_data_path + '/unit_test/guilded/exp_mosaicApplyCellSet_output.isxd'

        delete_files_silently([output_file])

        with pytest.raises(Exception) as error:
            isx.apply_cell_set(input_file, input_cell_set, output_file, threshold=0.0)
        assert 'Expected data set to be of type: Cell Set' in str(error.value)

        assert not is_file(output_file)
        delete_files_silently([output_file])

    @pytest.mark.isxd_trace
    def test_ApplyRois(self):
        input_movie_files = [
            test_data_path + '/unit_test/longReg_movie0.isxd',
            test_data_path + '/unit_test/longReg_movie1.isxd'
        ]
        output_cell_set_files = [
            test_data_path + '/unit_test/output/test_output_applyRoi0.isxd',
            test_data_path + '/unit_test/output/test_output_applyRoi1.isxd'
        ]

        delete_files_silently(output_cell_set_files)

        rois = [
            [(30, 71),(30, 71),(29, 71),(28, 69),(27, 69),(26, 69),(25, 69),(25, 68),(24, 68),(23, 68),(22, 68),(22, 68),(21, 68),(21, 69),(20, 69),(20, 69),(19, 70),(19, 71),(19, 72),(19, 73),(18, 73),(18, 75),(18, 76),(19, 77),(19, 78),(20, 79),(20, 80),(21, 80),(24, 81),(25, 81),(27, 81),(28, 81),(29, 81),(30, 81),(30, 80),(30, 80),(31, 78),(31, 77),(31, 76),(31, 75),(31, 75),(31, 75),(31, 74),(31, 73),(31, 73),(31, 72),(31, 72),(30, 72),(30, 72),(30, 72),(30, 71)],
            [(95, 55),(95, 55),(94, 55),(92, 55),(91, 54),(90, 54),(90, 54),(89, 54),(89, 54),(88, 54),(87, 55),(86, 55),(86, 56),(85, 56),(85, 57),(85, 57),(85, 58),(85, 58),(85, 59),(85, 60),(85, 61),(85, 61),(86, 62),(86, 62),(87, 63),(88, 64),(88, 65),(89, 65),(90, 66),(90, 66),(91, 66),(92, 66),(93, 65),(94, 65),(95, 64),(95, 64),(95, 64),(95, 62),(96, 60),(96, 59),(96, 58),(96, 57),(96, 57),(95, 57),(95, 56),(95, 56),(95, 56),(95, 56),(95, 55)],
            [(153, 47),(153, 47),(153, 47),(150, 47),(149, 47),(148, 47),(147, 47),(146, 47),(146, 47),(145, 48),(145, 48),(145, 49),(145, 49),(145, 50),(145, 51),(145, 52),(145, 52),(145, 53),(146, 54),(147, 55),(148, 55),(149, 55),(150, 55),(152, 55),(153, 55),(154, 54),(155, 53),(155, 52),(156, 52),(156, 50),(156, 49),(156, 48),(155, 47),(155, 47),(154, 46),(154, 46),(153, 46),(153, 46),(153, 46),(153, 47)],
            [(104, 96),(103, 96),(103, 96),(101, 96),(101, 96),(100, 95),(99, 95),(99, 95),(99, 95),(98, 95),(98, 95),(98, 96),(97, 96),(97, 96),(97, 97),(97, 97),(97, 97),(97, 98),(96, 98),(96, 98),(96, 99),(95, 100),(95, 100),(95, 101),(95, 102),(95, 102),(95, 103),(95, 104),(96, 104),(96, 105),(97, 105),(97, 106),(98, 106),(99, 106),(100, 106),(101, 106),(101, 106),(102, 106),(103, 106),(104, 106),(104, 106),(104, 105),(104, 105),(105, 104),(105, 104),(106, 103),(106, 102),(106, 101),(106, 101),(106, 100),(106, 99),(106, 99),(105, 98),(105, 98),(104, 97),(104, 97),(104, 97),(104, 97),(103, 97),(103, 97),(104, 96)],
            [(104, 174),(104, 174),(104, 173),(102, 172),(101, 172),(100, 171),(100, 171),(99, 171),(97, 171),(96, 171),(96, 172),(95, 172),(95, 173),(95, 173),(95, 175),(95, 176),(95, 177),(95, 178),(96, 179),(97, 180),(98, 181),(100, 181),(101, 181),(103, 180),(104, 179),(104, 179),(105, 178),(105, 177),(105, 176),(105, 175),(105, 174),(104, 173),(104, 173),(104, 173),(104, 174)],
            [(177, 147),(177, 147),(176, 147),(174, 146),(174, 146),(173, 146),(172, 147),(171, 147),(171, 147),(171, 148),(171, 148),(171, 148),(171, 149),(171, 150),(171, 150),(171, 151),(171, 152),(172, 153),(172, 154),(173, 154),(174, 155),(175, 155),(176, 155),(178, 155),(180, 154),(180, 153),(181, 152),(181, 151),(181, 149),(181, 148),(180, 147),(179, 147),(178, 146),(178, 146),(177, 146),(177, 146),(177, 146),(177, 147)],
            [(22, 121),(22, 121),(22, 121),(19, 121),(18, 121),(18, 121),(18, 121),(18, 122),(17, 122),(16, 123),(16, 124),(15, 124),(15, 125),(15, 126),(15, 127),(16, 128),(16, 128),(17, 129),(18, 130),(19, 130),(21, 130),(22, 130),(23, 129),(24, 128),(25, 128),(25, 128),(25, 127),(25, 127),(25, 126),(25, 124),(26, 123),(26, 122),(26, 122),(25, 122),(25, 121),(24, 121),(24, 121),(23, 121),(23, 121),(23, 121),(22, 121)],
        ]

        cell_names = [
            "c0",
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
            "c6"
        ]

        isx.apply_rois(
            input_movie_files=input_movie_files, 
            output_cell_set_files=output_cell_set_files,
            rois=rois,
            cell_names=cell_names
        )

        # verify trace sums of output cell sets
        expected_trace_sums = [
            127510,
            79431,
            163446,
            154379,
            80295,
            165443,
            75685,
        ]

        assert_isxd_cellsets_trace_sums(
            output_cell_set_files,
            expected_trace_sums
        )

        assert_isxd_cellsets_cell_names(
            output_cell_set_files,
            cell_names
        )

        delete_files_silently(output_cell_set_files)
    
    @pytest.mark.isxd_trace
    def test_ApplyRoisNoCellNames(self):
        input_movie_files = [
            test_data_path + '/unit_test/longReg_movie0.isxd',
            test_data_path + '/unit_test/longReg_movie1.isxd'
        ]
        output_cell_set_files = [
            test_data_path + '/unit_test/output/test_output_applyRoi0.isxd',
            test_data_path + '/unit_test/output/test_output_applyRoi1.isxd'
        ]

        delete_files_silently(output_cell_set_files)

        rois = [
            [(30, 71),(30, 71),(29, 71),(28, 69),(27, 69),(26, 69),(25, 69),(25, 68),(24, 68),(23, 68),(22, 68),(22, 68),(21, 68),(21, 69),(20, 69),(20, 69),(19, 70),(19, 71),(19, 72),(19, 73),(18, 73),(18, 75),(18, 76),(19, 77),(19, 78),(20, 79),(20, 80),(21, 80),(24, 81),(25, 81),(27, 81),(28, 81),(29, 81),(30, 81),(30, 80),(30, 80),(31, 78),(31, 77),(31, 76),(31, 75),(31, 75),(31, 75),(31, 74),(31, 73),(31, 73),(31, 72),(31, 72),(30, 72),(30, 72),(30, 72),(30, 71)],
            [(95, 55),(95, 55),(94, 55),(92, 55),(91, 54),(90, 54),(90, 54),(89, 54),(89, 54),(88, 54),(87, 55),(86, 55),(86, 56),(85, 56),(85, 57),(85, 57),(85, 58),(85, 58),(85, 59),(85, 60),(85, 61),(85, 61),(86, 62),(86, 62),(87, 63),(88, 64),(88, 65),(89, 65),(90, 66),(90, 66),(91, 66),(92, 66),(93, 65),(94, 65),(95, 64),(95, 64),(95, 64),(95, 62),(96, 60),(96, 59),(96, 58),(96, 57),(96, 57),(95, 57),(95, 56),(95, 56),(95, 56),(95, 56),(95, 55)],
            [(153, 47),(153, 47),(153, 47),(150, 47),(149, 47),(148, 47),(147, 47),(146, 47),(146, 47),(145, 48),(145, 48),(145, 49),(145, 49),(145, 50),(145, 51),(145, 52),(145, 52),(145, 53),(146, 54),(147, 55),(148, 55),(149, 55),(150, 55),(152, 55),(153, 55),(154, 54),(155, 53),(155, 52),(156, 52),(156, 50),(156, 49),(156, 48),(155, 47),(155, 47),(154, 46),(154, 46),(153, 46),(153, 46),(153, 46),(153, 47)],
            [(104, 96),(103, 96),(103, 96),(101, 96),(101, 96),(100, 95),(99, 95),(99, 95),(99, 95),(98, 95),(98, 95),(98, 96),(97, 96),(97, 96),(97, 97),(97, 97),(97, 97),(97, 98),(96, 98),(96, 98),(96, 99),(95, 100),(95, 100),(95, 101),(95, 102),(95, 102),(95, 103),(95, 104),(96, 104),(96, 105),(97, 105),(97, 106),(98, 106),(99, 106),(100, 106),(101, 106),(101, 106),(102, 106),(103, 106),(104, 106),(104, 106),(104, 105),(104, 105),(105, 104),(105, 104),(106, 103),(106, 102),(106, 101),(106, 101),(106, 100),(106, 99),(106, 99),(105, 98),(105, 98),(104, 97),(104, 97),(104, 97),(104, 97),(103, 97),(103, 97),(104, 96)],
            [(104, 174),(104, 174),(104, 173),(102, 172),(101, 172),(100, 171),(100, 171),(99, 171),(97, 171),(96, 171),(96, 172),(95, 172),(95, 173),(95, 173),(95, 175),(95, 176),(95, 177),(95, 178),(96, 179),(97, 180),(98, 181),(100, 181),(101, 181),(103, 180),(104, 179),(104, 179),(105, 178),(105, 177),(105, 176),(105, 175),(105, 174),(104, 173),(104, 173),(104, 173),(104, 174)],
            [(177, 147),(177, 147),(176, 147),(174, 146),(174, 146),(173, 146),(172, 147),(171, 147),(171, 147),(171, 148),(171, 148),(171, 148),(171, 149),(171, 150),(171, 150),(171, 151),(171, 152),(172, 153),(172, 154),(173, 154),(174, 155),(175, 155),(176, 155),(178, 155),(180, 154),(180, 153),(181, 152),(181, 151),(181, 149),(181, 148),(180, 147),(179, 147),(178, 146),(178, 146),(177, 146),(177, 146),(177, 146),(177, 147)],
            [(22, 121),(22, 121),(22, 121),(19, 121),(18, 121),(18, 121),(18, 121),(18, 122),(17, 122),(16, 123),(16, 124),(15, 124),(15, 125),(15, 126),(15, 127),(16, 128),(16, 128),(17, 129),(18, 130),(19, 130),(21, 130),(22, 130),(23, 129),(24, 128),(25, 128),(25, 128),(25, 127),(25, 127),(25, 126),(25, 124),(26, 123),(26, 122),(26, 122),(25, 122),(25, 121),(24, 121),(24, 121),(23, 121),(23, 121),(23, 121),(22, 121)],
        ]

        isx.apply_rois(
            input_movie_files=input_movie_files, 
            output_cell_set_files=output_cell_set_files,
            rois=rois,
            cell_names=[]
        )

        # verify trace sums of output cell sets
        expected_trace_sums = [
            127510,
            79431,
            163446,
            154379,
            80295,
            165443,
            75685,
        ]

        assert_isxd_cellsets_trace_sums(
            output_cell_set_files,
            expected_trace_sums
        )
        
        delete_files_silently(output_cell_set_files)

    @pytest.mark.isxd_trace
    @pytest.mark.isxd_movie
    def test_LongitudinalRegistration(self):
        input_dir = test_data_path + '/unit_test'
        output_dir = input_dir + '/output'
        exp_dir = input_dir + '/guilded'

        input_cellset_filenames = [input_dir + '/longReg_cellSet{}.isxd'.format(i) for i in range(3)]
        output_cellset_filenames = [output_dir + '/test_output_longReg_cellSet{}.isxd'.format(i) for i in range(3)]
        input_movie_filenames = [input_dir + '/longReg_movie{}.isxd'.format(i) for i in range(3)]
        output_movie_filenames = [output_dir + '/test_output_longReg_movie{}.isxd'.format(i) for i in range(3)]
        expected_cells = [exp_dir + '/exp_mosaicLongitudinalRegistration_CellOutput{}.isxd'.format(i) for i in range(3)]
        expected_movies = [exp_dir + '/exp_mosaicLongitudinalRegistration_MovieOutput{}.isxd'.format(i) for i in range(3)]

        delete_files_silently(output_cellset_filenames)
        delete_files_silently(output_movie_filenames)

        isx.longitudinal_registration(input_cellset_filenames, output_cellset_filenames, input_movie_files=input_movie_filenames, output_movie_files=output_movie_filenames)

        for f in range(3):
            assert_isxd_cellsets_are_close_by_path(expected_cells[f], output_cellset_filenames[f])
            assert_isxd_movies_are_close_by_path(expected_movies[f], output_movie_filenames[f])

        delete_files_silently(output_cellset_filenames)
        delete_files_silently(output_movie_filenames)

    @pytest.mark.isxd_trace
    @pytest.mark.isxd_movie
    def test_LongitudinalRegistrationWithCsvs(self):
        input_dir = test_data_path + '/unit_test'
        output_dir = input_dir + '/output'
        exp_dir = input_dir + '/guilded'

        input_cellset_filenames = [input_dir + '/longReg_cellSet{}.isxd'.format(i) for i in range(3)]
        output_cellset_filenames = [output_dir + '/test_output_longReg_cellSet{}.isxd'.format(i) for i in range(3)]
        input_movie_filenames = [input_dir + '/longReg_movie{}.isxd'.format(i) for i in range(3)]
        output_movie_filenames = [output_dir + '/test_output_longReg_movie{}.isxd'.format(i) for i in range(3)]
        expected_cells = [exp_dir + '/exp_mosaicLongitudinalRegistration_CellOutput{}.isxd'.format(i) for i in range(3)]
        expected_movies = [exp_dir + '/exp_mosaicLongitudinalRegistration_MovieOutput{}.isxd'.format(i) for i in range(3)]

        output_corr_filename = output_dir + '/longReg_corr.csv'
        expected_corr_filename = exp_dir + '/exp_mosaicLongitudinalRegistration_corr.csv'

        output_tfm_filename = output_dir + '/longReg_transforms.csv'
        expected_tfm_filename = exp_dir + '/exp_mosaicLongitudinalRegistration_transforms-v2.csv'

        output_crop_filename = output_dir + '/longReg_crop.csv'
        expected_crop_filename = exp_dir + '/exp_mosaicLongitudinalRegistration_crop.csv'

        delete_files_silently(output_cellset_filenames)
        delete_files_silently(output_movie_filenames)
        delete_files_silently([output_corr_filename])
        delete_files_silently([output_tfm_filename])
        delete_files_silently([output_crop_filename])

        isx.longitudinal_registration(input_cellset_filenames, output_cellset_filenames, input_movie_files=input_movie_filenames, output_movie_files=output_movie_filenames, csv_file=output_corr_filename, transform_csv_file=output_tfm_filename, crop_csv_file=output_crop_filename)

        for f in range(3):
            assert_isxd_cellsets_are_close_by_path(expected_cells[f], output_cellset_filenames[f])
            assert_isxd_movies_are_close_by_path(expected_movies[f], output_movie_filenames[f])

        assert_csv_files_are_close_by_path(expected_corr_filename, output_corr_filename)
        assert_csv_files_are_equal_by_path(expected_crop_filename, output_crop_filename)
        assert_csv_files_are_equal_by_path(expected_tfm_filename, output_tfm_filename)

        delete_files_silently(output_cellset_filenames)
        delete_files_silently(output_movie_filenames)
        delete_files_silently([output_corr_filename])
        delete_files_silently([output_tfm_filename])
        delete_files_silently([output_crop_filename])

    @pytest.mark.parametrize(('n_of_not_cellset'), range(3))
    def test_LongitudinalRegistration_negativeinput_file_not_cellset(self, n_of_not_cellset):
        not_existing_file_path = test_data_path + 'not_existing_file.isxd'
        input_cellset_filenames = [test_data_path + '/unit_test/longReg_cellSet0.isxd',
                                   test_data_path + '/unit_test/longReg_cellSet1.isxd',
                                   test_data_path + '/unit_test/longReg_cellSet2.isxd']
        output_cellset_filenames = [test_data_path + '/unit_test/output/test_output_longReg_cellSet0.isxd',
                                    test_data_path + '/unit_test/output/test_output_longReg_cellSet1.isxd',
                                    test_data_path + '/unit_test/output/test_output_longReg_cellSet2.isxd']
        input_movie_filenames = [test_data_path + '/unit_test/longReg_movie0.isxd',
                                 test_data_path + '/unit_test/longReg_movie1.isxd',
                                 test_data_path + '/unit_test/longReg_movie2.isxd']
        output_movie_filenames = [test_data_path + '/unit_test/output/test_output_longReg_movie0.isxd',
                                  test_data_path + '/unit_test/output/test_output_longReg_movie1.isxd',
                                  test_data_path + '/unit_test/output/test_output_longReg_movie2.isxd']

        delete_files_silently(output_cellset_filenames)
        delete_files_silently(output_movie_filenames)

        input_cellset_filenames[n_of_not_cellset] = not_existing_file_path

        with pytest.raises(Exception) as error:
            isx.longitudinal_registration(input_cellset_filenames,
                                          output_cellset_filenames,
                                          input_movie_files=input_movie_filenames,
                                          output_movie_files=output_movie_filenames)
        assert 'File does not exist' in str(error.value)

        for f in range(3):
            assert not is_file(output_cellset_filenames[f])
            assert not is_file(output_movie_filenames[f])

        delete_files_silently(output_cellset_filenames)
        delete_files_silently(output_movie_filenames)

    @pytest.mark.isxd_events
    def test_AutoAcceptReject(self):
        original_cellset_file = test_data_path + '/unit_test/classify_cell_statuses/50fr10_l1-3cells_he-PCA-ICA.isxd'
        input_events_file = test_data_path + '/unit_test/classify_cell_statuses/50fr10_l1-3cells_he-PCA-ICA-ED.isxd'
        input_cellset_file = test_data_path + '/unit_test/output/test_output_classifyCellStatus.isxd'
        copyfile(original_cellset_file, input_cellset_file)

        expected = test_data_path + '/unit_test/guilded/exp_mosaicClassifyCellStatus_output_default.isxd'

        isx.auto_accept_reject(input_cellset_file, input_events_file)

        assert_isxd_cellsets_are_close_by_path(expected, input_cellset_file)

        delete_files_silently([input_cellset_file])

    @pytest.mark.isxd_events
    def test_AutoAcceptReject_CustomFilter(self):
        original_cellset_file = test_data_path + '/unit_test/classify_cell_statuses/50fr10_l1-3cells_he-PCA-ICA.isxd'
        input_events_file = test_data_path + '/unit_test/classify_cell_statuses/50fr10_l1-3cells_he-PCA-ICA-ED.isxd'
        input_cellset_file = test_data_path + '/unit_test/output/test_output_classifyCellStatus.isxd'
        copyfile(original_cellset_file, input_cellset_file)

        expected_ARA = test_data_path + '/unit_test/guilded/exp_mosaicClassifyCellStatus_output_ARA.isxd'
        filter_ARA = [('SNR', '>', 1.35)]

        isx.auto_accept_reject(input_cellset_file, input_events_file, filter_ARA)

        assert_isxd_cellsets_are_close_by_path(expected_ARA, input_cellset_file)

        delete_files_silently([input_cellset_file])

    @pytest.mark.isxd_events
    def test_AutoAcceptReject_CustomFilters(self):
        original_cellset_file = test_data_path + '/unit_test/classify_cell_statuses/50fr10_l1-3cells_he-PCA-ICA.isxd'
        input_events_file = test_data_path + '/unit_test/classify_cell_statuses/50fr10_l1-3cells_he-PCA-ICA-ED.isxd'
        input_cellset_file = test_data_path + '/unit_test/output/test_output_classifyCellStatus.isxd'
        copyfile(original_cellset_file, input_cellset_file)

        expected_ARR = test_data_path + '/unit_test/guilded/exp_mosaicClassifyCellStatus_output_ARR.isxd'
        filter_ARR = [('SNR', '>', 1.32), ('Event Rate', '>', 1), ('Cell Size', '>', 6)]

        isx.auto_accept_reject(input_cellset_file, input_events_file, filter_ARR)

        assert_isxd_cellsets_are_close_by_path(expected_ARR, input_cellset_file)

        delete_files_silently([input_cellset_file])

    def test_AutoAcceptReject_negative_input_file_not_cellset(self):
        input_events_file = test_data_path + '/unit_test/classify_cell_statuses/50fr10_l1-3cells_he-PCA-ICA-ED.isxd'
        input_cellset_file = test_data_path + '/unit_test/50fr10_l1-3cells_he.isxd'

        with pytest.raises(Exception) as error:
            isx.auto_accept_reject(input_cellset_file, input_events_file)
        assert 'Expected data set to be of type: Cell Set' in str(error.value)

    # def test_MultiplaneRegistration(self):
    #     input_cell_set_files = [
    #         test_data_path + '/unit_test/mcr/mcr_in1.isxd',
    #         test_data_path + '/unit_test/mcr/mcr_in2.isxd',
    #         test_data_path + '/unit_test/mcr/mcr_in3.isxd'
    #     ]
    #     lcr_output_files = [x + '-LCR.isxd' for x in input_cell_set_files]
    #     mcr_output_file = test_data_path + '/unit_test/mcr/test-MCR.isxd'
    #     expected_cell_set_file = test_data_path + '/unit_test/mcr/mcr_exp.isxd'

    #     isx.multiplane_registration(input_cell_set_files, mcr_output_file)
    #     assert_isxd_cellsets_are_close_by_path(expected_cell_set_file, mcr_output_file)

    #     tr_paths = [
    #         test_data_path + '/unit_test/mcr/test-LCR_001-TR.isxd',
    #         test_data_path + '/unit_test/mcr/test-LCR_002-TR.isxd',
    #         test_data_path + '/unit_test/mcr/test-LCR_003-TR.isxd'
    #     ]
    #     ed_paths = [
    #         test_data_path + '/unit_test/mcr/test-LCR_001-TR-ED.isxd',
    #         test_data_path + '/unit_test/mcr/test-LCR_002-TR-ED.isxd',
    #         test_data_path + '/unit_test/mcr/test-LCR_003-TR-ED.isxd'
    #     ]
    #     delete_files_silently(lcr_output_files + tr_paths + ed_paths + [mcr_output_file])

    @pytest.mark.isxd_movie
    def test_DeinterleaveDualcolorMovie(self):
        test_dir = test_data_path + '/unit_test/dual_color'
        input_movie_files = [test_dir + '/DualColorMultiplexingMovie.isxd']
        output_green_movie_files = [test_data_path + '/unit_test/output/' + 'tmp_output_green_channel.isxd']
        output_red_movie_files = [test_data_path + '/unit_test/output/' + 'tmp_output_red_channel.isxd']

        # IDPS-857 Upgrade version of test files since higher precision sampling rate results
        # in a slightly different computed start time for de-interleaved movies
        # IDPS-900 Upgrade version of test files since epoch start time is calculated based on tsc values
        expected_green_movies = [test_data_path + '/unit_test/guilded/' + 'de_interleave_dualcolor_multiplexing_green_channel_v2.isxd']
        expected_red_movies = [test_data_path + '/unit_test/guilded/' + 'de_interleave_dualcolor_multiplexing_red_channel_v2.isxd']

        delete_files_silently(output_green_movie_files)
        delete_files_silently(output_red_movie_files)

        isx.de_interleave_dualcolor(input_movie_files, output_green_movie_files, output_red_movie_files)

        for o, e in zip(output_green_movie_files, expected_green_movies):
            assert_isxd_movies_are_close_by_path(e, o)
        for o, e in zip(output_red_movie_files, expected_red_movies):
            assert_isxd_movies_are_close_by_path(e, o)

        delete_files_silently(output_green_movie_files)
        delete_files_silently(output_red_movie_files)

    @pytest.mark.isxd_movie
    def test_DeinterleaveDualcolorMovieWideField(self):
        test_dir = test_data_path + '/unit_test/widefield'
        input_movie_files = [test_dir + '/DualSpheres_2023-11-10-11-34-11_video_multiplexing-PP-TPC.isxd']
        output_green_movie_files = [test_data_path + '/unit_test/output/' + 'tmp_output_green_channel.isxd']
        output_red_movie_files = [test_data_path + '/unit_test/output/' + 'tmp_output_red_channel.isxd']

        delete_files_silently(output_green_movie_files)
        delete_files_silently(output_red_movie_files)

        isx.de_interleave_dualcolor(input_movie_files, output_green_movie_files, output_red_movie_files, correct_chromatic_shift=True)

        green_movie = isx.Movie.read(output_green_movie_files[0])
        first_green_frame = green_movie.get_frame_data(0)
        expected_corrected_frame_sum = 73429192
        assert np.sum(first_green_frame) == expected_corrected_frame_sum
        del green_movie

        delete_files_silently(output_green_movie_files)
        delete_files_silently(output_red_movie_files)

    @pytest.mark.isxd_movie
    def test_DeinterleaveDualcolorMovieWideFieldNoCorrection(self):
        test_dir = test_data_path + '/unit_test/widefield'
        input_movie_files = [test_dir + '/DualSpheres_2023-11-10-11-34-11_video_multiplexing-PP-TPC.isxd']
        output_green_movie_files = [test_data_path + '/unit_test/output/' + 'tmp_output_green_channel.isxd']
        output_red_movie_files = [test_data_path + '/unit_test/output/' + 'tmp_output_red_channel.isxd']

        delete_files_silently(output_green_movie_files)
        delete_files_silently(output_red_movie_files)

        isx.de_interleave_dualcolor(input_movie_files, output_green_movie_files, output_red_movie_files, correct_chromatic_shift=False)

        green_movie = isx.Movie.read(output_green_movie_files[0])
        first_green_frame = green_movie.get_frame_data(0)
        expected_corrected_frame_sum = 73429192
        assert np.sum(first_green_frame) != expected_corrected_frame_sum
        del green_movie

        delete_files_silently(output_green_movie_files)
        delete_files_silently(output_red_movie_files)

    # def test_MulticolorRegistration_AcceptedOnly(self):
    #     test_dir = test_data_path + '/unit_test/dual_color'
    #     input_cellset_file1 = test_dir + '/cellset_green_dynamic.isxd'
    #     input_cellset_file2 = test_dir + '/cellset_red_static.isxd'
    #     output_directory = test_data_path + '/unit_test/tmp_output_multicolor_reg'
    #     output_spatial_overlap_csv_file = output_directory + '/tmp_output_spatial_overlap.csv'
    #     output_registration_matrix_csv_file = output_directory + '/tmp_output_reg_matrix.csv'

    #     delete_dirs_silently(output_directory)
    #     os.makedirs(output_directory)

    #     pad_value = np.nan
    #     lower_threshold = 0.1
    #     upper_threshold = 0.3
    #     accepted_cells_only = True

    #     isx.multicolor_registration(
    #         input_cellset_file1, input_cellset_file2, output_spatial_overlap_csv_file, output_registration_matrix_csv_file,
    #         output_directory, pad_value, lower_threshold, upper_threshold,
    #         accepted_cells_only
    #     )

    #     # intermediate files
    #     assert os.path.exists(output_directory + "/cellset_red_static-CSB.isxd")
    #     assert os.path.exists(output_directory + "/cellset_green_dynamic-CST.isxd")
    #     assert os.path.exists(output_directory + "/cellset_green_dynamic-CST-CSB.isxd")
    #     assert os.path.exists(output_directory + "/cellset_red_static-CSB-cellmap.tiff")
    #     assert os.path.exists(output_directory + "/cellset_green_dynamic-CST-CSB-cellmap.tiff")
    #     assert os.path.exists(output_directory + "/cellset_green_dynamic-CST-CSB-cellmap-overlay.tiff")

    #     # spatial overlap
    #     expected_spatial_overlap_csv_file = output_directory + '/expected_output_spatial_overlap.csv'
    #     spatial_overlap_data = [["", "C01", "C02"],
    #                             ["C0", 1, 0],
    #                             ["C1", 0, 0.333333],
    #                             ["C2", 0, 0]]
    #     with open(expected_spatial_overlap_csv_file, 'w', newline='') as file:
    #         writer = csv.writer(file, delimiter=',')
    #         writer.writerows(spatial_overlap_data)
    #     assert_txt_files_are_equal_by_path(expected_spatial_overlap_csv_file, output_spatial_overlap_csv_file)

    #     # reg matrix
    #     expected_registration_matrix_csv_file = output_directory + '/expected_output_reg_matrix.csv'
    #     reg_matrix_data = [["","primary","max_jaccard_index","secondary","match","colocalization"],
    #                         [0,"C0",1,"C01",True,True],
    #                         [1,"C1",0.333333,"C02",True,True],
    #                         [2,"C2",0,"C01",False,False]]
    #     with open(expected_registration_matrix_csv_file, 'w', newline='') as file:
    #         writer = csv.writer(file, delimiter=',')
    #         writer.writerows(reg_matrix_data)
    #     assert_txt_files_are_equal_by_path(expected_registration_matrix_csv_file, output_registration_matrix_csv_file)

    #     delete_dirs_silently(output_directory)

    # def test_MulticolorRegistration_AcceptedAndUndecided(self):
    #     test_dir = test_data_path + '/unit_test/dual_color'
    #     input_cellset_file1 = test_dir + '/cellset_green_dynamic.isxd'
    #     input_cellset_file2 = test_dir + '/cellset_red_static.isxd'
    #     output_directory = test_data_path + '/unit_test/tmp_output_multicolor_reg'
    #     output_spatial_overlap_csv_file = output_directory + '/tmp_output_spatial_overlap.csv'
    #     output_registration_matrix_csv_file = output_directory + '/tmp_output_reg_matrix.csv'

    #     delete_dirs_silently(output_directory)
    #     os.makedirs(output_directory)

    #     pad_value = np.nan
    #     lower_threshold = 0.35
    #     upper_threshold = 0.5
    #     accepted_cells_only = False

    #     isx.multicolor_registration(
    #         input_cellset_file1, input_cellset_file2, output_spatial_overlap_csv_file, output_registration_matrix_csv_file,
    #         output_directory, pad_value, lower_threshold, upper_threshold,
    #         accepted_cells_only
    #     )

    #     # intermediate files
    #     assert os.path.exists(output_directory + "/cellset_red_static-CSB.isxd")
    #     assert os.path.exists(output_directory + "/cellset_green_dynamic-CST.isxd")
    #     assert os.path.exists(output_directory + "/cellset_green_dynamic-CST-CSB.isxd")
    #     assert os.path.exists(output_directory + "/cellset_red_static-CSB-cellmap.tiff")
    #     assert os.path.exists(output_directory + "/cellset_green_dynamic-CST-CSB-cellmap.tiff")
    #     assert os.path.exists(output_directory + "/cellset_green_dynamic-CST-CSB-cellmap-overlay.tiff")

    #     # spatial overlap
    #     expected_spatial_overlap_csv_file = output_directory + '/expected_output_spatial_overlap.csv'
    #     spatial_overlap_data = [["", "C01", "C02"],
    #                             ["C0", 1, 0],
    #                             ["C1", 0, 0.333333],
    #                             ["C2", 0, 0],
    #                             ["C3", 0.5625, 0]]
    #     with open(expected_spatial_overlap_csv_file, 'w', newline='') as file:
    #         writer = csv.writer(file, delimiter=',')
    #         writer.writerows(spatial_overlap_data)
    #     assert_txt_files_are_equal_by_path(expected_spatial_overlap_csv_file, output_spatial_overlap_csv_file)

    #     # reg matrix
    #     expected_registration_matrix_csv_file = output_directory + '/expected_output_reg_matrix.csv'
    #     reg_matrix_data = [["","primary","max_jaccard_index","secondary","match","colocalization"],
    #                        [0,"C0",1,"C01",True,True],
    #                        [1,"C1",0.333333,"C02",False,False],
    #                        [2,"C2",0,"C01",False,False],
    #                        [3,"C3",0.5625,"C01",False,""]]
    #     with open(expected_registration_matrix_csv_file, 'w', newline='') as file:
    #         writer = csv.writer(file, delimiter=',')
    #         writer.writerows(reg_matrix_data)
    #     assert_txt_files_are_equal_by_path(expected_registration_matrix_csv_file, output_registration_matrix_csv_file)

    #     delete_dirs_silently(output_directory)

    def test_binarize_cellset_absolute_threshold(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file = os.path.join(base_dir, 'cellset_pcaica_2cells.isxd')
        actual_output_cellset_file = os.path.join(output_dir, 'actual_binary_cellset.isxd')
        expected_output_cellset_file = os.path.join(base_dir, 'cellset_pcaica_2cells_absolute_threshold_5.isxd')
        delete_files_silently([actual_output_cellset_file])

        isx.binarize_cell_set(input_cellset_file, actual_output_cellset_file, threshold=5, use_percentile_threshold=False)

        assert_isxd_cellsets_are_close_by_path(expected_output_cellset_file.replace('\\','/'), actual_output_cellset_file.replace('\\','/'), assert_status=False)
        delete_files_silently([actual_output_cellset_file])


    def test_binarize_cellset_percentile_threshold(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file = os.path.join(base_dir, 'cellset_pcaica_2cells.isxd')
        actual_output_cellset_file = os.path.join(output_dir, 'actual_binary_cellset.isxd')
        expected_output_cellset_file = os.path.join(base_dir, 'cellset_pcaica_2cells_percentile_threshold_27.isxd')
        delete_files_silently([actual_output_cellset_file])

        isx.binarize_cell_set(input_cellset_file, actual_output_cellset_file, threshold=27, use_percentile_threshold=True)

        assert_isxd_cellsets_are_close_by_path(expected_output_cellset_file.replace('\\','/'), actual_output_cellset_file.replace('\\','/'), assert_status=False)
        delete_files_silently([actual_output_cellset_file])

    def test_crop_cell_set(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'cellset_crop')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file = os.path.join(base_dir, 'cellset_binary_5x5_3cells.isxd')
        actual_output_cellset_file = os.path.join(output_dir, 'actual_cropped_cellset.isxd')
        expected_output_cellset_file = os.path.join(base_dir, 'cellset_binary_5x5_3cells_cropped.isxd')
        delete_files_silently([actual_output_cellset_file])

        isx.crop_cell_set(input_cellset_file, actual_output_cellset_file, [2, 0, 2, 0])

        assert_isxd_cellsets_are_close_by_path(expected_output_cellset_file.replace('\\','/'), actual_output_cellset_file.replace('\\','/'))
        delete_files_silently([actual_output_cellset_file])

    def test_transform_cell_set(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'cellset_transform')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file = os.path.join(base_dir, 'input_cellset_pcaica_uneven_crop.isxd')
        actual_output_cellset_file = os.path.join(output_dir, 'actual_transformed_cellset.isxd')
        expected_output_cellset_file = os.path.join(base_dir, 'expected_cellset_pcaica_uneven_crop.isxd')
        delete_files_silently([actual_output_cellset_file])

        isx.transform_cell_set(input_cellset_file, actual_output_cellset_file, np.nan)

        assert_isxd_cellsets_are_close_by_path(expected_output_cellset_file.replace('\\','/'), actual_output_cellset_file.replace('\\','/'))
        delete_files_silently([actual_output_cellset_file])

    def test_compute_spatial_overlap_cell_set_analog(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file1 = os.path.join(base_dir, 'movie_920_green_resonant-BP-MC-CNMFE-undecided.isxd')
        input_cellset_file2 = os.path.join(base_dir, 'movie_920_green_resonant-BP-MC-CNMFE.isxd')

        actual_output_csv_file = os.path.join(output_dir, 'actual_output_scores.csv')
        expected_output_csv_file = os.path.join(base_dir, 'ncc_matrix.csv')
        delete_files_silently([actual_output_csv_file])

        isx.compute_spatial_overlap_cell_set(input_cellset_file1, input_cellset_file2, actual_output_csv_file)

        assert_csv_pairwise_spatial_overlap_matrices_are_close_by_path(expected_output_csv_file.replace('\\','/'), actual_output_csv_file.replace('\\','/'))
        delete_files_silently([actual_output_csv_file])

    def test_compute_spatial_overlap_cell_set_analog_accepted_only(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file1 = os.path.join(base_dir, 'movie_920_green_resonant-BP-MC-CNMFE-undecided.isxd')
        input_cellset_file2 = os.path.join(base_dir, 'movie_920_green_resonant-BP-MC-CNMFE.isxd')

        actual_output_csv_file = os.path.join(output_dir, 'actual_output_scores.csv')
        expected_output_csv_file = os.path.join(base_dir, 'ncc_matrix_accepted_only.csv')
        delete_files_silently([actual_output_csv_file])

        isx.compute_spatial_overlap_cell_set(input_cellset_file1, input_cellset_file2, actual_output_csv_file, accepted_cells_only=True)

        assert_csv_pairwise_spatial_overlap_matrices_are_close_by_path(expected_output_csv_file.replace('\\','/'), actual_output_csv_file.replace('\\','/'))
        delete_files_silently([actual_output_csv_file])

    def test_compute_spatial_overlap_cell_set_binary_full_overlap(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file1 = os.path.join(base_dir, 'cellset_binary_3cells_3x4fov.isxd')
        input_cellset_file2 = os.path.join(base_dir, 'cellset_binary_3cells_3x4fov.isxd')

        input_binarized_cellset_file1 =  os.path.join(output_dir, 'cellset_binary_3cells_3x4fov.isxd')
        input_binarized_cellset_file2 =  os.path.join(output_dir, 'cellset_binary_3cells_3x4fov2.isxd')

        actual_output_csv_file = os.path.join(output_dir, 'actual_output_f1_scores.csv')
        expected_output_csv_file = os.path.join(base_dir, 'cellset_binary_f1_scores_full_overlap.csv')
        delete_files_silently([actual_output_csv_file, input_binarized_cellset_file1, input_binarized_cellset_file2])

        # test data is already binarized - however we need the cell sets to have type metadata indicating they are "binary"
        # so we first binarize - which simply sets the metadata without actually changing the test data
        isx.binarize_cell_set(input_cellset_file1, input_binarized_cellset_file1, threshold=0.5, use_percentile_threshold=False)
        isx.binarize_cell_set(input_cellset_file2, input_binarized_cellset_file2, threshold=0.5, use_percentile_threshold=False)
        isx.compute_spatial_overlap_cell_set(input_binarized_cellset_file1, input_binarized_cellset_file2, actual_output_csv_file)

        assert_csv_pairwise_spatial_overlap_matrices_are_close_by_path(expected_output_csv_file.replace('\\','/'), actual_output_csv_file.replace('\\','/'))
        delete_files_silently([actual_output_csv_file, input_binarized_cellset_file1, input_binarized_cellset_file2])

    def test_compute_spatial_overlap_cell_set_binary_partial_overlap(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file1 = os.path.join(base_dir, 'cellset_binary_3cells_3x4fov.isxd')
        input_cellset_file2 = os.path.join(base_dir, 'cellset_binary_3cells_3x4fov_partial_overlap.isxd')

        input_binarized_cellset_file1 =  os.path.join(output_dir, 'cellset_binary_3cells_3x4fov.isxd')
        input_binarized_cellset_file2 =  os.path.join(output_dir, 'cellset_binary_3cells_3x4fov2.isxd')

        actual_output_csv_file = os.path.join(output_dir, 'actual_output_f1_scores.csv')
        expected_output_csv_file = os.path.join(base_dir, 'cellset_binary_f1_scores_partial_overlap.csv')
        delete_files_silently([actual_output_csv_file, input_binarized_cellset_file1, input_binarized_cellset_file2])

        # test data is already binarized - however we need the cell sets to have type metadata indicating they are "binary"
        # so we first binarize - which simply sets the metadata without actually changing the test data
        isx.binarize_cell_set(input_cellset_file1, input_binarized_cellset_file1, threshold=0.5, use_percentile_threshold=False)
        isx.binarize_cell_set(input_cellset_file2, input_binarized_cellset_file2, threshold=0.5, use_percentile_threshold=False)
        isx.compute_spatial_overlap_cell_set(input_binarized_cellset_file1, input_binarized_cellset_file2, actual_output_csv_file)

        assert_csv_pairwise_spatial_overlap_matrices_are_close_by_path(expected_output_csv_file.replace('\\','/'), actual_output_csv_file.replace('\\','/'))
        delete_files_silently([actual_output_csv_file, input_binarized_cellset_file1, input_binarized_cellset_file2])

    def test_compute_spatial_overlap_cell_set_binary_no_overlap(self):
        base_dir = os.path.join(test_data_path, 'unit_test', 'create_cell_map')
        output_dir = os.path.join(test_data_path, 'unit_test', 'output')

        input_cellset_file1 = os.path.join(base_dir, 'cellset_binary_3cells_3x4fov.isxd')
        input_cellset_file2 = os.path.join(base_dir, 'cellset_binary_3cells_3x4fov_no_overlap.isxd')

        input_binarized_cellset_file1 =  os.path.join(output_dir, 'cellset_binary_3cells_3x4fov.isxd')
        input_binarized_cellset_file2 =  os.path.join(output_dir, 'cellset_binary_3cells_3x4fov2.isxd')

        actual_output_csv_file = os.path.join(output_dir, 'actual_output_f1_scores.csv')
        expected_output_csv_file = os.path.join(base_dir, 'cellset_binary_f1_scores_no_overlap.csv')
        delete_files_silently([actual_output_csv_file, input_binarized_cellset_file1, input_binarized_cellset_file2])

        # test data is already binarized - however we need the cell sets to have type metadata indicating they are "binary"
        # so we first binarize - which simply sets the metadata without actually changing the test data
        isx.binarize_cell_set(input_cellset_file1, input_binarized_cellset_file1, threshold=0.5, use_percentile_threshold=False)
        isx.binarize_cell_set(input_cellset_file2, input_binarized_cellset_file2, threshold=0.5, use_percentile_threshold=False)
        isx.compute_spatial_overlap_cell_set(input_binarized_cellset_file1, input_binarized_cellset_file2, actual_output_csv_file)

        assert_csv_pairwise_spatial_overlap_matrices_are_close_by_path(expected_output_csv_file.replace('\\','/'), actual_output_csv_file.replace('\\','/'))
        delete_files_silently([actual_output_csv_file, input_binarized_cellset_file1, input_binarized_cellset_file2])

    def test_MulticolorRegistration_EmptyCellSet(self):
        test_dir = test_data_path + '/unit_test/dual_color'
        input_cellset_file1 = test_dir + '/cellset_green_dynamic_no_accepted.isxd'
        input_cellset_file2 = test_dir + '/cellset_red_static.isxd'
        output_directory = test_data_path + '/unit_test/tmp_output_multicolor_reg'
        output_spatial_overlap_csv_file = output_directory + '/tmp_output_spatial_overlap.csv'
        output_registration_matrix_csv_file = output_directory + '/tmp_output_reg_matrix.csv'

        delete_dirs_silently(output_directory)
        os.makedirs(output_directory)

        lower_threshold = 0.1
        upper_threshold = 0.3
        accepted_cells_only = True
        save_matched_cellset = False
        save_unmatched_cellset = False
        save_uncertain_cellset = False
        image_format = "tiff"

        try:
            isx.multicolor_registration(
                input_cellset_file1, input_cellset_file2, output_spatial_overlap_csv_file, output_registration_matrix_csv_file,
                output_directory, lower_threshold, upper_threshold,
                accepted_cells_only, save_matched_cellset, save_unmatched_cellset, save_uncertain_cellset, image_format
            )
        except Exception as e:
            assert str(e) == "Error calling C library function isx_multicolor_registration.\nThere are no cells to process"

            # check no intermediate files generated
            assert not os.path.exists(output_directory + "/cellset_red_static-BIN.isxd")
            assert not os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG.isxd")
            assert not os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN.isxd")
            assert not os.path.exists(output_directory + "/cellset_red_static-BIN-cellmap.tiff")
            assert not os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN-cellmap.tiff")
            assert not os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN-cellmap-overlay.tiff")
            assert not os.path.exists(output_spatial_overlap_csv_file)
            assert not os.path.exists(output_registration_matrix_csv_file)

        delete_dirs_silently(output_directory)

    def test_MulticolorRegistration_NonEmptyCellSet(self):
        test_dir = test_data_path + '/unit_test/dual_color'
        input_cellset_file1 = test_dir + '/cellset_green_dynamic_no_accepted.isxd'
        input_cellset_file2 = test_dir + '/cellset_red_static.isxd'
        output_directory = test_data_path + '/unit_test/tmp_output_multicolor_reg'
        output_spatial_overlap_csv_file = output_directory + '/tmp_output_spatial_overlap.csv'
        output_registration_matrix_csv_file = output_directory + '/tmp_output_reg_matrix.csv'

        delete_dirs_silently(output_directory)
        os.makedirs(output_directory)

        lower_threshold = 0.1
        upper_threshold = 0.3
        accepted_cells_only = False
        save_matched_cellset = True
        save_unmatched_cellset = True
        save_uncertain_cellset = True
        image_format = "tiff"

        isx.multicolor_registration(
            input_cellset_file1, input_cellset_file2, output_spatial_overlap_csv_file, output_registration_matrix_csv_file,
            output_directory, lower_threshold, upper_threshold,
            accepted_cells_only, save_matched_cellset, save_unmatched_cellset, save_uncertain_cellset, image_format
        )

        # intermediate files
        assert os.path.exists(output_directory + "/cellset_red_static-BIN.isxd")
        assert os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG.isxd")
        assert os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN.isxd")

        assert os.path.exists(output_directory + "/cellset_red_static-BIN-cellmap.tiff")
        assert os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN-cellmap.tiff")
        assert os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN-cellmap-overlay.tiff")

        assert not os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN-cellset-matched.isxd")
        assert os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN-cellset-non-matched.isxd")
        assert os.path.exists(output_directory + "/cellset_green_dynamic_no_accepted-REG-BIN-cellset-uncertain.isxd")

        assert os.path.exists(output_spatial_overlap_csv_file)
        assert os.path.exists(output_registration_matrix_csv_file)

        delete_dirs_silently(output_directory)

    
    def test_MulticolorRegistration_DynamicDynamicIdentity(self):
        test_dir = test_data_path + '/unit_test/dual_color'
        input_cellset_file1 = test_dir + '/cellset_green_dynamic.isxd'
        input_cellset_file2 = test_dir + '/cellset_green_dynamic.isxd'
        output_directory = test_data_path + '/unit_test/tmp_output_multicolor_reg'
        output_spatial_overlap_csv_file = output_directory + '/tmp_output_spatial_overlap.csv'
        output_registration_matrix_csv_file = output_directory + '/tmp_output_reg_matrix.csv'

        delete_dirs_silently(output_directory)
        os.makedirs(output_directory)

        lower_threshold = 0.1
        upper_threshold = 0.3
        accepted_cells_only = False
        save_matched_cellset = False
        save_unmatched_cellset = False
        save_uncertain_cellset = False

        isx.multicolor_registration(
            input_cellset_file1, input_cellset_file2, output_spatial_overlap_csv_file, output_registration_matrix_csv_file,
            output_directory, lower_threshold, upper_threshold,
            accepted_cells_only, save_matched_cellset, save_unmatched_cellset, save_uncertain_cellset
        )

        # Each cell should match with itself
        df = pd.read_csv(output_spatial_overlap_csv_file, index_col=0)
        np.testing.assert_equal(df.values.diagonal(), np.array([1., 1., 1., 1.]))

        df = pd.read_csv(output_registration_matrix_csv_file)
        for index, row in df.iterrows():
            assert row['primary'] == row['secondary']
            assert row['max_ncc'] == 1.
            assert row['match'] == "yes"
        
        delete_dirs_silently(output_directory)
        
    
    def test_CellsetRegistration_AnalogIdentity(self):
        test_dir = test_data_path + '/unit_test/dual_color'
        input_cellset_file1 = test_dir + '/cellset_green_dynamic.isxd'
        input_cellset_file2 = test_dir + '/cellset_green_dynamic.isxd'
        output_directory = test_data_path + '/unit_test/tmp_output_multicolor_reg'
        output_spatial_overlap_csv_file = output_directory + '/tmp_output_spatial_overlap.csv'
        output_registration_matrix_csv_file = output_directory + '/tmp_output_reg_matrix.csv'

        delete_dirs_silently(output_directory)
        os.makedirs(output_directory)

        isx.register_cellsets(
            input_cellset_file1,
            input_cellset_file2,
            output_spatial_overlap_csv_file,
            output_registration_matrix_csv_file,
            output_directory,
            lower_threshold=0.1,
            upper_threshold=0.3,
            accepted_cells_only=False,
            primary_cellset_name="bob",
            secondary_cellset_name="joe",
            primary_color=0x123456,
            secondary_color=0x000099
        )

        # Each cell should match with itself
        df = pd.read_csv(output_spatial_overlap_csv_file, index_col=0)
        np.testing.assert_equal(df.values.diagonal(), np.array([1., 1., 1., 1.]))

        df = pd.read_csv(output_registration_matrix_csv_file)
        for index, row in df.iterrows():
            assert row['bob'] == row['joe']
            assert row['max_ncc'] == 1.
            assert row['match'] == "yes"
        
        delete_dirs_silently(output_directory)

    def test_CellsetDeconvolve_Denoised(self):
        input_raw_cellset_file = test_data_path + '/unit_test/cellset_deconvolve/idps_movie_128x128x1000-CNMFe_OASIS.isxd'
        output_denoised_cellset_file = test_data_path + '/unit_test/output/denoised_cellset.isxd'

        expected = test_data_path + '/unit_test/guilded/exp_mosaicCellSetDeconvolveDenoised128x128x1000.isxd'

        delete_files_silently([output_denoised_cellset_file])

        isx.deconvolve_cellset(
            input_raw_cellset_file,
            output_denoised_cellset_files=output_denoised_cellset_file,
            output_spike_eventset_files=None,
            accepted_only=False,
            spike_snr_threshold=3.0,
            noise_range=(0.25, 0.5),
            noise_method='mean',
            first_order_ar=True,
            lags=5,
            fudge_factor=0.96,
            deconvolution_method='oasis')

        assert_isxd_cellsets_are_close_by_path(expected, output_denoised_cellset_file, relative_tolerance=1e-5, use_cosine=True)

        delete_files_silently([output_denoised_cellset_file])

    def test_CellsetDeconvolve_Spikes(self):
        input_raw_cellset_file = test_data_path + '/unit_test/cellset_deconvolve/idps_movie_128x128x1000-CNMFe_OASIS.isxd'
        output_spike_eventset_file = test_data_path + '/unit_test/output/spike_eventset.isxd'

        expected = test_data_path + '/unit_test/guilded/exp_mosaicCellSetDeconvolveSpikes128x128x1000.isxd'

        delete_files_silently([output_spike_eventset_file])

        isx.deconvolve_cellset(
            input_raw_cellset_file,
            output_denoised_cellset_files=None,
            output_spike_eventset_files=output_spike_eventset_file,
            accepted_only=False,
            spike_snr_threshold=3.0,
            noise_range=(0.25, 0.5),
            noise_method='mean',
            first_order_ar=True,
            lags=5,
            fudge_factor=0.96,
            deconvolution_method='oasis')

        assert_isxd_event_sets_are_close_by_path(expected, output_spike_eventset_file, relative_tolerance=1e-4)

        delete_files_silently([output_spike_eventset_file])

    def test_estimate_vessel_diameter(self):
        input_movie_files = [
            test_data_path + "/unit_test/bloodflow/bloodflow_movie_1.isxd",
            test_data_path + "/unit_test/bloodflow/bloodflow_movie_2.isxd"
        ]
        vs_out_files = [
            test_data_path + "/unit_test/output/bloodflow_movie_1_vesselset.isxd",
            test_data_path + "/unit_test/output/bloodflow_movie_2_vesselset.isxd"
        ]
        exp_vs_files =  [
            test_data_path + "/unit_test/bloodflow/blood_flow_movie_1-VD_window2s_increment1s.isxd",
            test_data_path + "/unit_test/bloodflow/blood_flow_movie_2-VD_window2s_increment1s.isxd"
        ]
        delete_files_silently(vs_out_files)

        test_contours = [
            [[96, 95], [222, 182]],
            [[348, 301], [406, 311]],
            [[439, 302], [482, 357]],
            [[110, 355], [128, 409]]
        ]

        try:
            isx.estimate_vessel_diameter(
                input_movie_files,
                vs_out_files,
                test_contours,
                time_window=2,
                time_increment=1,
                output_units="microns",
                estimation_method="Parametric FWHM",
                auto_accept_reject=False)
        except Exception as error:
            # Skip test if blood flow features are disabled in this version
            if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error):
                return
            else:
                raise error
        
        # Test traces
        for i in range(2):
            assert_isxd_vesselsets_are_close_by_path(exp_vs_files[i], vs_out_files[i])
        
        delete_files_silently(vs_out_files)

    def test_estimate_vessel_diameter_non_parametric(self):
        input_movie_files = [
            test_data_path + "/unit_test/bloodflow/bloodflow_movie_1.isxd",
            test_data_path + "/unit_test/bloodflow/bloodflow_movie_2.isxd"
        ]
        vs_out_files = [
            test_data_path + "/unit_test/output/bloodflow_movie_1_vesselset.isxd",
            test_data_path + "/unit_test/output/bloodflow_movie_2_vesselset.isxd"
        ]
        exp_vs_files =  [
            test_data_path + "/unit_test/bloodflow/blood_flow_movie_1-VD_window2s_increment1s_non_parametric.isxd",
            test_data_path + "/unit_test/bloodflow/blood_flow_movie_2-VD_window2s_increment1s_non_parametric.isxd"
        ]
        delete_files_silently(vs_out_files)

        test_contours = [
            [[96, 95], [222, 182]],
            [[348, 301], [406, 311]],
            [[439, 302], [482, 357]],
            [[110, 355], [128, 409]]
        ]

        try:
            isx.estimate_vessel_diameter(
                input_movie_files,
                vs_out_files,
                test_contours,
                time_window=2,
                time_increment=1,
                output_units="microns",
                estimation_method="Non-Parametric FWHM",
                auto_accept_reject=True,
                rejection_threshold_fraction=0.2,
                rejection_threshold_count=5)
        except Exception as error:
            # Skip test if blood flow features are disabled in this version
            if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error):
                return
            else:
                raise error
        
        # Test traces
        for i in range(2):
            assert_isxd_vesselsets_are_close_by_path(exp_vs_files[i], vs_out_files[i])
        
        delete_files_silently(vs_out_files)
    
    def test_estimate_vessel_diameter_microns_probe_none(self):
        input_movie_file = test_data_path + '/unit_test/baseplate/2021-06-28-23-34-09_video_sched_0_probe_none.isxd'
        vs_out_file = test_data_path + '/unit_test/output/bloodflow_movie_2_vesselset.isxd'
        delete_files_silently([vs_out_file])

        test_points = np.array([[[0,0],[1,1]],[[100,100],[200,200]],[[4,4],[5,5]]])

        with pytest.raises(Exception) as error:
            isx.estimate_vessel_diameter(input_movie_file, vs_out_file, test_points, 2, 2, "microns")

        # Skip test if blood flow features are disabled in this version
        if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error.value):
            return
        
        assert 'Baseplate type does not support output unit conversion to Microns. Please select "Pixels" as output units.' in str(error.value)

        assert not is_file(vs_out_file)
    
    def test_estimate_vessel_diameter_microns_probe_custom(self):
        input_movie_file = test_data_path + '/unit_test/baseplate/2021-06-28-23-45-49_video_sched_0_probe_custom.isxd'
        vs_out_file = test_data_path + '/unit_test/output/bloodflow_movie_1_vesselset.isxd'
        delete_files_silently([vs_out_file])

        test_points = np.array([[[0,0],[1,1]],[[100,100],[200,200]],[[4,4],[5,5]]])

        with pytest.raises(Exception) as error:
            isx.estimate_vessel_diameter(input_movie_file, vs_out_file, test_points, 2, 2, "microns")

        # Skip test if blood flow features are disabled in this version
        if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error.value):
            return

        assert 'Baseplate type does not support output unit conversion to Microns. Please select "Pixels" as output units.' in str(error.value)

        assert not is_file(vs_out_file)

    def test_estimate_vessel_diameter_vessel_names(self):
        input_movie_files = [
            test_data_path + "/unit_test/bloodflow/bloodflow_movie_1.isxd",
            test_data_path + "/unit_test/bloodflow/bloodflow_movie_2.isxd"
        ]
        vs_out_files = [
            test_data_path + "/unit_test/output/bloodflow_movie_1_vesselset.isxd",
            test_data_path + "/unit_test/output/bloodflow_movie_2_vesselset.isxd"
        ]
        delete_files_silently(vs_out_files)

        test_contours = [
            [[96, 95], [222, 182]],
            [[348, 301], [406, 311]],
            [[439, 302], [482, 357]],
            [[110, 355], [128, 409]]
        ]

        vessel_names = [
            "v0",
            "v1",
            "v2",
            "v3",
        ]

        try:
            isx.estimate_vessel_diameter(
                input_movie_files,
                vs_out_files,
                test_contours,
                time_window=1.5,
                time_increment=0.5,
                output_units="microns",
                estimation_method="Non-Parametric FWHM",
                auto_accept_reject=True,
                rejection_threshold_fraction=0.2,
                rejection_threshold_count=5,
                vessel_names=vessel_names
            )
        except Exception as error:
            # Skip test if blood flow features are disabled in this version
            if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error):
                return
            else:
                raise error
        
        # verify vessel names in output vessel set match input vessel names
        assert_isxd_vesselsets_vessel_names(
            vs_out_files,
            vessel_names
        )
        
        delete_files_silently(vs_out_files)

    def test_estimate_rbc_velocity(self):
        input_movie_files = [
            test_data_path + "/unit_test/bloodflow/rbcv_movie_1-BP.isxd",
            test_data_path + "/unit_test/bloodflow/rbcv_movie_2-BP.isxd"
        ]
        vs_out_files = [
            test_data_path + "/unit_test/output/rbcv_movie_out_vs_1.isxd",
            test_data_path + "/unit_test/output/rbcv_movie_out_vs_2.isxd"
        ]
        exp_vs_files =  [
            test_data_path + "/unit_test/bloodflow/rbcv_movie_1-RBCV_microns.isxd",
            test_data_path + "/unit_test/bloodflow/rbcv_movie_2-RBCV_microns.isxd"
        ]
        delete_files_silently(vs_out_files)

        test_contours = [
            [[124, 25], [153, 36], [90, 202], [61, 191]],
            [[24, 42], [43, 34], [85, 148], [65, 156]]
        ]
        
        try:
            isx.estimate_rbc_velocity(
                input_movie_files,
                vs_out_files,
                test_contours,
                time_window=2,
                time_increment=1,
                output_units="microns",
                save_correlation_heatmaps=True)
        except Exception as error:
            # Skip test if blood flow features are disabled in this version
            if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error):
                return
            else:
                raise error
        
        # Test traces
        for i in range(2):
            assert_isxd_vesselsets_are_close_by_path(exp_vs_files[i], vs_out_files[i])
        
        delete_files_silently(vs_out_files)

    def test_estimate_rbc_velocity_microns_probe_none(self):
        input_movie_file = test_data_path + '/unit_test/baseplate/2021-06-28-23-34-09_video_sched_0_probe_none.isxd'
        vs_out_file = test_data_path + '/unit_test/output/microns_probe_none_vesselset.isxd'
        delete_files_silently([vs_out_file])

        test_contours = np.array([ [ [0, 0], [0, 10], [10, 0], [10, 10] ] ])

        with pytest.raises(Exception) as error:
            isx.estimate_rbc_velocity(input_movie_file, vs_out_file, test_contours, 2.5, 1.25, "microns")

        # Skip test if blood flow features are disabled in this version
        if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error.value):
            return

        assert 'Baseplate type does not support output unit conversion to Microns. Please select "Pixels" as output units.' in str(error.value)

        assert not is_file(vs_out_file)

    def test_estimate_rbc_velocity_microns_probe_custom(self):
        input_movie_file = test_data_path + '/unit_test/baseplate/2021-06-28-23-45-49_video_sched_0_probe_custom.isxd'
        vs_out_file = test_data_path + '/unit_test/output/microns_probe_custom_vesselset.isxd'
        delete_files_silently([vs_out_file])

        test_contours = np.array([ [ [0, 0], [0, 10], [10, 0], [10, 10] ] ])

        with pytest.raises(Exception) as error:
            isx.estimate_rbc_velocity(input_movie_file, vs_out_file, test_contours, 2.5, 1.25, "microns")

        # Skip test if blood flow features are disabled in this version
        if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error.value):
            return

        assert 'Baseplate type does not support output unit conversion to Microns. Please select "Pixels" as output units.' in str(error.value)

        assert not is_file(vs_out_file)

    def test_create_neural_activity_movie(self):
        input_cell_set_files = [
            test_data_path + "/unit_test/cellset_crop/cellset_binary_5x5_3cells.isxd"
        ]
        output_neural_movie_files = [
            test_data_path + "/unit_test/output/tmp_cellset_binary_5x5_3cells-NA.isxd"
        ]

        isx.create_neural_activity_movie(
            input_cell_set_files,
            output_neural_movie_files,
            accepted_cells_only=False)

        expected_movie_path = test_data_path + "/unit_test/create_neural_movie/cellset_binary_5x5_3cells-NA.isxd"
        expected_movie = isx.Movie.read(expected_movie_path)
        actual_movie = isx.Movie.read(output_neural_movie_files[0])

        assert_isxd_movies_are_close(expected_movie, actual_movie)

        del expected_movie
        del actual_movie
        delete_files_silently(output_neural_movie_files)

    def test_interpolate_movie(self):
        input_movie_files = [
            test_data_path + "/unit_test/dual_color/DualColorMultiplexingMovie_green1_red4-channel_red-PP_001-TPC.isxd"
        ]
        output_interpolated_movie_files = [
            test_data_path + "/unit_test/output/DualColorMultiplexingMovie_green1_red4-channel_red-PP_001-TPC-IN.isxd"
        ]
        delete_files_silently(output_interpolated_movie_files)

        isx.interpolate_movie(
            input_movie_files,
            output_interpolated_movie_files,
            interpolate_dropped=True,
            interpolate_blank=True,
            max_consecutive_invalid_frames=2)

        expected_movie_path = test_data_path + "/unit_test/dual_color/DualColorMultiplexingMovie_green1_red4-channel_red-PP_001-TPC-IN.isxd"
        expected_movie = isx.Movie.read(expected_movie_path)
        actual_movie = isx.Movie.read(output_interpolated_movie_files[0])

        assert_isxd_movies_are_close(expected_movie, actual_movie)

        del expected_movie
        del actual_movie
        delete_files_silently(output_interpolated_movie_files)

    def test_estimate_vessel_diameter_single_vessel(self):
        input_movie_file = test_data_path + "/unit_test/bloodflow/bloodflow_movie_1.isxd"
        test_contour = [[96, 95], [222, 182]]
        start_frame = 0
        end_frame = 1000
        
        try:
            line_profile, fit, estimate, line_coords = isx.estimate_vessel_diameter_single_vessel(
                input_movie_file,
                test_contour,
                start_frame,
                end_frame,
                output_coordinates=True)
        except Exception as error:
            # Skip test if blood flow features are disabled in this version
            if "Blood flow algorithms are not available in this version of the software. Please contact support in order to enable these features." in str(error):
                return
            else:
                raise error

        exp_line_profile = np.array(
            [2.07333333,   6.30666667,   4.22      ,   0.        , 4.77      ,   5.39333333,   6.01      ,   4.63666667,
            7.05333333,   9.27      ,   7.64      ,   8.71      , 6.15666667,   8.13333333,  11.42333333,  10.37666667,
            14.42      ,  16.68666667,  13.42333333,  13.07666667, 15.63333333,  13.79      ,  17.6       ,  20.16666667,
            19.90333333,  18.05333333,  20.09666667,  22.09333333, 27.35333333,  26.99666667,  24.11333333,  28.29      ,
            30.93      ,  32.32666667,  37.73666667,  46.77333333, 58.55666667,  71.56666667,  76.56333333,  84.95      ,
            94.81      ,  99.30666667, 117.75      , 119.57666667, 124.58      , 135.04666667, 150.82666667, 154.97666667,
            161.90666667, 174.62      , 184.61      , 191.95666667, 203.53      , 210.23666667, 217.59      , 226.18      ,
            237.95      , 242.23333333, 254.26333333, 255.51666667, 259.54666667, 256.04      , 261.36666667, 263.47333333,
            268.17      , 264.17333333, 264.65      , 261.53333333, 270.47333333, 265.57333333, 271.27666667, 264.63666667,
            263.        , 259.03666667, 259.23333333, 256.96666667, 255.28666667, 253.82      , 247.49666667, 243.03666667,
            243.11666667, 236.26666667, 227.68      , 218.70333333, 213.98      , 202.26666667, 197.24333333, 181.61333333,
            176.42333333, 166.50666667, 158.40333333, 151.28333333, 138.89      , 132.61666667, 114.60333333, 109.48      ,
            103.64333333,  88.59666667,  78.71666667,  65.34      , 63.71333333,  51.06666667,  47.50333333,  44.85      ,
            49.08      ,  40.26666667,  38.83      ,  39.94666667, 35.51      ,  36.40333333,  40.23666667,  43.46666667,
            37.20333333,  37.07333333,  35.58666667,  39.43666667, 39.5       ,  40.32666667,  38.67      ,  37.65333333,
            42.38333333,  39.06      ,  35.36333333,  34.71333333, 35.58      ,  41.67666667,  38.01]
        )
        np.testing.assert_allclose(line_profile, exp_line_profile, rtol=1e5)

        exp_line_coords = np.array(
            [[ 96,  95],[ 97,  96],[ 98,  96],[ 99,  97],[100,  98],[101,  98],[102,  99],[103, 100],[104, 101],[105, 101],[106, 102],
            [107, 103],[108, 103],[109, 104],[110, 105],[111, 105],[112, 106],[113, 107],[114, 107],[115, 108],[116, 109],[117, 109],
            [118, 110],[119, 111],[120, 112],[121, 112],[122, 113],[123, 114],[124, 114],[125, 115],[126, 116],[127, 116],[128, 117],[129, 118],
            [130, 118],[131, 119],[132, 120],[133, 121],[134, 121],[135, 122],[136, 123],[137, 123],[138, 124],[139, 125],[140, 125],[141, 126],
            [142, 127],[143, 127],[144, 128],[145, 129],[146, 130],[147, 130],[148, 131],[149, 132],[150, 132],[151, 133],[152, 134],[153, 134],
            [154, 135],[155, 136],[156, 136],[157, 137],[158, 138],[159, 138],[160, 139],[161, 140],[162, 141],[163, 141],[164, 142],[165, 143],
            [166, 143],[167, 144],[168, 145],[169, 145],[170, 146],[171, 147],[172, 147],[173, 148],[174, 149],[175, 150],[176, 150],[177, 151],
            [178, 152],[179, 152],[180, 153],[181, 154],[182, 154],[183, 155],[184, 156],[185, 156],[186, 157],[187, 158],[188, 159],[189, 159],
            [190, 160],[191, 161],[192, 161],[193, 162],[194, 163],[195, 163],[196, 164],[197, 165],[198, 165],[199, 166],[200, 167],[201, 167],[202, 168],
            [203, 169],[204, 170],[205, 170],[206, 171],[207, 172],[208, 172],[209, 173],[210, 174],[211, 174],[212, 175],[213, 176],[214, 176],[215, 177],
            [216, 178],[217, 179],[218, 179],[219, 180],[220, 181],[221, 181],[222, 182]],
            dtype=np.int32
        )
        np.testing.assert_array_equal(line_coords, exp_line_coords)

        exp_fit = {
            'amplitude': 18135.996082935464,
            'fwhm': 39.27577937146833,
            'peak_center': 69.69494166725066
        }
        np.testing.assert_allclose(fit['amplitude'], exp_fit['amplitude'], rtol=1e5)
        np.testing.assert_allclose(fit['fwhm'], exp_fit['fwhm'], rtol=1e5)
        np.testing.assert_allclose(fit['peak_center'], exp_fit['peak_center'], rtol=1e5)
        
        exp_estimate = {
            'length': 47.35285870356088,
            'center': 84.02773357871772
        }
        np.testing.assert_allclose(estimate['length'], exp_estimate['length'], rtol=1e5)
        np.testing.assert_allclose(estimate['center'], exp_estimate['center'], rtol=1e5)

    def test_decompress_isxc_movie(self):
        input_compressed_file = test_data_path + "/unit_test/compressed/2022-05-12-21-28-30_video_DR_5_OQ_5.isxc"
        output_dir = test_data_path + "/unit_test/output/"

        isx.decompress(input_compressed_file, output_dir)

        expected_movie_path = test_data_path + '/unit_test/compressed/2022-05-12-21-28-30_video_DR_5_OQ_5-decompressed.isxd'
        actual_movie_path = output_dir + '2022-05-12-21-28-30_video_DR_5_OQ_5-decompressed.isxd'
        expected_movie = isx.Movie.read(expected_movie_path)
        actual_movie = isx.Movie.read(actual_movie_path)

        assert_isxd_movies_are_close(expected_movie, actual_movie)

        del expected_movie
        del actual_movie

        delete_files_silently([actual_movie_path])

    @pytest.mark.tiff_movie
    @pytest.mark.csv_trace
    def test_CellSetExporter(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_cell_set = unit_test_dir + '/eventDetectionCellSet.isxd'
        output_dir = unit_test_dir + '/output'
        output_trace = output_dir + '/trace_output.csv'
        output_image = output_dir + '/image_output.tiff'
        output_images = [output_dir + '/image_output_C{}.tiff'.format(i) for i in range(3)]

        delete_files_silently([output_trace] + output_images)

        isx.export_cell_set_to_csv_tiff([input_cell_set], output_trace, output_image, 'start')

        expected_base_path = unit_test_dir + '/guilded/exp_mosaicCellSetExporter_'
        assert_csv_traces_are_close_by_path(expected_base_path + 'TraceOutput.csv', output_trace)
        expected_images = [expected_base_path + 'ImageOutput-v2_C{}.tiff'.format(i) for i in range(3)]
        for exp, act in zip(expected_images, output_images):
            assert_tiff_files_equal_by_path(exp, act)

        assert not os.path.exists(output_dir + '/trace_output-props.csv')

        delete_files_silently([output_trace] + output_images)

    @pytest.mark.csv_trace
    def test_CellSetExporterWithProps(self):
        unit_test_dir = test_data_path + '/unit_test'
        test_dir = unit_test_dir + '/cellset_exporter'
        input_cell_sets = ['{}/50fr10_l{}-3cells_he-ROI-LCR.isxd'.format(test_dir, i) for i in range(1, 4)]
        output_dir = unit_test_dir + '/output'
        output_props = output_dir + '/properties.csv'

        delete_files_silently([output_props])

        isx.export_cell_set_to_csv_tiff(input_cell_sets, '', '', 'start', output_props_file=output_props)

        exp_props = unit_test_dir + '/guilded/exp_CellSetExporterWithProps-v2.csv'
        assert_csv_files_are_equal_by_path(exp_props, output_props)

        delete_files_silently([output_props])

    @pytest.mark.tiff_movie
    @pytest.mark.csv_trace
    def test_VesselSetExporter(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_vessel_set = unit_test_dir + '/bloodflow/rbcv_movie_1-RBCV_microns.isxd'
        output_dir = unit_test_dir + '/output'
        output_trace = output_dir + '/trace_output.csv'
        output_line = output_dir + '/line_output.csv'
        output_map = output_dir + '/map_output.tiff'
        output_heatmaps = output_dir + 'heatmaps'

        delete_files_silently([output_trace, output_line, output_map])
        delete_dirs_silently([output_heatmaps])
        os.makedirs(output_heatmaps)

        isx.export_vessel_set_to_csv_tiff([input_vessel_set],
            output_trace_csv_file=output_trace,
            output_line_csv_file=output_line,
            output_map_tiff_file=output_map,
            output_heatmaps_tiff_dir=output_heatmaps,
            time_ref='start')

        exp_trace_data = [
            'Vessel ID,Vessel Status,Index,Time (s),Velocity (um/s),Direction (deg),Clipping Error,Direction Change Error,No Significant Pixels Error,Invalid Frame Error\n',
            'V0,undecided,0,1,2800.791,248.087,False,False,False,False\n',
            'V0,undecided,1,2,3007.701,247.697,False,False,False,False\n',
            'V0,undecided,2,3,3026.005,247.423,False,False,False,False\n',
            'V0,undecided,3,4,3320.473,247.733,False,False,False,False\n',
            'V0,undecided,4,5,2187.080,246.729,False,False,False,False\n',
            'V0,undecided,5,6,nan,nan,False,False,False,True\n',
            'V1,undecided,0,1,640.503,112.884,False,False,False,False\n',
            'V1,undecided,1,2,587.908,113.486,False,False,False,False\n',
            'V1,undecided,2,3,611.888,114.251,False,False,False,False\n',
            'V1,undecided,3,4,610.198,113.903,False,False,False,False\n',
            'V1,undecided,4,5,539.932,112.015,False,False,False,False\n',
            'V1,undecided,5,6,nan,nan,False,False,False,True\n'
        ]
        with open(output_trace, 'r') as f:
            trace_data = f.read()
        assert trace_data == ''.join(exp_trace_data)

        exp_line_data = [
            'Name,Status,ColorR,ColorG,ColorB,PointX0,PointY0,PointX1,PointY1,PointX2,PointY2,PointX3,PointY3,Max Velocity(um/s)\n',
            'V0,undecided,255,255,255,124,25,153,36,90,202,61,191,6882.64\n',
            'V1,undecided,255,255,255,24,42,43,34,85,148,65,156,4709.46\n'
        ]
        with open(output_line, 'r') as f:
            line_data = f.read()
        assert line_data == ''.join(exp_line_data)

        assert os.path.exists(output_map)
        
        num_vessels = 2
        assert len(os.listdir(output_heatmaps)) == num_vessels

        delete_files_silently([output_trace, output_line, output_map])
        delete_dirs_silently([output_heatmaps])

    
    @pytest.mark.tiff_movie
    @pytest.mark.csv_trace
    def test_VesselSetExporterNoOutputs(self):
        unit_test_dir = test_data_path + '/unit_test'
        input_vessel_set = unit_test_dir + '/bloodflow/rbcv_movie_1-RBCV_microns.isxd'

        with pytest.raises(ValueError) as e:
            isx.export_vessel_set_to_csv_tiff([input_vessel_set])

    @pytest.mark.json_cell_contours
    def test_ExportCellContours(self):
        cell_set_file = test_data_path + '/unit_test/cell_metrics/cell_metrics_movie-PCA-ICA.isxd'

        output = test_data_path + '/unit_test/output/cell_contours_3cells_python.json'
        expected = test_data_path + '/unit_test/cell_metrics/expected_cell_contours_3cells_python-v2.json'

        delete_files_silently([output])

        isx.export_cell_contours(cell_set_file, output, threshold=0.0, rectify_first=True)

        assert_json_files_equal_by_path(expected, output)

        delete_files_silently([output])
