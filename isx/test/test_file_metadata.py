from .utilities.setup import delete_files_silently, test_data_path

import pytest

import isx

data_expected = {'Acquisition SW Version': '1.2.0',
                 'Animal Date of Birth': '2018, early foggy morning in the mid November',
                 'Animal Description': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.',
                 'Animal ID': 'Algernon XII Junior',
                 'Animal Sex': 'm',
                 'Animal Species': 'Lorem ipsum dolor sit amet',
                 'Animal Weight': 100500,
                 'Experimenter Name': 'John James "Jimmy" O\'Grady',
                 'Exposure Time (ms)': 17,
                 'Microscope EX LED Power (mw/mm^2)': 0.7,
                 'Microscope Focus': 500,
                 'Microscope Gain': 1,
                 'Microscope OG LED Power (mw/mm^2)': 5,
                 'Microscope Serial Number': 11094105,
                 'Microscope Type': 'NVoke2',
                 'Probe Diameter (mm)': 0.5,
                 'Probe Flip': 'none',
                 'Probe Length (mm)': 8.4,
                 'Probe Pitch': 2,
                 'Probe Rotation (degrees)': 0,
                 'Probe Type': 'Straight Lens',
                 'Session Name': 'Session 20181120-182836'
                 }
input_file = test_data_path + r'/unit_test/hub/2018-11-20-18-35-03/' + '2018-11-20-18-35-03_video.isxd'
output_dir = test_data_path + r'/unit_test/output'


class TestNV3FileMetadataMovie:
    def test_nv3_isxd_movie_captured(self):
        metadata = isx.Movie.read(input_file).get_acquisition_info()
        assert metadata == data_expected

    @pytest.mark.skipif(not isx._is_with_algos, reason="Only for algo tests")
    @pytest.mark.parametrize('algo', [] if not isx._is_with_algos else [isx.preprocess, isx.spatial_filter, isx.dff])
    def test_nv3_isxd_movie_processed(self, algo):
        output_file = output_dir + r'/' + algo.__name__ + '.isxd'
        delete_files_silently(output_file)

        algo(input_file, output_file)
        metadata = isx.Movie.read(output_file).get_acquisition_info()
        assert metadata == data_expected
        delete_files_silently(output_file)

    @pytest.mark.skipif(not isx._is_with_algos, reason="Only for algo tests")
    def test_nv3_isxd_movie_processed_mc(self):
        output_file = output_dir + r'/' + 'motion-correct' + '.isxd'
        delete_files_silently(output_file)

        data_expected['Motion correction padding'] = False

        isx.motion_correct(input_file, output_file)
        metadata = isx.Movie.read(output_file).get_acquisition_info()
        assert metadata == data_expected
        delete_files_silently(output_file)
        del data_expected['Motion correction padding']

    @pytest.mark.skipif(not isx._is_with_algos, reason="Only for algo tests")
    def test_nv3_isxd_cellset_pcaica(self):
        output_file = output_dir + r'/' + 'pca-ica' + '.isxd'
        delete_files_silently(output_file)

        data_expected['Cell Identification Method'] = 'pca-ica'
        data_expected['Trace Units'] = 'dF over F'

        isx.pca_ica(input_file, output_file, 8, 7)
        metadata = isx.CellSet.read(output_file).get_acquisition_info()
        assert metadata == data_expected
        delete_files_silently(output_file)

    @pytest.mark.skipif(not isx._is_with_algos, reason="Only for algo tests")
    def test_nv3_isxd_eventset(self):
        cellset_path = output_dir + r'/' + 'pca-ica-for-ed' + '.isxd'
        delete_files_silently(cellset_path)

        data_expected['Cell Identification Method'] = 'pca-ica'
        data_expected['Trace Units'] = 'dF over F'

        isx.pca_ica(input_file, cellset_path, 8, 7)
        output_eventset = output_dir + r'/' + 'event_detection' + '.isxd'
        delete_files_silently(output_eventset)
        isx.event_detection(cellset_path, output_eventset)
        metadata = isx.EventSet.read(output_eventset).get_acquisition_info()
        assert metadata == data_expected
        delete_files_silently(cellset_path)
        delete_files_silently(output_eventset)
