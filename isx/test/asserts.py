import json

import h5py
import numpy as np
import pandas
import tifffile as tf
import scipy.spatial.distance

import isx


def compare_h5_attrs(actual, expected):
    for k in expected.keys():
        if isinstance(expected[k], np.ndarray):
            assert (actual[k] == expected[k]).all()
        else:
            if not str(expected[k]) == 'nan':
                assert actual[k] == expected[k]


def compare_h5_datasets(actual, expected):
    assert actual.shape == expected.shape
    assert actual.size == expected.size
    assert actual.dtype == expected.dtype
    compare_h5_attrs(actual.attrs, expected.attrs)
    if actual.shape == ():
        assert expected.shape == ()
    else:
        assert (actual[:] == expected[:]).all()


def compare_h5_groups(actual, expected, keys_to_skip):
    for k in expected.keys():
        if k in keys_to_skip:
            continue
        else:
            compare_h5_attrs(actual[k].attrs, expected[k].attrs)
            if isinstance(expected[k], h5py.Dataset):
                compare_h5_datasets(actual[k], expected[k])
            elif isinstance(expected[k], h5py.Group):
                compare_h5_groups(actual[k], expected[k], keys_to_skip)
            else:
                assert False


def assert_isxd_movies_are_close(exp_movie, act_movie, relative_tolerance=1e-05):
    assert exp_movie.data_type == act_movie.data_type
    assert exp_movie.spacing == act_movie.spacing
    assert exp_movie.timing == act_movie.timing
    for f in range(act_movie.timing.num_samples):
        exp_frame = exp_movie.get_frame_data(f)
        act_frame = act_movie.get_frame_data(f)
        np.testing.assert_allclose(exp_frame, act_frame, rtol=relative_tolerance)


def assert_isxd_movies_are_close_by_path(exp_movie_path, act_movie_path, relative_tolerance=1e-05):
    exp_movie = isx.Movie.read(exp_movie_path)
    assert exp_movie.file_path == exp_movie_path
    act_movie = isx.Movie.read(act_movie_path)
    assert act_movie.file_path == act_movie_path

    assert exp_movie.data_type == act_movie.data_type
    assert exp_movie.spacing == act_movie.spacing
    assert exp_movie.timing == act_movie.timing
    for f in range(act_movie.timing.num_samples):
        exp_frame = exp_movie.get_frame_data(f)
        act_frame = act_movie.get_frame_data(f)
        np.testing.assert_allclose(exp_frame, act_frame, rtol=relative_tolerance)


def assert_isxd_movies_are_close_range_by_path(exp_movie_path, act_movie_path,
                                               x_range, y_range, relative_tolerance=1e-05):
    exp_movie = isx.Movie.read(exp_movie_path)
    assert exp_movie.file_path == exp_movie_path
    act_movie = isx.Movie.read(act_movie_path)
    assert act_movie.file_path == act_movie_path

    assert exp_movie.data_type == act_movie.data_type
    assert exp_movie.spacing == act_movie.spacing
    assert exp_movie.timing == act_movie.timing
    for f in range(act_movie.timing.num_samples):
        exp_frame = exp_movie.get_frame_data(f)
        act_frame = act_movie.get_frame_data(f)
        np.testing.assert_allclose(exp_frame[slice(*y_range), slice(*x_range)],
                                   act_frame[slice(*y_range), slice(*x_range)],
                                   rtol=relative_tolerance)


def assert_isxd_images_are_close_by_path_nan_zero(exp_image_path, act_image_path, relative_tolerance=1e-05):
    exp_image = isx.Image.read(exp_image_path)
    assert exp_image.file_path == exp_image_path
    act_image = isx.Image.read(act_image_path)
    assert act_image.file_path == act_image_path

    assert exp_image.data_type == act_image.data_type
    assert exp_image.spacing == act_image.spacing

    exp_data = np.nan_to_num(exp_image.get_data())
    act_data = np.nan_to_num(act_image.get_data())
    np.testing.assert_allclose(exp_data, act_data, rtol=relative_tolerance)


def assert_isxd_cellsets_are_close_by_path(exp_cellset_path, act_cellset_path, relative_tolerance=1e-05, assert_spacing=True, assert_status=True, use_cosine=False):
    exp_cellset = isx.CellSet.read(exp_cellset_path)
    assert exp_cellset.file_path == exp_cellset_path
    act_cellset = isx.CellSet.read(act_cellset_path)
    assert act_cellset.file_path == act_cellset_path

    assert exp_cellset.num_cells == act_cellset.num_cells
    if assert_spacing:
        assert exp_cellset.spacing == act_cellset.spacing
    assert exp_cellset.timing == act_cellset.timing
    for c in range(exp_cellset.num_cells):
        if assert_status:
            assert exp_cellset.get_cell_status(c) == act_cellset.get_cell_status(c)

        if use_cosine:
            if np.linalg.norm(exp_cellset.get_cell_image_data(c).flatten()) == 0 or np.linalg.norm(act_cellset.get_cell_image_data(c).flatten()) == 0:
                assert np.linalg.norm(exp_cellset.get_cell_image_data(c).flatten()) == np.linalg.norm(act_cellset.get_cell_image_data(c).flatten())
            else:
                assert (1.0 - scipy.spatial.distance.cosine(exp_cellset.get_cell_image_data(c).flatten(), act_cellset.get_cell_image_data(c).flatten())) >= relative_tolerance
            if np.linalg.norm(exp_cellset.get_cell_trace_data(c)) == 0 or np.linalg.norm(act_cellset.get_cell_trace_data(c)) == 0:
                assert np.linalg.norm(exp_cellset.get_cell_trace_data(c)) == np.linalg.norm(act_cellset.get_cell_trace_data(c))
            else:
                assert (1.0 - scipy.spatial.distance.cosine(exp_cellset.get_cell_trace_data(c), act_cellset.get_cell_trace_data(c))) >= relative_tolerance
        else:
            np.testing.assert_allclose(exp_cellset.get_cell_image_data(c), act_cellset.get_cell_image_data(c), rtol=relative_tolerance)
            np.testing.assert_allclose(exp_cellset.get_cell_trace_data(c), act_cellset.get_cell_trace_data(c), rtol=relative_tolerance)

def assert_isxd_cellsets_trace_sums(output_cell_set_files, expected_trace_sums):
    cell_sets = [isx.CellSet.read(f) for f in output_cell_set_files]
    num_cells = cell_sets[0].num_cells
    for i in range(num_cells):
        trace_sum = 0
        for cell_set in cell_sets:
            trace = cell_set.get_cell_trace_data(i)
            trace_sum += np.sum(trace, dtype=float)
        
        assert round(trace_sum) == expected_trace_sums[i]

def assert_isxd_cellsets_cell_names(output_cell_set_files, cell_names):
    cell_sets = [isx.CellSet.read(f) for f in output_cell_set_files]
    num_cells = cell_sets[0].num_cells
    for i in range(num_cells):
        for cell_set in cell_sets:
            assert cell_set.get_cell_name(i) == cell_names[i]

def assert_isxd_vesselsets_are_close_by_path(exp_vesselset_path, act_vesselset_path, relative_tolerance=1e-05, assert_spacing=True, assert_status=True):
    exp_vesselset = isx.VesselSet.read(exp_vesselset_path)
    assert exp_vesselset.file_path == exp_vesselset_path
    act_vesselset = isx.VesselSet.read(act_vesselset_path)
    assert act_vesselset.file_path == act_vesselset_path

    assert exp_vesselset.num_vessels == act_vesselset.num_vessels
    if assert_spacing:
        assert exp_vesselset.spacing == act_vesselset.spacing

    vessel_type = exp_vesselset.get_vessel_set_type()
    assert vessel_type == act_vesselset.get_vessel_set_type()
    assert exp_vesselset.has_correlation_heatmaps() == act_vesselset.has_correlation_heatmaps()

    assert exp_vesselset.timing == act_vesselset.timing
    for c in range(exp_vesselset.num_vessels):
        if assert_status:
            assert exp_vesselset.get_vessel_status(c) == act_vesselset.get_vessel_status(c)

        assert exp_vesselset.get_vessel_name(c) == act_vesselset.get_vessel_name(c)

        np.testing.assert_allclose(exp_vesselset.get_vessel_image_data(c), act_vesselset.get_vessel_image_data(c), rtol=relative_tolerance)
        np.testing.assert_allclose(exp_vesselset.get_vessel_trace_data(c), act_vesselset.get_vessel_trace_data(c), rtol=relative_tolerance)

        if vessel_type == isx.VesselSet.VesselSetType.VESSEL_DIAMETER:
            np.testing.assert_allclose(exp_vesselset.get_vessel_center_trace_data(c), act_vesselset.get_vessel_center_trace_data(c), rtol=relative_tolerance)
        else:
            np.testing.assert_allclose(exp_vesselset.get_vessel_direction_trace_data(c), act_vesselset.get_vessel_direction_trace_data(c), rtol=relative_tolerance)

            if exp_vesselset.has_correlation_heatmaps():
                for t in range(exp_vesselset.timing.num_samples):
                    np.testing.assert_allclose(exp_vesselset.get_vessel_correlations_data(c, t), act_vesselset.get_vessel_correlations_data(c, t), rtol=relative_tolerance)


def assert_isxd_vesselsets_vessel_names(output_vessel_set_files, vessel_names):
    vessel_sets = [isx.VesselSet.read(f) for f in output_vessel_set_files]
    num_vessels = vessel_sets[0].num_vessels
    for i in range(num_vessels):
        for vessel_set in vessel_sets:
            assert vessel_set.get_vessel_name(i) == vessel_names[i]

def assert_isxd_cellsets_are_close_range_by_path(exp_cellset_path, act_cellset_path,
                                                 x_range, y_range, relative_tolerance=1e-05,
                                                 absolute_tolerance=0):
    exp_cellset = isx.CellSet.read(exp_cellset_path)
    assert exp_cellset.file_path == exp_cellset_path
    act_cellset = isx.CellSet.read(act_cellset_path)
    assert act_cellset.file_path == act_cellset_path

    assert exp_cellset.num_cells == act_cellset.num_cells
    assert exp_cellset.spacing == act_cellset.spacing
    assert exp_cellset.timing == act_cellset.timing
    for c in range(exp_cellset.num_cells):
        assert exp_cellset.get_cell_status(c) == act_cellset.get_cell_status(c)
        np.testing.assert_allclose(exp_cellset.get_cell_image_data(c)[slice(*y_range), slice(*x_range)],
                                   act_cellset.get_cell_image_data(c)[slice(*y_range), slice(*x_range)],
                                   rtol=relative_tolerance, atol=absolute_tolerance)
        np.testing.assert_allclose(exp_cellset.get_cell_trace_data(c), act_cellset.get_cell_trace_data(c),
                                   rtol=relative_tolerance)


def assert_isxd_event_sets_are_close_by_path(exp_set_path, act_set_path, relative_tolerance=1e-05, use_cosine=False):
    exp_events = isx.EventSet.read(exp_set_path)
    assert exp_events.file_path == exp_set_path
    act_events = isx.EventSet.read(act_set_path)
    assert act_events.file_path == act_set_path

    assert exp_events.num_cells == act_events.num_cells
    assert exp_events.timing == act_events.timing
    for c in range(exp_events.num_cells):
        assert act_events.get_cell_name(c) == exp_events.get_cell_name(c)
        exp_cell_time, exp_cell_data = exp_events.get_cell_data(c)
        act_cell_time, act_cell_data = act_events.get_cell_data(c)
        
        if use_cosine:
            assert (1.0 - scipy.spatial.distance.cosine(exp_cell_time, act_cell_time)) >= relative_tolerance
            assert (1.0 - scipy.spatial.distance.cosine(exp_cell_data, act_cell_data)) >= relative_tolerance
        else:
            np.testing.assert_allclose(exp_cell_time, act_cell_time, rtol=relative_tolerance)
            np.testing.assert_allclose(exp_cell_data, act_cell_data, rtol=relative_tolerance)


def assert_tiff_files_equal_by_path(expected_file_path, tiff_file_path):
    exp_tiff = tf.imread(expected_file_path)
    actual_tiff = tf.imread(tiff_file_path)
    np.testing.assert_array_equal(exp_tiff, actual_tiff)


def assert_csv_traces_are_close_by_path(expected_csv_path, output_csv_path, relative_tolerance=1e-05):
    with open(expected_csv_path) as expected_csv, open(output_csv_path) as output_csv:
        expected_data = pandas.read_csv(expected_csv, header=[0, 1])
        output_data = pandas.read_csv(output_csv, header=[0, 1])
        np.testing.assert_allclose(expected_data, output_data, rtol=relative_tolerance)


def assert_csv_events_are_equal_by_path(expected_csv_path, output_csv_path):
    assert_csv_files_are_equal_by_path(expected_csv_path, output_csv_path)


def assert_csv_files_are_equal_by_path(expected_csv_path, output_csv_path):
    with open(expected_csv_path) as expected_csv, open(output_csv_path) as output_csv:
        expected_data = pandas.read_csv(expected_csv)
        output_data = pandas.read_csv(output_csv)
        np.testing.assert_array_equal(expected_data, output_data)
        assert expected_data.columns.tolist() == output_data.columns.tolist()


def assert_csv_files_are_close_by_path(expected_csv_path, output_csv_path, relative_tolerance=1e-05):
    with open(expected_csv_path) as expected_csv, open(output_csv_path) as output_csv:
        expected_data = pandas.read_csv(expected_csv)
        output_data = pandas.read_csv(output_csv)
        np.testing.assert_allclose(expected_data, output_data, rtol=relative_tolerance)


def assert_csv_cell_metrics_are_close_by_path(expected_csv_path, output_csv_path, relative_tolerance=1e-05):
    with open(expected_csv_path) as expected_csv, open(output_csv_path) as output_csv:
        expected_data = pandas.read_csv(expected_csv, header=0)
        output_data = pandas.read_csv(output_csv, header=0)
        np.testing.assert_array_equal(expected_data['cellName'], output_data['cellName'])

        exp_num_data, out_num_data = [df.drop('cellName', axis=1) for df in [expected_data, output_data]]
        np.testing.assert_allclose(exp_num_data, out_num_data, rtol=relative_tolerance)


def assert_json_files_equal_by_path(expected, output):
    with open(expected) as expected_file, open(output) as output_file:
        expected_json = json.load(expected_file)
        output_json = json.load(output_file)
        assert expected_json == output_json

def assert_txt_files_are_equal_by_path(expected, output):
        with open(expected) as expected_txt, open(output) as output_txt:
            for expected_line, output_line in zip(expected_txt, output_txt):
                assert expected_line == output_line

def assert_csv_pairwise_spatial_overlap_matrices_are_close_by_path(expected_csv_path, output_csv_path, relative_tolerance=1e-5):
    with open(expected_csv_path) as expected_csv, open(output_csv_path) as output_csv:
        expected_data = pandas.read_csv(expected_csv)
        output_data = pandas.read_csv(output_csv)
        assert expected_data.columns.tolist() == output_data.columns.tolist()
        expected_scores = np.array(expected_data.values[:, 1:], dtype=float)
        output_scores = np.array(output_data.values[:, 1:], dtype=float)
        np.testing.assert_allclose(expected_scores, output_scores, rtol=relative_tolerance)
