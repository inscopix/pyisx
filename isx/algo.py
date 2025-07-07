"""
The algo module deals with running algorithms on movies,
cell sets, and event sets.
"""

import os
import ctypes

import numpy as np

import isx._internal


def preprocess(
        input_movie_files, output_movie_files,
        temporal_downsample_factor=1, spatial_downsample_factor=1,
        crop_rect=None, crop_rect_format="tlbr",
        fix_defective_pixels=True, trim_early_frames=True):
    """
    Preprocess movies, optionally spatially and temporally downsampling and cropping.

    For more details see :ref:`preprocessing`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths to the input movies.
    output_movie_files : list<str>
        The file paths to write the preprocessed output movies to.
        This must be the same length as input_movie_files.
    temporal_downsample_factor : int >= 1
        The factor that determines how much the movie is temporally downsampled.
    spatial_downsample_factor : int >= 1
        The factor that determines how much the movie is spatially downsampled.
    crop_rect : 4-tuple<int>
        A list of 4 values representing the coordinates of the area to crop.
        Can be represented as one of two formats, specified by `crop_rect_format`.
        If `crop_rect_format == "tlbr"`, then the coordinates represent
        the top-left and bottom-right corners of the area to crop:
        [top_left_y, top_left_x, bottom_right_y, bottom_right_x]. All coordinates are specfied relative to the
        top-left corner of the field of view. For example, to trim 10 pixels all around a field of view of size 100x50
        pixels, the cropping vertices would be specfied as [10, 10, 39, 89].
        If `crop_rect_format == "tlwh", then the coordinates represent
        the top-left corner, width, and height of the area to crop"
        [top_left_x, top_left_y, width, height]. The top-left corner is specfied relative to the
        top-left corner of the field of view. For example, to trim 10 pixels all around a field of view of size 100x50
        pixels, the cropping vertices would be specfied as [10, 10, 90, 40].
    crop_rect_format : {'tlbr', 'tlwh'}
        The format of `crop_rect`.
        The format 'tlbr' stands for: top-left, bottom-right -- the two corners of the area to crop.
        The format 'tlwh' stands for: top-left, width, height -- the top-left corner and the size of the area to crop. 
    fix_defective_pixels : bool
        If True, then check for defective pixels and correct them.
    trim_early_frames : bool
        If True, then remove early frames that are usually dark or dim.
    """
    crop_rect_formats = ("tlbr", "tlwh")
    if crop_rect_format not in crop_rect_formats:
        raise ValueError(f"Invalid crop rect format ({crop_rect_format}), must be one of the following: {crop_rect_formats}")

    if crop_rect is None:
        crop_rect = (-1, -1, -1, -1)
    elif crop_rect_format == "tlwh":
        # format crop rect as top-left, bottom-right
        top_left_x, top_left_y, width, height = crop_rect
        bottom_right_x, bottom_right_y = (top_left_x + width - 1), (top_left_y + height - 1)
        crop_rect = (top_left_y, top_left_x, bottom_right_y, bottom_right_x)

    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)
    isx._internal.c_api.isx_preprocess_movie(
            num_files, in_arr, out_arr, temporal_downsample_factor, spatial_downsample_factor,
            crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3], fix_defective_pixels, trim_early_frames)


def de_interleave(input_movie_files, output_movie_files, in_efocus_values):
    """
    De-interleave multiplane movies.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths to the input movies.
        All files should have the same efocus values and same number of planes.
    output_movie_files : list<str>
        The file paths to write the de-interleaved output movies to.
        This must be the length of input_movie_files * the number of planes.
        The sequence of every number of planes elements must match the sequence of efocus values.
        E.g: [in_1, in_2], [efocus1, efocus2] -> [out_1_efocus1, out_1_efocus2, out_2_efocus1, out_2_efocus2]
    in_efocus_values : list<int>
        The efocus value for each planes.
        This must in range 0 <= efocus <= 1000.
    """
    efocus_arr = isx._internal.list_to_ctypes_array(in_efocus_values, ctypes.c_uint16)
    num_planes = len(in_efocus_values)
    num_in_files, in_arr = isx._internal.check_input_files(input_movie_files)
    num_output_files, out_arr = isx._internal.check_input_files(output_movie_files)

    if num_output_files != num_in_files * num_planes:
        raise ValueError('Number of output files must match the number of input files times the number of planes.')

    isx._internal.c_api.isx_deinterleave_movie(num_in_files, num_planes, efocus_arr, in_arr, out_arr)


def motion_correct(
        input_movie_files, output_movie_files, max_translation=20,
        low_bandpass_cutoff=0.004, high_bandpass_cutoff=0.016, roi=None,
        reference_segment_index=0, reference_frame_index=0, reference_file_name='',
        global_registration_weight=1.0, output_translation_files=None,
        output_crop_rect_file=None, preserve_input_dimensions=False):
    """
    Motion correct movies to a reference frame.

    For more details see :ref:`motionCorrection`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to motion correct.
    output_movie_files : list<str>
        The file paths of the output movies.
        This must be the same length as input_movie_files.
    max_translation : int > 0
        The maximum translation allowed by motion correction in pixels.
    low_bandpass_cutoff : float > 0
        If not None, then the low cutoff of the spatial filter applied to each frame prior to motion estimation.
    high_bandpass_cutoff : float > 0
        If not None, then the high cutoff for a spatial filter applied to each frame prior to motion estimation.
    roi : Nx2 array-like
        If not None, each row is a vertex of the ROI to use for motion estimation.
        Otherwise, use the entire frame.
    reference_segment_index : int > 0
        If a reference frame is to be specified, this parameter indicates the index of the movie whose frame will
        be utilized, with respect to input_movie_files.
        If only one movie is specified to be motion corrected, this parameter must be 0.
    reference_frame_index : int > 0
        Use this parameter to specify the index of the reference frame to be used, with respect to reference_segment_index.
        If reference_file_name is specified, this parameter, as well as reference_segment_index, is ignored.
    reference_file_name : str
        If an external reference frame is to be used, this parameter should be set to path of the .isxd file
        that contains the reference image.
    global_registration_weight : 0.05 <= float <= 1
        When this is set to 1, only the reference frame is used for motion estimation.
        When this is less than 1, the previous frame is also used for motion estimation.
        The closer this value is to 0, the more the previous frame is used and the less
        the reference frame is used.
    output_translation_files : list<str>
        A list of file names to write the X and Y translations to.
        Must be either None, in which case no files are written, or a list of valid file names equal
        in length to the number of input and output file names.
        The output translations are written into a .csv file with three columns.
        The first two columns, "translationX" and "translationY", store the X and Y translations from
        each frame to the reference frame respectively.
        The third column contains the time of the frame since the beginning of the movie.
        The first row stores the column names as a header.
        Each subsequent row contains the X translation, Y translation, and time offset for that frame.
    output_crop_rect_file : str
        The path to a file that will contain the crop rectangle applied to the input movies to generate the output
        movies.
        The format of the crop rectangle is a comma separated list: x,y,width,height.
    preserve_input_dimensions: bool
        If true, the output movie will be padded along the edges to match the dimensions of the input movie.
        The padding value will be set to the 5th percentile of the pixel value distribution collected from 10 evenly sampled frames from the input movie.
    """
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)

    use_low = int(low_bandpass_cutoff is not None)
    use_high = int(high_bandpass_cutoff is not None)

    if use_low == 0:
        low_bandpass_cutoff = 0.0
    if use_high == 0:
        high_bandpass_cutoff = 1.0

    # The first two elements tell the C layer the number of ROIs, then the
    # number of vertices in the first ROI.
    if roi is not None:
        roi_np = isx._internal.convert_to_nx2_numpy_array(roi, int, 'roi')
        roi_arr = isx._internal.list_to_ctypes_array([1, roi_np.shape[0]] + list(roi_np.ravel()), ctypes.c_int)
    else:
        roi_arr = isx._internal.list_to_ctypes_array([0], ctypes.c_int)

    if reference_file_name is None:
        ref_file_name = ''
    else:
        ref_file_name = reference_file_name

    out_trans_arr = isx._internal.list_to_ctypes_array([''], ctypes.c_char_p)
    write_output_translations = int(output_translation_files is not None)
    if write_output_translations:
        out_trans_files = isx._internal.ensure_list(output_translation_files)
        assert len(out_trans_files) == num_files, "Number of output translation files must match number of input movies ({} != {})".format(len(out_trans_files), len(in_arr))
        out_trans_arr = isx._internal.list_to_ctypes_array(out_trans_files, ctypes.c_char_p)

    write_crop_rect = int(output_crop_rect_file is not None)
    if not write_crop_rect:
        output_crop_rect_file = ''

    isx._internal.c_api.isx_motion_correct_movie(
            num_files, in_arr, out_arr, max_translation,
            use_low, low_bandpass_cutoff, use_high, high_bandpass_cutoff,
            roi_arr, reference_segment_index, reference_frame_index,
            ref_file_name.encode('utf-8'), global_registration_weight,
            write_output_translations, out_trans_arr,
            write_crop_rect, output_crop_rect_file.encode('utf-8'), preserve_input_dimensions)


def pca_ica(
        input_movie_files, output_cell_set_files, num_pcs, num_ics=120, unmix_type='spatial',
        ica_temporal_weight=0, max_iterations=100, convergence_threshold=1e-5, block_size=1000,
        auto_estimate_num_ics=False, average_cell_diameter=13):
    """
    Run PCA-ICA cell identification on movies.

    For more details see :ref:`PCA_ICA`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to run PCA-ICA on.
    output_cell_set_files : list<str>
        The paths of the output cell set files. Must be same length as input_movie_files.
    num_pcs : int > 0
        The number of principal components (PCs) to estimate. Must be >= num_ics.
    num_ics : int > 0
        The number of independent components (ICs) to estimate.
    unmix_type : {'temporal', 'spatial', 'both'}
        The unmixing type or dimension.
    ica_temporal_weight : 0 <= float <= 1
        The temporal weighting factor used for ICA.
    max_iterations : int > 0
        The maximum number of iterations for ICA.
    convergence_threshold : float > 0
        The convergence threshold for ICA.
    block_size : int > 0
        The size of the blocks for the PCA step. The larger the block size, the more memory that will be used.
    auto_estimate_num_ics : bool
        If True the number of ICs will be automatically estimated during processing.
    average_cell_diameter : int > 0
        Average cell diameter in pixels (only used when auto_estimate_num_ics is set to True)
    Returns
    -------
    bool
        True if PCA-ICA converged, False otherwise.
    """
    unmix_type_int = isx._internal.lookup_enum('unmix_type', isx._internal.ICA_UNMIX_FROM_STRING, unmix_type)
    if ica_temporal_weight < 0 or ica_temporal_weight > 1:
        raise ValueError("ica_temporal_weight must be between zero and one")

    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_cell_set_files)
    converged = ctypes.c_int()
    isx._internal.c_api.isx_pca_ica_movie(
            num_files, in_arr, out_arr, num_pcs, num_ics, unmix_type_int, ica_temporal_weight,
            max_iterations, convergence_threshold, block_size, ctypes.byref(converged), 0,
            auto_estimate_num_ics, average_cell_diameter)

    return converged.value > 0

def run_cnmfe(
        input_movie_files, output_cell_set_files, output_dir='.',
        cell_diameter=7,
        min_corr=0.8, min_pnr=10, bg_spatial_subsampling=2, ring_size_factor=1.4,
        gaussian_kernel_size=0, closing_kernel_size=0, merge_threshold=0.7,
        processing_mode="parallel_patches", num_threads=4, patch_size=80, patch_overlap=20,
        output_unit_type="df_over_noise"):
    """
    Run CNMFe cell identification on movies.

    For more details see :ref:`CNMFe`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to run CNMFe on.
    output_cell_set_files : list<str>
        The paths of the deconvolved output cell set files. Must be same length as input_movie_files.
    output_dir : str
        Output directory for intermediary files (e.g., memory map files)
    cell_diameter: int > 0
        Expected average diameter of a neuron in pixels
    min_corr: float
        Minimum correlation with neighbours when searching for seed pixels
    min_pnr: float
        Minimum peak-to-noise ratio when searching for seed pixels
    bg_spatial_subsampling: int > 0 (1 for no downsampling)
        Background spatial downsampling factor
    ring_size_factor: float > 0
        Ratio of ring radius to neuron diameter used for estimating background
    gaussian_kernel_size: int >= 0 (0 for automatic estimation)
        Width of Gaussian kernel to use for spatial filtering
    closing_kernel_size: int >= 0 (0 for automatic estimation)
        Morphological closing kernel size
    merge_threshold: float
        Temporal correlation threshold for merging spatially close cells
    processing_mode: string in {'all_in_memory', 'sequential_patches', 'parallel_patches'}
        Processing mode for Cnmfe
    num_threads: int > 0
        Number of threads to use for processing the data
    patch_size: int > 1
        Size of a single patch
    patch_overlap: int >= 0
        Amount of overlap between patches in pixels
    output_unit_type: string in {'df', 'df_over_noise'}
        Output trace units for temporal components
    """
    processing_mode_map = {'all_in_memory':0, 'sequential_patches':1, 'parallel_patches':2}
    output_unit_type_map = {'df' : 0, 'df_over_noise' : 1}

    num_cell_files, in_movie_arr1, out_cell_arr = isx._internal.check_input_and_output_files(input_movie_files, output_cell_set_files)

    isx._internal.c_api.isx_cnmfe_movie(
        num_cell_files, in_movie_arr1, out_cell_arr, output_dir.encode('utf-8'),
        cell_diameter,
        min_corr, min_pnr, bg_spatial_subsampling, ring_size_factor,
        gaussian_kernel_size, closing_kernel_size, merge_threshold,
        processing_mode_map[processing_mode], num_threads, patch_size, patch_overlap,
        output_unit_type_map[output_unit_type])

def spatial_filter(
        input_movie_files, output_movie_files, low_cutoff=0.005, high_cutoff=0.500,
        retain_mean=False, subtract_global_minimum=True):
    """
    Apply spatial bandpass filtering to each frame of one or more movies.

    For more details see :ref:`spatialBandpassFilter`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to filter.
    output_movie_files : list<str>
        The file paths of the output movies. Must be the same length as input_movie_files.
    low_cutoff : float > 0
        If not None, then the low cutoff for the spatial filter.
    high_cutoff : float > 0
        If not None, then the high cutoff for the spatial filter.
    retain_mean : bool
        If True, retain the mean pixel intensity for each frame (the DC component).
    subtract_global_minimum : bool
        If True, compute the minimum pixel intensity across all movies, and subtract this
        after frame-by-frame mean subtraction.
        By doing this, all pixel intensities will stay positive valued, and integer-valued
        movies can stay that way.
    """
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)
    use_low = int(low_cutoff is not None)
    use_high = int(high_cutoff is not None)
    low_cutoff = low_cutoff if use_low else 0
    high_cutoff = high_cutoff if use_high else 0
    isx._internal.c_api.isx_spatial_band_pass_movie(
            num_files, in_arr, out_arr, use_low, low_cutoff, use_high, high_cutoff,
            int(retain_mean), int(subtract_global_minimum))


def dff(input_movie_files, output_movie_files, f0_type='mean'):
    """
    Compute DF/F movies, where each output pixel value represents a relative change
    from a baseline.

    For more details see :ref:`DFF`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the input movies.
    output_movie_files : list<str>
        The file paths of the output movies.
    f0_type : {'mean', 'min}
        The reference image or baseline image used to compute DF/F.
    """
    f0_type_int = isx._internal.lookup_enum('f0_type', isx._internal.DFF_F0_FROM_STRING, f0_type)
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)
    isx._internal.c_api.isx_delta_f_over_f(num_files, in_arr, out_arr, f0_type_int)


def project_movie(input_movie_files, output_image_file, stat_type='mean'):
    """
    Project movies to a single statistic image.

    For more details see :ref:`movieProjection`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to project.
    output_image_file : str
        The file path of the output image.
    stat_type: {'mean', 'min', 'max', 'standard_deviation'}
        The type of statistic to compute.
    """
    stat_type_int = isx._internal.lookup_enum('stat_type', isx._internal.PROJECTION_FROM_STRING, stat_type)
    num_files, in_arr = isx._internal.check_input_files(input_movie_files)
    isx._internal.c_api.isx_project_movie(num_files, in_arr, output_image_file.encode('utf-8'), stat_type_int)


def event_detection(
        input_cell_set_files, output_event_set_files, threshold=5, tau=0.2,
        event_time_ref='beginning', ignore_negative_transients=True, accepted_cells_only=False):
    """
    Perform event detection on cell sets.

    For more details see :ref:`eventDetection`.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell sets to perform event detection on.
    output_event_set_files : list<str>
        The file paths of the output event sets.
    threshold : float > 0
        The threshold in median-absolute-deviations that the trace has to cross to be considered an event.
    tau : float > 0
        The minimum time in seconds that an event has to last in order to be considered.
    event_time_ref : {'maximum', 'beginning', 'mid_rise'}
        The temporal reference that defines the event time.
    ignore_negative_transients : bool
        Whether or not to ignore negative events.
    accepted_cells_only : bool
        If True, detect events only for accepted cells.
    """
    event_time_ref_int = isx._internal.lookup_enum('event_time_ref', isx._internal.EVENT_REF_FROM_STRING, event_time_ref)
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_cell_set_files, output_event_set_files)
    isx._internal.c_api.isx_event_detection(
            num_files, in_arr, out_arr, threshold, tau, event_time_ref_int,
            int(ignore_negative_transients), int(accepted_cells_only))


def trim_movie(input_movie_file, output_movie_file, crop_segments, keep_start_time=False):
    """
    Trim frames from a movie to produce a new movie.

    For more details see :ref:`trimMovie`.

    Arguments
    ---------
    input_movie_file : str
        The file path of the movie.
    output_movie_file : str
        The file path of the trimmed movie.
    crop_segments : Nx2 array-like
        A numpy array of shape (num_segments, 2), where each row contains the start and
        end indices of frames that will be cropped out of the movie. Or a list like:
        [(start_index1, end_index1), (start_index2, end_index2), ...].
    keep_start_time : bool
        If true, keep the start time of the movie, even if some of its initial frames are to be trimmed.
    """
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_file, output_movie_file)
    if num_files != 1:
        raise TypeError("Only one movie can be specified.")

    crop_segs = isx._internal.convert_to_nx2_numpy_array(crop_segments, int, 'crop_segments')
    indices_arr = isx._internal.list_to_ctypes_array([crop_segs.shape[0]] + list(crop_segs.ravel()), ctypes.c_int)

    isx._internal.c_api.isx_temporal_crop_movie(1, in_arr, out_arr, indices_arr, keep_start_time)


def apply_cell_set(input_movie_files, input_cell_set_file, output_cell_set_files, threshold):
    """
    Apply the images of a cell set to movies, producing a new cell sets.

    For more details see :ref:`applyContours`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to apply the cell set to.
    input_cell_set_file : list<str>
        The file path of the cell set to apply.
    output_cell_set_files : list<str>
        The file paths of the output cell sets that will contain the images and new traces.
    threshold : 0 >= float >= 1
        A threshold that will be applied to each footprint prior to application.
        This indicates the fraction of the maximum image value that will be used as the
        absolute threshold.
    """
    num_movies, in_movie_arr, out_cs_arr = isx._internal.check_input_and_output_files(input_movie_files, output_cell_set_files)
    num_cs_in, in_cs_arr = isx._internal.check_input_files(input_cell_set_file)
    if num_cs_in != 1:
        raise TypeError("Only one input cell set can be specified.")
    isx._internal.c_api.isx_apply_cell_set(num_movies, in_movie_arr, out_cs_arr, in_cs_arr[0], threshold)


def apply_rois(
    input_movie_files,
    output_cell_set_files,
    rois,
    cell_names=[]
):
    """
    Apply manually drawn rois on movies, producing new cell sets.

    For more details see :ref:`manualRois`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to apply the rois to.
    output_cell_set_files : list<str>
        The file paths of the output cell sets that will contain the images and new traces.
    rois: list<list<tuple<int>>>
        List of rois to apply. Must be one or more rois.
        Each roi is a list of tuples of 2 integers representing the x, y coordinates of a single point.
    cell_names: list<str>
        List of names to assign cells in the output cell sets.
        If empty, then cells will have default names.
    """
    num_movies, in_movie_arr, out_cs_arr = isx._internal.check_input_and_output_files(input_movie_files, output_cell_set_files)
    
    # get cell names if not empty
    num_rois = len(rois)
    cell_names_arr = isx._internal.list_to_ctypes_array(cell_names, ctypes.c_char_p)
    use_cell_names = len(cell_names) > 0 
    if cell_names:
        if num_rois != len(cell_names):
            raise ValueError("Number of rois must equal number of cell names.")

    # get roi points
    if num_rois == 0:
        raise ValueError('At least one roi needs to be specified')
    
    # count the number of points per roi
    num_points_per_roi = []
    for roi in rois:
        num_points_per_roi.append(len(roi))
        for point in roi:
            if(len(point) != 2):
                raise ValueError('All points must only have two coordinates (x and y-coordinates, respectively)')

    # flatten the roi array so it can be passed to c_types
    rois = [
        x
        for xs in rois
        for x in xs
    ]
    
    # get pointer to points memory
    points = isx._internal.ndarray_as_type(np.array(rois), np.dtype(np.int64))
    points_p = points.ctypes.data_as(isx._internal.Int64Ptr)

    num_points_per_roi = isx._internal.ndarray_as_type(np.array(num_points_per_roi), np.dtype(np.int64))
    num_points_per_roi_p = num_points_per_roi.ctypes.data_as(isx._internal.Int64Ptr)

    isx._internal.c_api.isx_apply_rois(
        num_movies,
        in_movie_arr,
        out_cs_arr,
        num_rois,
        num_points_per_roi_p,
        points_p,
        use_cell_names,
        cell_names_arr
    )


def longitudinal_registration(
        input_cell_set_files, output_cell_set_files, input_movie_files=[], output_movie_files=[],
        csv_file='', min_correlation=0.5, accepted_cells_only=False,
        transform_csv_file='', crop_csv_file=''):
    """
    Run longitudinal registration on multiple cell sets.

    Optionally, also register the corresponding movies the cell sets were derived from.

    For more details see :ref:`LongitudinalRegistration`.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell sets to register.
    output_cell_set_files : list<str>
        The file paths of the output cell sets.
    input_movie_files : list<str>
        The file paths of the associated input movies (optional).
    output_movie_files: list<str>
        The file paths of the output movies (optional)
    csv_file : str
        The path of the output CSV file to be written (optional).
    min_correlation : 0 >= float >= 1
        The minimum correlation between cells to be considered a match.
    accepted_cells_only : bool
        Whether or not to use accepted cells from the input cell sets only, or to use both accepted and undecided cells.
    transform_csv_file : str
        The file path of the CSV file to store the affine transform parameters
        from the reference cellset to each cellset.
        Each row represents an input cell set and contains the values in the
        2x3 affine transform matrix in a row-wise order.
        I.e. if we use a_{i,j} to represent the values in the 2x2 upper left
        submatrix and t_{i} to represent the translations, the values are
        written in the order: a_{0,0}, a_{0,1}, t_{0}, a_{1,0}, a_{1,1}, t_{1}.
    crop_csv_file : str
        The file path of the CSV file to store the crop rectangle applied after
        transforming the cellsets and movies.
        The format of the crop rectangle is a comma separated list: x,y,width,height.
    """
    num_cell_files, in_cell_arr, out_cell_arr = isx._internal.check_input_and_output_files(input_cell_set_files, output_cell_set_files)
    num_movie_files, in_movie_arr, out_movie_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)
    if (num_movie_files > 0) and (num_movie_files != num_cell_files):
        raise ValueError("If specified, the number of movies must be the same as the number of cell sets.")
    isx._internal.c_api.isx_longitudinal_registration(num_cell_files, in_cell_arr, out_cell_arr, in_movie_arr, out_movie_arr, csv_file.encode('utf-8'), min_correlation, int(not accepted_cells_only), int(num_movie_files > 0), transform_csv_file.encode('utf-8'), crop_csv_file.encode('utf-8'))


def auto_accept_reject(input_cell_set_files, input_event_set_files, filters=None):
    """
    Automatically classify cell statuses as accepted or rejected.

    For more details see :ref:`autoAcceptReject`.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell sets to classify.
    input_event_set_files : list<str>
        The file paths of the event sets to use for classification.
    filters : list<3-tuple>
        Each element describes a filter as (<statistic>, <operator>, <value>).
        The statistic must be one of {'# Comps', 'Cell Size', 'SNR', 'Event Rate'}.
        The operator must be one of {'<', '=', '>'}.
        The value is a floating point number.
    """
    num_cell_sets, in_cell_arr = isx._internal.check_input_files(input_cell_set_files)
    num_event_sets, in_event_arr = isx._internal.check_input_files(input_event_set_files)

    statistics = []
    operators = []
    values = []
    num_filters = 0
    if filters is not None:
        if isinstance(filters, list):
            statistics, operators, values = map(list, zip(*filters))
            num_filters = len(filters)
        else:
            raise TypeError('Filters must be contained in a list.')

    in_statistics = isx._internal.list_to_ctypes_array(statistics, ctypes.c_char_p)
    in_operators = isx._internal.list_to_ctypes_array(operators, ctypes.c_char_p)
    in_values = isx._internal.list_to_ctypes_array(values, ctypes.c_double)

    isx._internal.c_api.isx_classify_cell_status(
            num_cell_sets, in_cell_arr, num_event_sets, in_event_arr,
            num_filters, in_statistics, in_operators, in_values,
            0, isx._internal.SizeTPtr())


def cell_metrics(input_cell_set_files, input_event_set_files, output_metrics_file, recompute_metrics=True):
    """
    Compute cell metrics for a given cell set and events combination.

    For more details see :ref:`cellMetrics`.

    Arguments
    ---------
    input_cell_set_files : list<str>
        One or more input cell sets.
    input_event_set_files : list<str>
        One or more events files associated with the input cell sets.
    output_metrics_file : str
        One .csv file that will be written which contain cell metrics.
        If more than one input cell set & events file is passed, then the inputs are treated as a time-series.
    recompute_metrics : bool
        Flag indicating whether cell metrics should be recomputed from input files
        otherwise cell metrics stored in the input files are exported.
        If no cell metrics are stored in the input files and this flag is set to False,
        then this function will throw an error.
    """
    num_cs_in, in_cs_arr = isx._internal.check_input_files(input_cell_set_files)
    num_events_in, in_events_arr = isx._internal.check_input_files(input_event_set_files)
    if num_cs_in != num_events_in:
        raise TypeError("The number of cell sets and events must be the same.")
    isx._internal.c_api.isx_compute_cell_metrics(num_cs_in, in_cs_arr, in_events_arr, output_metrics_file.encode('utf-8'), recompute_metrics)


def export_cell_contours(input_cell_set_file, output_json_file, threshold=0.0, rectify_first=True):
    """
    Export cell contours to a JSON file.

    If a cell image has multiple components the contour for each component is exported in a separate array.

    These are the contours calculated from preprocessed cell images as described in :ref:`cellMetrics`.

    Arguments
    ---------
    input_movie_file : str
        The file path of a cell set.
    output_json_file : str
        The file path to the output JSON file to be written.
    threshold : 0 >= float >= 1
        The threshold to apply to the footprint before computing the contour, specified as a
        fraction of the maximum pixel intensity.
    rectify_first : bool
        Whether or not to rectify the image (remove negative components) prior to computing the threshold.
    """
    num_cs_in, in_cs_arr, out_js_arr = isx._internal.check_input_and_output_files(input_cell_set_file, output_json_file)
    if num_cs_in != 1:
        raise TypeError("Only one input cell set can be specified.")
    isx._internal.c_api.isx_export_cell_contours(num_cs_in, in_cs_arr, out_js_arr, threshold, int(rectify_first))


def multiplane_registration(
        input_cell_set_files,
        output_cell_set_file,
        min_spat_correlation=0.5,
        temp_correlation_thresh=0.99,
        accepted_cells_only=False):
    """
    Identify unique signals in 4D imaging data using longitudinal registration
    of spatial footprints and temporal correlation of activity.

    :param input_cell_set_files: (list <str>) the file paths of the cell sets from de-interleaved multiplane movies.
    :param output_cell_set_file: (str) the file path of the output cell set of multiplane registration.
    :param min_spat_correlation: (0 <= float <= 1) the minimum spatial overlap between cells to be considered a match.
    :param temp_correlation_thresh: (0 <= float <= 1) the percentile of the comparison distribution below which
                                    activity correlations are considered from distinct signals
    :param accepted_cells_only: (bool) whether or not to include only accepted cells from the input cell sets.
    """
    if not 0 <= min_spat_correlation <= 1:
        raise TypeError("Spatial correlation must be between 0 and 1.")
    if not 0 <= temp_correlation_thresh <= 1:
        raise TypeError("Temporal correlation threshold must be between 0 and 1.")
    num_cs_in, in_cs_arr, out_cs_arr = isx._internal.check_input_and_output_files(input_cell_set_files, output_cell_set_file, True)
    isx._internal.c_api.isx_multiplane_registration(
        num_cs_in,
        in_cs_arr,
        out_cs_arr,
        min_spat_correlation,
        temp_correlation_thresh,
        accepted_cells_only
    )

def estimate_num_ics(
        input_image_files,
        average_diameter = None,
        min_diameter = None,
        max_diameter = None,
        min_inter_dist = 0):
    """
    Estimates ICs parameter on a projection image of a movie to be run through PCA-ICA.
    Images should be DF/F projections for best results.
    Should either give the average diameter, or the min and max diameter of cells.

    :param input_image_files: (list <str>) the file paths of the df/f projection images
    :pstsm average_diameter: (0 < float) average diameter of cells in pixels
    :param min_diameter: (0 < float) minimum diameter of a cell in pixels
    :param max_diameter: (min_diameter < float) maximum diameter of a cell in pixels
    :param min_inter_dist: (0 <= float) minimum allowable distance between adjacent cells
    :return: (list <int>) number of estimated cells in each input image
    """
    if (average_diameter is None) and (min_diameter is None or max_diameter is None):
        raise ValueError("Either average diameter or min and max diameters should be given.")

    if min_diameter is not None and max_diameter is not None:
        if not 0 < min_diameter:
            raise ValueError("Minimum diameter should be positive.")
        if not min_diameter < max_diameter:
            raise ValueError("Maximum diameter should be greater than minimum diameter.")
    else:
        if not 0 < average_diameter:
            raise ValueError("Average diameter should be positive.")
        min_diameter = average_diameter / 3.
        max_diameter = average_diameter * (5. / 3.)

    num_img_in, in_img_arr = isx._internal.check_input_files(input_image_files)
    ic_count = (ctypes.c_size_t * num_img_in)()
    isx._internal.c_api.isx_estimate_num_ics(
        num_img_in,
        in_img_arr,
        min_diameter,
        max_diameter,
        min_inter_dist,
        ic_count
    )
    
    return list(ic_count) if num_img_in > 1 else ic_count[0]

def de_interleave_dualcolor(input_movie_files, output_green_movie_files, output_red_movie_files, correct_chromatic_shift=True):
    """
    De-interleave dual-color movies.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths to the input movies.
    output_green_movie_files : list<str>
        The file paths to write the de-interleaved green output movies to.
    output_red_movie_files : list<str>
        The file paths to write the de-interleaved red output movies to.
    correct_chromatic_shift : bool
        If true, correct chromatic shift of green and red channels.
        The correction will only be applied if the movie is wide-field (i.e., acquired from the LScape module).
    """
    num_in_files, in_arr = isx._internal.check_input_files(input_movie_files)
    num_output_files_green, out_arr_green = isx._internal.check_input_files(output_green_movie_files)
    num_output_files_red, out_arr_red = isx._internal.check_input_files(output_red_movie_files)

    if num_output_files_green != num_in_files:
        raise ValueError('Number of green output files must match the number of input files.')
    if num_output_files_red != num_in_files:
        raise ValueError('Number of red output files must match the number of input files.')

    isx._internal.c_api.isx_deinterleave_dualcolor_movie(num_in_files, in_arr, out_arr_green, out_arr_red, correct_chromatic_shift)

def multicolor_registration(
        input_cellset_file1,
        input_cellset_file2,
        output_spatial_overlap_csv_file,
        output_registration_matrix_csv_file,
        output_directory='.',
        lower_threshold=0.2,
        upper_threshold=0.5,
        accepted_cells_only=False,
        save_matched_cellset=True,
        save_unmatched_cellset=True,
        save_uncertain_cellset=True,
        image_format="tiff"):
    """
    Run multicolor registration on two cell sets.

    Arguments
    ---------
    input_cellset_file1 : str
        Path to the first .isxd cellset file.
    input_cellset_file2 : str
        Path to the second .isxd cellset file.
    output_spatial_overlap_csv_file : str
        Path to the .csv file containing the pairwise spatial overlap scores.
    output_registration_matrix_csv_file : str
        Path to the .csv file containing the registration matrix.
    output_directory : str
        Path to the output directory. Generated cellsets and images will be saved in this directory.
    lower_threshold : double
        Maximum score between two cells that can be rejected as a match.
    upper_threshold : double
        Minimum score between two cells that can be accepted as a match.
    accepted_cells_only : bool
        Whether or not to use accepted cells from the input cell sets only, or to use both accepted and undecided cells.
    save_matched_cellset : bool
        Whether or not to save the matched cells from the primary cellset to a cellset file.
    save_unmatched_cellset : bool
        Whether or not to save the unmatched cells from the primary cellset to a cellset file.
    save_uncertain_cellset : bool
        Whether or not to save the uncertain cells from the primary cellset to a cellset file.
    image_format : str in {"tiff", "png"}
        File format to use for the images to save
    """
    for input_file in [input_cellset_file1, input_cellset_file2]:
        if not os.path.exists(input_file):
            raise FileNotFoundError('Input file not found: {}'.format(input_file))

    for output_file in [output_spatial_overlap_csv_file, output_registration_matrix_csv_file]:
        if os.path.exists(output_file):
            raise FileExistsError('Output file already exists: {}'.format(output_file))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not 0 <= lower_threshold <= 1:
        raise TypeError("Lower threshold must be between 0 and 1.")
    if not 0 <= upper_threshold <= 1:
        raise TypeError("Upper threshold must be between 0 and 1.")
    if image_format not in ["tiff","png"]:
        raise TypeError("Image format must be either 'tiff' or 'png'.")

    isx._internal.c_api.isx_multicolor_registration(
        input_cellset_file1.encode('utf-8'),
        input_cellset_file2.encode('utf-8'),
        output_spatial_overlap_csv_file.encode('utf-8'),
        output_registration_matrix_csv_file.encode('utf-8'),
        output_directory.encode('utf-8'),
        lower_threshold,
        upper_threshold,
        accepted_cells_only,
        save_matched_cellset,
        save_unmatched_cellset,
        save_uncertain_cellset,
        image_format.encode('utf-8'))

def binarize_cell_set(input_cellset_file, output_cellset_file, threshold, use_percentile_threshold=False):
    """Apply a threshold to each footprint in a cell set to produce a new cell set with binary footprints.

    Arguments
    ---------
    input_cellset_file : str
        Path to the .isxd cellset file to binarize. Each cell footprint is
        transformed independently and added back to the output cellset.
    output_cellset_file : str
        Path to the .isxd cellset file that has been binarized.
    threshold : double
        Threshold for updating pixels.
        Pixels with values above threshold are set to 1, otherwise set to 0.
    use_percentile_threshold : bool
        If true, the provided threshold is treated as a percentile.
    """
    if not os.path.exists(input_cellset_file):
        raise FileExistsError('Input file not found: {}'.format(input_cellset_file))

    if os.path.exists(output_cellset_file):
        raise FileExistsError('Output file already exists: {}'.format(output_cellset_file))

    if use_percentile_threshold and (threshold < 0 or threshold > 100):
        raise ValueError('Percentile threshold must be between 0 and 100.')

    isx._internal.c_api.isx_binarize_cell_set(
        input_cellset_file.encode('utf-8'),
        output_cellset_file.encode('utf-8'),
        threshold,
        use_percentile_threshold)

def crop_cell_set(input_cellset_file, output_cellset_file, crop):
    """Crop each footprint of a cell set to produce a new cell set with the desired size.

    Arguments
    ---------
    input_cellset_file : str
        Path to the .isxd cellset file to crop.
    output_cellset_file : str
        Path to the .isxd cellset file that has been cropped.
    crop : 4-tuple<int>
        A list of 4 values indicating how many pixels to crop on each side: [left, right, top, bottom].
    """
    if not os.path.exists(input_cellset_file):
        raise FileNotFoundError('Input file not found: {}'.format(input_file))

    if os.path.exists(output_cellset_file):
        raise FileExistsError('Output file already exists: {}'.format(output_cellset_file))

    if len(crop) != 4:
        raise ValueError('The amount of cropping for all 4 sides must be specified as [left, right, top, bottom].')
    if any(k < 0 for k in crop):
        raise ValueError('The amount of cropping on each side must be a positive integer.')

    isx._internal.c_api.isx_crop_cell_set(
        input_cellset_file.encode('utf-8'),
        output_cellset_file.encode('utf-8'),
        crop[0], crop[1], crop[2], crop[3])


def transform_cell_set(input_cellset_file, output_cellset_file, pad_value=np.nan):
    """Transform an isxd cell set to its pre-motion-correction dimensions by padding the cell footprints.

    Arguments
    ---------
    isxd_cellset_file : str
        Path to a .isxd cellset file to align. Each cell footprint is
        transformed independently and added back to the output cellset.
    output_transformed_cellset_file : str
        Path to the .isxd cellset file that has been transformed.
    pad_value : valid numpy value
        Value to fill the padded region of the footprints.
    """
    if not os.path.exists(input_cellset_file):
        raise FileNotFoundError('Input file not found: {}'.format(input_cellset_file))

    if os.path.exists(output_cellset_file):
        raise FileExistsError('Output file already exists: {}'.format(output_cellset_file))

    isx._internal.c_api.isx_transform_cell_set(
        input_cellset_file.encode('utf-8'),
        output_cellset_file.encode('utf-8'),
        pad_value)

def compute_spatial_overlap_cell_set(input_cellset_file1, input_cellset_file2, output_csv_file, accepted_cells_only=False):
    """Compute the pairwise spatial overlap between footprints from two input cell sets.
        If the two footprints are binary then the spatial overlap is computed as the pairwise Jaccard index
        If the two footprints are analog then the spatial overlap is computed as the pairwise normalized cross correlation
        In order to compute the spatial overlap between binary and analog footprints it's necessary to binarize the analog footprints first.

    Arguments
    ---------
    input_cellset_file1 : str
        Path to the first .isxd cellset file.
    input_cellset_file2 : str
        Path to the second .isxd cellset file.
    output_csv_file : str
        Path to the .csv file containing the f1 scores.
    accepted_cells_only : bool
        Whether or not to use accepted cells from the input cell sets only, or to use both accepted and undecided cells.
    """
    for input_file in [input_cellset_file1, input_cellset_file2]:
        if not os.path.exists(input_file):
            raise FileNotFoundError('Input file not found: {}'.format(input_file))

    if os.path.exists(output_csv_file):
        raise FileExistsError('Output file already exists: {}'.format(output_csv_file))

    isx._internal.c_api.isx_compute_spatial_overlap_cell_set(
        input_cellset_file1.encode('utf-8'),
        input_cellset_file2.encode('utf-8'),
        output_csv_file.encode('utf-8'),
        accepted_cells_only)

def register_cellsets(
        input_cellset_file1,
        input_cellset_file2,
        output_spatial_overlap_csv_file,
        output_registration_matrix_csv_file,
        output_directory='.',
        lower_threshold=0.3,
        upper_threshold=0.7,
        accepted_cells_only=False,
        primary_cellset_name='primary',
        secondary_cellset_name='secondary',
        primary_color=0x00FF00,
        secondary_color=0xFF0000):
    """
    Register two cellsets
    Computes the pairwise spatial overlap of two cellsets in order to match cells
    If the two footprints are binary then the spatial overlap is computed as the pairwise Jaccard index
    If the two footprints are analog then the spatial overlap is computed as the pairwise normalized cross correlation
    Throws an exception if the footprint types of the cellsets are incomptible (i.e., binary and analog)

    Arguments
    ---------
    input_cellset_file1 : str
        Path to the first .isxd cellset file.
    input_cellset_file2 : str
        Path to the second .isxd cellset file.
    output_spatial_overlap_csv_file : str
        Path to the .csv file containing the pairwise spatial overlap scores.
    output_registration_matrix_csv_file : str
        Path to the .csv file containing the registration matrix.
    output_directory : str
        Path to the output directory. Cellmaps will be save to this directory. 
    lower_threshold : double
        Maximum score between two cells that can be rejected as a match.
    upper_threshold : double
        Minimum score between two cells that can be accepted as a match.
    accepted_cells_only : bool
        Whether or not to use accepted cells from the input cell sets only, or to use both accepted and undecided cells.
    primary_cellset_name : string
        Name of the first cellset to use in .csv files
    secondary_cellset_name : string
        Name of the second cellset to use in .csv files
    primary_color : int > 0
        Color of cells from the first cellset to use in cellmaps
    secondary_color : int > 0
        Color of cells from the second cellset to use in cellmaps
    """
    for input_file in [input_cellset_file1, input_cellset_file2]:
        if not os.path.exists(input_file):
            raise FileNotFoundError('Input file not found: {}'.format(input_file))

    for output_file in [output_spatial_overlap_csv_file, output_registration_matrix_csv_file]:
        if os.path.exists(output_file):
            raise FileExistsError('Output file already exists: {}'.format(output_file))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not 0 <= lower_threshold <= 1:
        raise TypeError("Lower threshold must be between 0 and 1.")
    if not 0 <= upper_threshold <= 1:
        raise TypeError("Upper threshold must be between 0 and 1.")

    isx._internal.c_api.isx_register_cellsets(
        input_cellset_file1.encode('utf-8'),
        input_cellset_file2.encode('utf-8'),
        output_spatial_overlap_csv_file.encode('utf-8'),
        output_registration_matrix_csv_file.encode('utf-8'),
        output_directory.encode('utf-8'),
        accepted_cells_only,
        lower_threshold,
        upper_threshold,
        primary_cellset_name.encode('utf-8'),
        secondary_cellset_name.encode('utf-8'),
        primary_color,
        secondary_color)

def deconvolve_cellset(
    input_raw_cellset_files,
    output_denoised_cellset_files=None,
    output_spike_eventset_files=None,
    accepted_only=False,
    spike_snr_threshold=3.0,
    noise_range=(0.25, 0.5),
    noise_method='mean',
    first_order_ar=True,
    lags=5,
    fudge_factor=0.96,
    deconvolution_method='oasis'):
    """
    Deconvolve temporal traces of cellsets.

    Arguments
    ---------
    input_raw_cellset_files : list<str>
        The file paths of the cellsets to perform deconvolution on.
    output_denoised_cellset_files : list<str>
        The file paths of the output denoised cellsets. If None, then not created.
    output_spike_eventset_files : list<str>
        The file paths of the output spike eventsets. If None, then not created.
    accepted_only : bool
        If True, only deconvolve for accepted cells, otherwise accepted and undecided.
    spike_snr_threshold : float > 0
        SNR threshold for spike outputs. This is in units of noise which is estimated from the raw temporal traces.
    noise_range : 0 <= 2-tuple<float> <= 1
        Range of frequencies to average for estimating pixel noise.
        Maximum frequency must be greater than or equal to minimum frequency.
    noise_method : str
        Specifies averaging method for noise. Must be one of ('mean', 'median', 'logmexp').
    first_order_ar : bool
        If True, use an AR(1) model, otherwise use AR(2).
    lags : int > 0
        Number of lags for estimating time constant.
    fudge_factor : float > 0
        Fudge factor for reducing time constant bias.
    deconvolution_method : str
        Decoonvolution method for calcium dynamics. Must be one of ('oasis', 'scs').
        Note: SCS is significantly slower than OASIS but AR(2) models are currently only supported with SCS.
    """

    if deconvolution_method == 'oasis' and not first_order_ar:
        raise ValueError("Deconvolution with OASIS only works for an AR(1) model");

    if noise_range[1] < noise_range[0]:
        raise ValueError("Maximum must be greater than or equal to minimum for noise range");

    if not output_denoised_cellset_files and not output_spike_eventset_files:
        raise ValueError("Must specify at least one type of deconvolution output");

    noise_method_map = { 'mean' : 0, 'median' : 1, 'logmexp' : 2 }
    deconvolution_method_map = { 'oasis' : 0, 'scs' : 1 }

    noise_method = noise_method_map[noise_method]
    deconvolution_method = deconvolution_method_map[deconvolution_method]

    out_denoised_arr = None
    if output_denoised_cellset_files:
        num_files, in_raw_arr, out_denoised_arr = isx._internal.check_input_and_output_files(
            input_raw_cellset_files, output_denoised_cellset_files)
    
    out_spike_arr = None
    if output_spike_eventset_files:
        num_files, in_raw_arr, out_spike_arr = isx._internal.check_input_and_output_files(
            input_raw_cellset_files, output_spike_eventset_files)
    
    isx._internal.c_api.isx_cellset_deconvolve(
        num_files,
        in_raw_arr,
        out_denoised_arr,
        out_spike_arr,
        accepted_only,
        spike_snr_threshold,
        noise_range[0],
        noise_range[1],
        noise_method,
        first_order_ar,
        lags,
        fudge_factor,
        deconvolution_method)

def estimate_vessel_diameter(
    input_movie_files, 
    output_vessel_set_files, 
    lines, 
    time_window = 2, 
    time_increment = 1,
    output_units = "pixels",
    estimation_method = "Non-Parametric FWHM",
    height_estimate_rule = "independent",
    auto_accept_reject = True,
    rejection_threshold_fraction = 0.2,
    rejection_threshold_count = 5,
    vessel_names = []):
    """
    Estimates blood vessel diameter along each input line over time

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to analyse.
    output_vessel_set_files : list<str>
        The file paths of the output vessel set files.
    lines : Union[list<list<list<int>>, np.ndarray]
        The pairs of points to perform analysis on.
        This can be represented as a list of points, or a numpy.ndarray object
        e.g. [ [ [1, 1], [2, 2] ], [ [2, 2], [3, 3] ], [ [3, 3], [4, 4] ]]
    time_window : float
        This specifies the duration in seconds of the time window to use for every measurement.
    time_increment : float
        This specifies the time shift in seconds between consecutive measurements.
        When the time increment is smaller than the time window, consecutive windows will overlap.
        The time increment must be less than or equal to the time window
    output_units : string in {'pixels', 'microns'}
        Output units for vessel diameter estimation.
    estimation_method : string in {'Non-Parametric FWHM', 'Parametric FWHM'}
        The type of method to use for vessel diameter estimation.
        Both methods estimate diameter from a line profile extracted from the input movie using the input contours.
        Parametric FWHM fits the line profile to a Lorentzian curve.
        Non-Parametric FWHM measures the distance between the midpoints of the line profile peak.
    height_estimate_rule: string in {'independent', 'global', 'local'}
        Used in Non-Parametric FWHM estimation method.
        Describes the method to use for determing the midpoint height on each side of the line profile peak.
        Can be one of the following values:
        * global: Take the halfway point between the max and the global min.
        * local: Take the largest of the two halfway points between min/max.
        * independent: The height estimate will be independent on both sides of the peak.
    auto_accept_reject: bool
        Flag indicating whether the vessels should be auto accepted/rejected.
        Rejected vessels are identified as those with derivatives greater than a particular fraction of the mean.
    rejection_threshold_fraction: float
        Parameter for auto accept/reject functionality.
        The max fraction of the mean diameter allowed for a derivative in a particular vessel diameter trace.
    rejection_threshold_count: int
        Parameter for auto accept/reject functionality.
        The number of threshold crossings allowed in a particular vessel diameter trace.
    vessel_names: list<str>
        List of names to assign vessel in the output vessel sets.
        If empty, then vessel will have default names.
    """

    # File checks
    #   - Input files must exist
    #   - Number of input files must match number of output files
    num_files, movie_files, vessel_set_files = isx._internal.check_input_and_output_files(input_movie_files, output_vessel_set_files)

    output_units_map = {'pixels' : 0, 'microns' : 1}

    # Points check
    num_lines = len(lines)
    if num_lines <= 0:
        raise ValueError('At least one line needs to be specified')
    
    for pair in lines:
        if(len(pair) != 2):
            raise ValueError('All pairs must have two points.')
        for point in pair:
            if(len(point) != 2):
                raise ValueError('All points must only have two coordinates (x and y-coordinates, respectively)')
    
    points = isx._internal.ndarray_as_type(np.array(lines), np.dtype(np.int64))
    points_p = points.ctypes.data_as(isx._internal.Int64Ptr)

    dim_points = points.ndim
    if dim_points != 3:
        raise ValueError('Input points must be a 3-D numpy array')
    
    # Time window and Time increment check
    if time_increment <= 0 or time_window <= 0:
        raise ValueError('Time increment and time window must be greater than 0')

    if not output_units in output_units_map.keys():
        raise ValueError('Invalid units. Valid units includes: {}'.format(*output_units_map.keys()))
    
    vessel_names_arr = isx._internal.list_to_ctypes_array(vessel_names, ctypes.c_char_p)
    use_vessel_names = len(vessel_names) > 0 
    if vessel_names and num_lines != len(vessel_names):
            raise ValueError("Number of lines must equal number of vessel names.")

    isx._internal.c_api.isx_estimate_vessel_diameter(
        num_files,
        movie_files,
        vessel_set_files,
        num_lines,
        points_p,
        time_window,
        time_increment,
        output_units_map[output_units],
        estimation_method.encode('utf-8'),
        height_estimate_rule.encode('utf-8'),
        auto_accept_reject,
        rejection_threshold_fraction,
        rejection_threshold_count,
        use_vessel_names,
        vessel_names_arr
    )
    return

def estimate_rbc_velocity(
    input_movie_files,
    output_vessel_set_files,
    rois,
    time_window = 10,
    time_increment = 2,
    output_units = "pixels",
    save_correlation_heatmaps = True
):
    """
    Estimates red blood cell (rbc) velocity within each region of interest over time

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to analyse.
    output_vessel_set_files : list<str>
        The file paths of the output vessel set files.
    rois : Union[list<list<list<int>>, np.ndarray]
        The groups of points to perform analysis on.
        This can be represented as a list of points, or a numpy.ndarray object
        E.g. [ [ [0, 0], [0, 1], [1, 0], [1, 1] ], [ [2, 0], [2, 1], [3, 0], [3, 1] ] ]
    time_window : float
        This specifies the duration in seconds of the time window to use for every measurement.
    time_increment : float
        This specifies the time shift in seconds between consecutive measurements.
        When the time increment is smaller than the time window, consecutive windows will overlap.
        The time increment must be less than or equal to the time window.
    output_units : string in {'pixels', 'microns'}
        Output units for vessel velocity estimation.
    save_correlation_heatmaps: bool
        This specifies whether to save the correlation heatmaps to the vessel set or not
    """

    # File checks
    #   - Input files must exist
    #   - Number of input files must match number of output files
    num_files, movie_files, vessel_set_files = isx._internal.check_input_and_output_files(input_movie_files, output_vessel_set_files)

    output_units_map = {'pixels' : 2, 'microns' : 3}

    # Points check
    num_rois = len(rois)
    if num_rois <= 0:
        raise ValueError('At least one ROI needs to be specified')
    
    for vertices in rois:
        if(len(vertices) != 4):
            raise ValueError('All rois must have 4 points.')
        for point in vertices:
            if(len(point) != 2):
                raise ValueError('All points must only have two coordinates (x and y-coordinates, respectively)')
    
    points = isx._internal.ndarray_as_type(np.array(rois), np.dtype(np.int64))
    points_p = points.ctypes.data_as(isx._internal.Int64Ptr)

    dim_points = points.ndim
    if dim_points != 3:
        raise ValueError('Input points must be a 3-D numpy array')
    
    # Time window and Time increment check
    if time_increment <= 0 or time_window <= 0:
        raise ValueError('Time increment and time window must be greater than 0')

    if not output_units in output_units_map.keys():
        raise ValueError('Invalid units. Valid units includes: {}'.format(*output_units_map.keys()))

    isx._internal.c_api.isx_estimate_rbc_velocity(num_files, movie_files, vessel_set_files, num_rois, points_p, time_window, time_increment, output_units_map[output_units], save_correlation_heatmaps)
    return

def create_neural_activity_movie(
    input_cell_set_files,
    output_neural_movie_files,
    accepted_cells_only=False
):
    """
    Create a neural activity movie by aggregating individual cell activity.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell set to process.
    output_neural_movie_files : list<str>
        The file paths of the output neural movie files.
    accepted_cells_only : bool
        If True, output movies will only included the activity from accepted cells.
    """
    num_files, movie_files, neural_movie_files = isx._internal.check_input_and_output_files(input_cell_set_files, output_neural_movie_files)

    isx._internal.c_api.isx_create_neural_activity_movie(
        num_files,
        movie_files,
        neural_movie_files,
        accepted_cells_only
    )


def interpolate_movie(
    input_movie_files,
    output_interpolated_movie_files,
    interpolate_dropped=True,
    interpolate_blank=True,
    max_consecutive_invalid_frames=1
):
    """
    Replace invalid movie time samples with interpolated data.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to process.
    output_interpolated_movie_files : list<str>
        The file paths of the output interpolated movie files.
    interpolate_dropped : bool
        If True, dropped frames will be interpolated.
    interpolate_blank : bool
        If True, blank frames will interpolated.
    max_consecutive_invalid_frames : int > 0
        The maximum number of consecutive invalid frames that can be interpolated over.
    """
    num_files, movie_files, interpolated_movie_files = isx._internal.check_input_and_output_files(input_movie_files, output_interpolated_movie_files)

    isx._internal.c_api.isx_interpolate_movie(
        num_files,
        movie_files,
        interpolated_movie_files,
        interpolate_dropped,
        interpolate_blank,
        max_consecutive_invalid_frames
    )

def estimate_vessel_diameter_single_vessel(
    input_movie_file,
    line,
    start_frame,
    end_frame,
    output_coordinates=False
):
    """
    Estimates blood vessel diameter for a single vessel.
    This function exposes an internal part of the function ``isx.estimate_vessel_diameter``,
    where a single measurement of diameter is estimated by fitting a Lorentzian curve to a line profile.
    The purpose of this function is help troubleshoot performance of the vessel diameter algorithm
    when it does not work as expected.

    Arguments
    ---------
    input_movie_file : str
        The file path of the movie to analyse.
    line : Union[list<list<int>>, tuple<tuple<int, int>, tuple<int, int>>, np.ndarray]
        A line to measure diameter with.
        This can be represented as a list of two points, a tuple of two points, or a ``np.ndarray`` object.
        E.g. ((1, 1), (2, 2))
    start_frame : int >= 0
        Start frame of the window to measure diameter from in the input movie.
    end_frame : int >= 0
        End frame of the window to measure diameter from in the input movie.
    output_coordinates : bool, optional
        If true, output (x, y) coordinates of the pixels sampled along the user-defined line
        This is an optional parameter that is set to false by default.

    Returns
    -------
    line_profile : np.ndarray
        1D array containing the line profile extracted from the input movie.
        The values in the line profile are in units of pixel intensity relative to the input movie.
        The values are extracted from a mean projection image of the input movie for the specified window range.
        The line profile is calculated by averaging three parallel line profiles of equal length including the input line
        and two additional lines, each one pixel apart on either side of the input line.
        The line profile is then background subtracted using the minimum value in the pixel values.
        **Note**: The line profile is extracted from the input movie by representing the input line as a raster line segment.
        The raster line segment is computed using the following OpenCV function: `cv::LineIterator <https://docs.opencv.org/4.x/dc/dd2/classcv_1_1LineIterator.html>`_.
        The raster scan results in a line profile that does not neccessarily have the same number of pixels as the length of the input line.
        This has implications for how the Lorentzian curve fit results should be interpreted.
        Two dictionaries are returned by this function.
        The first, ``model_fit``, is relative the raster line segment.
        The second, ``diameter_estimate``, is relative to the user-defined contour line in image space.
    model_fit : dict
        Dictionary containing the Lorentzian curve fit parameters. Includes the following keys.
        ``amplitude``: the peak amplitude of the Lorentzian curve.
        ``fwhm``: the full width half max of the Lorentzian curve.
        This measurement is in pixels and is relative to the number of pixels in the line profile.
        ``peak_center``: the peak center of Lorentzian curve.
        This measurement is in pixels and is relative to the number of pixels in the line profile.
        Using these curve fit parameters, the Lorentzian function can be charecterized as the following function:
        ``L(x) = (amplitude * 0.5 * fwhm / PI) / ((x - peak_center)^2 + (0.5 * fwhm)^2))``
    diameter_estimate : dict
        Dictionary containing the estimated diameter results.
        These values are obtained by scaling the ``fwhm`` and ``peak_center`` values from the ``model_fit`` dictionary by the relative length of the full contour line (in image space) to the length of the line profile.
        Includes the following keys.
        ``length``: the length of the diameter estimate.
        ``center``: the center point of the diameter estimate on the user line, relative to the start point of the input line.
    line_coords : np.ndarray, optional
        2D array containing (x, y) coordinates of the pixels values in image space for the raster line segment representing the input line.
    """
    def get_vessel_line_num_pixels(input_movie_file, line):
        movie = isx.Movie.read(input_movie_file)
        image_width, image_height = movie.spacing.num_pixels
        points = isx._internal.ndarray_as_type(np.array(line), np.dtype(np.int64))
        points_p = points.ctypes.data_as(isx._internal.Int64Ptr)

        num_pixels = ctypes.c_int()
        num_pixels_p = ctypes.pointer(num_pixels)
        isx._internal.c_api.isx_get_vessel_line_num_pixels(points_p, image_width, image_height, num_pixels_p)
        return num_pixels.value

    def get_vessel_line_coordinates(input_movie_file, line, num_pixels):
        movie = isx.Movie.read(input_movie_file)
        image_width, image_height = movie.spacing.num_pixels

        line_x = np.zeros((num_pixels,), dtype=np.int32)
        line_x_p = line_x.ctypes.data_as(isx._internal.IntPtr)

        line_y = np.zeros((num_pixels,), dtype=np.int32)
        line_y_p = line_y.ctypes.data_as(isx._internal.IntPtr)

        isx._internal.c_api.isx_get_vessel_line_coordinates(points_p, image_width, image_height, line_x_p, line_y_p)
        line_coords = np.column_stack((line_x, line_y))

        return line_coords
    
    num_pixels = get_vessel_line_num_pixels(input_movie_file, line)

    line_profile = np.zeros((num_pixels,), dtype=np.float64)
    fit_amplitude = ctypes.c_double()
    fit_fwhm = ctypes.c_double()
    fit_peak_center = ctypes.c_double()
    estimate_length = ctypes.c_double()
    estimate_center = ctypes.c_double()

    line_profile_p = line_profile.ctypes.data_as(isx._internal.DoublePtr)
    fit_amplitude_p = ctypes.pointer(fit_amplitude)
    fit_fwhm_p = ctypes.pointer(fit_fwhm)
    fit_peak_center_p = ctypes.pointer(fit_peak_center)
    estimate_length_p = ctypes.pointer(estimate_length)
    estimate_center_p = ctypes.pointer(estimate_center)

    points = isx._internal.ndarray_as_type(np.array(line), np.dtype(np.int64))
    points_p = points.ctypes.data_as(isx._internal.Int64Ptr)

    isx._internal.c_api.isx_estimate_vessel_diameter_single_vessel(
        input_movie_file.encode('utf-8'),
        points_p,
        start_frame,
        end_frame,
        line_profile_p,
        fit_amplitude_p,
        fit_fwhm_p,
        fit_peak_center_p,
        estimate_length_p,
        estimate_center_p)
    
    model_fit = {"amplitude" : fit_amplitude.value, "fwhm" : fit_fwhm.value, "peak_center" : fit_peak_center.value}
    diameter_estimate = {"length" : estimate_length.value, "center" : estimate_center.value}

    if output_coordinates:
        line_coords = get_vessel_line_coordinates(input_movie_file, line, num_pixels)
        return line_profile, model_fit, diameter_estimate, line_coords
    else:
        return line_profile, model_fit, diameter_estimate


def decompress(input_isxc_file, output_dir):
    """
    Decompress an isxc file to the corresponding isxd file.

    Output file will have the name <input_file_name>.isxd

    Arguments
    ---------
    input_isxc_file :str
        The file path of the isxc file to decompress.
    output_dir : str
        The path of the directory to write the isxd file.
    """
    isx._internal.c_api.isx_decompress(input_isxc_file.encode('utf-8'), output_dir.encode('utf-8'))
