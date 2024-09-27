import numpy as np

import isx


def write_sample_cellset(cs_out_file):

    # create sample data that will be used to make the cell set
    num_cells = 5
    timing = isx.Timing(num_samples=14, period=isx.Duration.from_msecs(10), dropped=[3, 11], cropped=[[4, 6], [8, 9]])
    spacing = isx.Spacing(num_pixels=(3, 5))
    names = ['C{}'.format(k) for k in range(num_cells)]
    images = np.random.randn(*[num_cells, *spacing.num_pixels]).astype(np.float32)
    traces = np.random.randn(*[num_cells, timing.num_samples]).astype(np.float32)

    cs_out = isx.CellSet.write(cs_out_file, timing, spacing)
    for k in range(num_cells):
        cs_out.set_cell_data(k, images[k, :, :], traces[k, :], name=names[k])
    cs_out.flush()

    return {'num_cells' : num_cells,
            'spacing' : spacing,
            'timing' : timing,
            'names' : names,
            'traces' : traces,
            'images' : images}

def write_sample_vessel_diameter_set(vs_out_file):

    # create sample data that will be used to make the vessel set
    num_vessels = 5
    timing = isx.Timing(num_samples=14, period=isx.Duration.from_msecs(10))
    spacing = isx.Spacing(num_pixels=(3, 5))
    names = ['V{}'.format(k) for k in range(num_vessels)]
    images = np.random.randn(*[num_vessels, *spacing.num_pixels]).astype(np.float32)
    lines = np.random.randint(0, min(spacing.num_pixels), (num_vessels, 2, 2))
    traces = np.random.randn(*[num_vessels, timing.num_samples]).astype(np.float32)
    cen_traces = np.random.randn(*[num_vessels, timing.num_samples]).astype(np.float32)
    vs_out = isx.VesselSet.write(vs_out_file, timing, spacing, 'vessel diameter')
    for k in range(num_vessels):
        vs_out.set_vessel_diameter_data(k, images[k, :, :], lines[k, :, :], traces[k, :], cen_traces[k, :],name=names[k])
    vs_out.flush()

    return {'num_vessels' : num_vessels,
            'spacing' : spacing,
            'timing' : timing,
            'names' : names,
            'traces' : traces,
            'center_traces' : cen_traces,
            'lines' : lines,
            'images' : images}

def write_sample_rbc_velocity_set(vs_out_file):

    # create sample data that will be used to make the vessel set
    num_vessels = 5
    correlation_sizes = np.random.randint(2, 5, size=(num_vessels, 2))
    timing = isx.Timing(num_samples=14, period=isx.Duration.from_msecs(10))
    spacing = isx.Spacing(num_pixels=(3, 5))
    names = ['V{}'.format(k) for k in range(num_vessels)]
    images = np.random.randn(*[num_vessels, *spacing.num_pixels]).astype(np.float32)
    lines = np.random.randint(0, min(spacing.num_pixels), (num_vessels, 4, 2))
    traces = np.random.randn(*[num_vessels, timing.num_samples]).astype(np.float32)
    dir_traces = np.random.randn(*[num_vessels, timing.num_samples]).astype(np.float32)
    corr_traces = [np.random.randn(*[timing.num_samples, 3, correlation_sizes[i][0], correlation_sizes[i][1]]).astype(np.float32) for i in range(num_vessels)]
    vs_out = isx.VesselSet.write(vs_out_file, timing, spacing, 'rbc velocity')
    for k in range(num_vessels):
        vs_out.set_rbc_velocity_data(k, images[k, :, :], lines[k, :, :], traces[k, :], dir_traces[k, :], corr_traces[k], name=names[k])
    vs_out.flush()

    return {'num_vessels' : num_vessels,
            'spacing' : spacing,
            'timing' : timing,
            'names' : names,
            'traces' : traces,
            'direction_traces' : dir_traces,
            'correlations_traces' : corr_traces,
            'lines' : lines,
            'images' : images}
