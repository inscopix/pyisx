"""
The util module contains miscellanenous functions that would ideally
be built-in for Python or are specific to the isx package.
"""

import os
import ctypes

import isx._internal

def get_file_stem(in_file):
    """Get the file stem for file that can have multiple extensions."""
    in_file_stem = os.path.basename(in_file)
    while '.' in in_file_stem:
        in_file_stem = os.path.splitext(in_file_stem)[0]
    return in_file_stem

def get_file_extension(in_file):
    """ Get full extension for file that can have multiple extensions."""
    basename = os.path.basename(in_file)
    return basename[basename.index('.'):]


def make_output_file_path(in_file, out_dir, suffix=None, ext='isxd'):
    """
    Make an output file path from an input path, output directory, suffix and extension.

    This is useful for generate output file paths for processing steps.

    Arguments
    ---------
    in_file : str
        The input file path.
    out_dir : str
        The output directory path.
    suffix : str
        The suffix to append to the file stem with a '-'.
        If left empty, no suffix is appended.
    ext : 'isxd'
        The output file extension, not including the '.'.

    Returns
    -------
    str
        The output file path.

    Examples
    --------
    Make the output file path for a preprocessed recording.

    >>> make_output_file_path('in_dir/in_file.xml', 'out_dir', 'PP')
    'out_dir/in_file-PP.isxd'
    """
    in_file_stem = get_file_stem(in_file)
    
    if suffix:
        return os.path.join(out_dir, '{}-{}.{}'.format(in_file_stem, suffix, ext))
    else:
         return os.path.join(out_dir, '{}.{}'.format(in_file_stem, ext))


def make_output_file_paths(in_files, out_dir, suffix, ext='isxd'):
    """ Like :func:`isx.make_output_file_path`, but for many files.
    """
    return [make_output_file_path(f, out_dir, suffix, ext=ext) for f in in_files]


def verify_deinterleave(in_file, efocus):
    """
    Verify if all frames from movie has the same efocus as provided.

    Arguments
    ---------
    in_file : str
        The input file path.
    efocus : int
        The efocus value to be compared with.

    Returns
    -------
    bool
        True if the movie is successfully verified, False otherwise.
    """
    success = ctypes.c_int()
    isx._internal.c_api.isx_movie_verify_deinterleave(in_file.encode('utf-8'), ctypes.c_uint16(efocus), ctypes.byref(success))
    return success.value > 0
