import os
import shutil
import isx

test_data_path = os.environ['ISX_TEST_DATA_PATH']


def delete_files_silently(files):
    if isinstance(files, str):
        files = [files]
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

def delete_dirs_silently(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    for d in dirs:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def is_file(f):
    return os.path.isfile(f)
