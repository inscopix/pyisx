"""
The Inscopix Python API package.
"""

from .core import *
from .io import *
from .util import *

from .core import __version__

from ._internal import is_minimal_api as _is_minimal_api
if not _is_minimal_api:
    from .algo import *
