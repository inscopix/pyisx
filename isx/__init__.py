"""
The Inscopix Python API package.
"""

from .core import *
from .io import *
from .util import *

from .core import __version__

from ._internal import is_with_algos as _is_with_algos
if _is_with_algos:
    from .algo import *
