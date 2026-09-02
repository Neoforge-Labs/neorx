"""Deprecated alias for :mod:`neorx.core`."""

import warnings

warnings.warn(
    "modules.neorx is deprecated; import from neorx.core instead. "
    "This shim is removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from neorx.core import *  # noqa: F401,F403,E402
from neorx.core import __doc__  # noqa: F401,E402
