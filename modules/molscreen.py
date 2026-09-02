"""Deprecated alias for :mod:`neorx.molscreen`."""

import warnings

warnings.warn(
    "modules.molscreen is deprecated; import from neorx.molscreen instead. "
    "This shim is removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from neorx.molscreen import *  # noqa: F401,F403,E402
from neorx.molscreen import __doc__  # noqa: F401,E402
