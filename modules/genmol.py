"""Deprecated alias for :mod:`neorx.genmol`."""

import warnings

warnings.warn(
    "modules.genmol is deprecated; import from neorx.genmol instead. "
    "This shim is removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from neorx.genmol import *  # noqa: F401,F403,E402
from neorx.genmol import __doc__  # noqa: F401,E402
