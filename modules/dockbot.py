"""Deprecated alias for :mod:`neorx.dockbot`."""

import warnings

warnings.warn(
    "modules.dockbot is deprecated; import from neorx.dockbot instead. "
    "This shim is removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from neorx.dockbot import *  # noqa: F401,F403,E402
from neorx.dockbot import __doc__  # noqa: F401,E402
