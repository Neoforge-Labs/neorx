"""Deprecated alias for :mod:`neorx.mirrorfold`."""

import warnings

warnings.warn(
    "modules.mirrorfold is deprecated; import from neorx.mirrorfold instead. "
    "This shim is removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from neorx.mirrorfold import *  # noqa: F401,F403,E402
from neorx.mirrorfold import __doc__  # noqa: F401,E402
