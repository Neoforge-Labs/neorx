"""Deprecated alias for :mod:`neorx.causalbiorl`."""

import warnings

warnings.warn(
    "modules.causalbiorl is deprecated; import from neorx.causalbiorl instead. "
    "This shim is removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from neorx.causalbiorl import *  # noqa: F401,F403,E402
from neorx.causalbiorl import __doc__  # noqa: F401,E402
