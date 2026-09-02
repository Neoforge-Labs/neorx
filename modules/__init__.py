"""Deprecated import path. Use ``neorx.*`` instead.

This package exists only inside the repository, to keep untracked
benchmark and evaluation scripts working after the move to src/neorx.
It is excluded from built artifacts and is removed in 0.3.0.

Any ``modules.<pkg>[.<sub>...]`` dotted import -- however deep -- is
resolved by a ``sys.meta_path`` finder installed below. The finder
aliases the corresponding ``neorx.<pkg>[...]`` module into
``sys.modules`` under the *old* dotted name, so the SAME module object
is returned (module identity and ``isinstance`` checks against the
real module keep working). This is why the shim does not use
``from neorx.x import *``: a star-import builds a distinct module
object and copies names into it, which breaks identity.

A ``DeprecationWarning`` is raised once per distinct old dotted name.

Mapping rules:

* ``modules.neorx[.rest...]`` -> ``neorx.core[.rest...]``, except that
  the following leaf names moved during the core subdivision and are
  translated first: ``graph_builder``, ``models``, ``persistence`` ->
  under ``graph``; ``identifier``, ``counterfactual`` -> under
  ``causal``; ``classifier``, ``tissue_filter`` -> under ``bio``;
  ``scorer``, ``admet`` -> under ``scoring``; ``data_sources`` ->
  renamed to ``sources``. Everything else under ``modules.neorx``
  (``pipeline``, ``report``, ``cache``, ``api``, ``validator``,
  ``literature_validator``) maps straight through to ``neorx.core``.
* Every other top-level (``genmol``, ``causalbiorl``, ``molscreen``,
  ``dockbot``, ``mirrorfold``) is a plain prefix swap:
  ``modules.X[.rest...]`` -> ``neorx.X[.rest...]``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

# Leaf names moved during the neorx.core subdivision. The value is the
# dotted path *under* neorx.core that now holds them.
_CORE_SUBDIVISION = {
    "graph_builder": "graph.graph_builder",
    "models": "graph.models",
    "persistence": "graph.persistence",
    "identifier": "causal.identifier",
    "counterfactual": "causal.counterfactual",
    "classifier": "bio.classifier",
    "tissue_filter": "bio.tissue_filter",
    "scorer": "scoring.scorer",
    "admet": "scoring.admet",
    "data_sources": "sources",
}

_KNOWN_TOP_LEVEL = {"neorx", "genmol", "causalbiorl", "molscreen", "dockbot", "mirrorfold"}

_warned_names: set[str] = set()


def _real_name(fullname: str) -> str | None:
    """Map a ``modules[.x...]`` dotted name to its ``neorx`` equivalent.

    Returns ``None`` if ``fullname`` is not a name this shim handles
    (e.g. ``modules`` itself, or an unrecognised top-level package).
    """
    if fullname == "modules" or not fullname.startswith("modules."):
        return None

    parts = fullname.split(".")
    top = parts[1]
    if top not in _KNOWN_TOP_LEVEL:
        return None
    rest = parts[2:]

    if top == "neorx":
        if not rest:
            return "neorx.core"
        first, remainder = rest[0], rest[1:]
        replacement = _CORE_SUBDIVISION.get(first, first)
        return ".".join(["neorx", "core", *replacement.split("."), *remainder])

    return ".".join(["neorx", top, *rest])


class _ModulesShimFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Aliases ``modules.*`` dotted imports onto the real ``neorx.*`` modules.

    ``create_module`` returns the already-imported real module object
    (not a fresh one), so the module inserted into ``sys.modules`` under
    the old name IS the real module -- ``sys.modules["modules.neorx.identifier"]
    is neorx.core.causal.identifier`` holds.
    """

    def find_spec(self, fullname, path, target=None):  # noqa: ARG002
        real_name = _real_name(fullname)
        if real_name is None:
            return None

        real_module = importlib.import_module(real_name)

        if fullname not in _warned_names:
            _warned_names.add(fullname)
            warnings.warn(
                f"{fullname} is deprecated; import from {real_name} instead. "
                "This shim is removed in 0.3.0.",
                DeprecationWarning,
                stacklevel=3,
            )

        spec = importlib.util.spec_from_loader(
            fullname,
            self,
            origin=getattr(real_module, "__file__", None),
            is_package=hasattr(real_module, "__path__"),
        )
        spec.loader_state = real_module
        return spec

    def create_module(self, spec):
        return spec.loader_state

    def exec_module(self, module):  # noqa: ARG002
        pass  # already fully initialised by the real import; nothing to do


def _install() -> None:
    if not any(isinstance(finder, _ModulesShimFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ModulesShimFinder())


_install()
