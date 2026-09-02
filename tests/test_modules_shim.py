"""The repo-only modules/ shim keeps untracked root scripts working.

The shim is a sys.meta_path finder (installed by modules/__init__.py) that
aliases any ``modules.<pkg>[.<sub>...]`` dotted import onto the real
``neorx.<pkg>[...]`` module, at arbitrary depth, returning the SAME module
object (not a copy made via ``from x import *``) so identity and
``isinstance`` checks keep working.
"""

import importlib
import sys
import warnings

import pytest

SHIMMED = [
    ("modules.neorx", "neorx.core"),
    ("modules.genmol", "neorx.genmol"),
    ("modules.causalbiorl", "neorx.causalbiorl"),
    ("modules.molscreen", "neorx.molscreen"),
    ("modules.dockbot", "neorx.dockbot"),
    ("modules.mirrorfold", "neorx.mirrorfold"),
]


@pytest.mark.parametrize("old,new", SHIMMED)
def test_old_path_warns_and_forwards(old, new):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim = importlib.import_module(old)

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        f"{old} must emit DeprecationWarning"
    )
    assert new in [w.message.args[0] for w in caught][0]

    real = importlib.import_module(new)
    assert shim.__doc__ == real.__doc__ or shim is not None


def test_moved_submodule_dotted_import():
    """graph_builder moved under neorx.core.graph during the core subdivision."""
    import neorx.core.graph.graph_builder as real

    shim = importlib.import_module("modules.neorx.graph_builder")
    assert shim is real


def test_stayed_at_core_submodule_dotted_import():
    """pipeline did not move; it stays directly under neorx.core."""
    import neorx.core.pipeline as real

    shim = importlib.import_module("modules.neorx.pipeline")
    assert shim is real


def test_plain_prefix_swap_dotted_import():
    """genmol (and causalbiorl, molscreen, dockbot, mirrorfold) are a
    straight modules.X.* -> neorx.X.* prefix swap, no subdivision."""
    import neorx.genmol.data.tokenizer as real

    shim = importlib.import_module("modules.genmol.data.tokenizer")
    assert shim is real


def test_renamed_nested_package_dotted_import():
    """data_sources was renamed to sources during the core subdivision, and
    the rename must still work for names nested underneath it."""
    import neorx.core.sources.open_targets as real

    shim = importlib.import_module("modules.neorx.data_sources.open_targets")
    assert shim is real


def test_aliased_module_identity_via_sys_modules():
    """The shim must alias the SAME module object, not a star-imported copy,
    so isinstance/identity checks against the real module keep working."""
    import neorx.core.causal.identifier as real

    importlib.import_module("modules.neorx.identifier")
    assert sys.modules["modules.neorx.identifier"] is real


def test_dotted_submodule_import_still_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("modules.dockbot.protein_prep")

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "dotted submodule imports must also emit DeprecationWarning"
    )
    messages = [w.message.args[0] for w in caught]
    assert any("neorx.dockbot.protein_prep" in m for m in messages)
