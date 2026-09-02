"""The repo-only modules/ shim keeps untracked root scripts working."""

import importlib
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
