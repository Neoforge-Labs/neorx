"""Every advertised name must import, from every subpackage."""

import importlib

import pytest

SUBPACKAGES = [
    "neorx",
    "neorx.core",
    "neorx.genmol",
    "neorx.causalbiorl",
    "neorx.molscreen",
    "neorx.dockbot",
    "neorx.mirrorfold",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_declares_all(name):
    mod = importlib.import_module(name)
    assert hasattr(mod, "__all__"), f"{name} must declare __all__"
    assert mod.__all__, f"{name}.__all__ must not be empty"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_every_exported_name_resolves(name):
    mod = importlib.import_module(name)
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, f"{name} exports names it does not define: {missing}"


def test_molscreen_exports_the_documented_screening_api():
    import neorx.molscreen as ms

    for fn in ("lipinski_filter", "qed_score", "pains_filter", "sa_score"):
        assert callable(getattr(ms, fn)), f"molscreen.{fn} must be callable"
