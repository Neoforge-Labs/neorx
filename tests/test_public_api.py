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


@pytest.mark.parametrize(
    "mod,names",
    [
        ("neorx.causalbiorl", ["DrugDiscoveryEnv", "CausalAgent"]),
        ("neorx.dockbot", ["prepare_protein", "prepare_ligand", "dock"]),
        ("neorx.mirrorfold", ["predict_pair", "compare_structures"]),
        ("neorx.molscreen", ["lipinski_filter", "qed_score", "pains_filter", "sa_score"]),
        ("neorx.genmol", ["MolVAE", "generate", "load_pretrained"]),
    ],
)
def test_spec_documented_entry_points_import(mod, names):
    """A name resolving because it happens to be in __all__ is not the same
    as the spec's documented quick-start actually working. This checks the
    literal usage examples from the design spec, not just internal
    consistency of each module's own __all__ list."""
    m = importlib.import_module(mod)
    missing = [n for n in names if not hasattr(m, n)]
    assert not missing, f"{mod} does not export documented API: {missing}"
