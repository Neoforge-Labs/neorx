"""What ships, and what must never ship."""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out)],
        check=True,
        capture_output=True,
    )
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    return wheels[0]


def _names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def test_modules_is_never_packaged(wheel):
    # modules/ no longer exists anywhere in the repository (it was deleted
    # outright rather than deprecated), so this assertion is trivially true
    # today. Kept as a regression guard: it is what would catch the shim
    # being reintroduced, or a future `artifacts`/`include` entry in
    # pyproject.toml accidentally sweeping it back into the wheel.
    leaked = [n for n in _names(wheel) if n.startswith("modules/")]
    assert not leaked, f"the modules/ shim must not ship: {leaked}"


def test_genmol_weights_are_packaged(wheel):
    names = _names(wheel)
    assert "neorx/genmol/assets/molvae_chembl36.pt" in names, (
        "weights absent -- check the hatchling artifacts entry; .gitignore "
        "excludes *.pt from the build by default"
    )
    assert "neorx/genmol/assets/tokenizer.json" in names


def test_all_six_subpackages_are_packaged(wheel):
    names = _names(wheel)
    for pkg in ("core", "genmol", "causalbiorl", "molscreen", "dockbot", "mirrorfold"):
        assert any(n.startswith(f"neorx/{pkg}/") for n in names), f"missing {pkg}"
