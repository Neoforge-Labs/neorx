"""The CLI must import without development-only dependencies.

`neorx.cli` imports the experiments app, which imports the registry, which
imports `neorx.experiments.capture`. That module imported vcrpy at module
level, and vcrpy is declared only in the dev group -- so every command
failed on a clean install with `ModuleNotFoundError: No module named
'vcr'`, including the published 0.3.0 package.

The nightly end-to-end job had been failing on exactly this for days while
its alert said "Likely an upstream API change", which is why it went
unread: the message pointed away from the cause.
"""

import subprocess
import sys
import tomllib
from importlib.abc import MetaPathFinder
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _runtime_dependency_names():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    return {d.split(">=")[0].split("[")[0].split("==")[0].strip().lower() for d in deps}


def test_vcrpy_is_not_a_runtime_dependency():
    # If it ever becomes one, the lazy import below is pointless and this
    # test should be deleted deliberately rather than quietly passing.
    assert "vcrpy" not in _runtime_dependency_names()


def test_the_cli_imports_without_vcrpy():
    """A fresh interpreter, with vcr made unimportable.

    Run in a subprocess because `neorx.cli` is already imported in this
    one, and a module that is in sys.modules cannot demonstrate that it
    imports cleanly.
    """
    code = (
        "import sys\n"
        "from importlib.abc import MetaPathFinder\n"
        "class Block(MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'vcr' or name.startswith('vcr.'):\n"
        "            raise ModuleNotFoundError(\"No module named 'vcr'\")\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import neorx.cli\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_capture_refuses_loudly_when_vcrpy_is_missing():
    """Not applying the recorder is not the same as recording nothing.

    A run that asked for its HTTP to be frozen and silently did not freeze
    it is unreplayable and does not say so, so the failure is raised with
    the command that fixes it.
    """
    from neorx.experiments.capture import RecorderUnavailableError, _vcr

    class Block(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "vcr" or name.startswith("vcr."):
                # What Python itself raises for a missing module. The lazy
                # import catches that precisely, so a vcr that is
                # installed but broken is not reported as absent.
                raise ModuleNotFoundError("No module named 'vcr'")
            return None

    blocker = Block()
    sys.meta_path.insert(0, blocker)
    saved = {k: v for k, v in sys.modules.items() if k == "vcr" or k.startswith("vcr.")}
    for k in saved:
        del sys.modules[k]
    try:
        with pytest.raises(RecorderUnavailableError) as excinfo:
            _vcr()
        assert "uv sync --group dev" in str(excinfo.value)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_vcrpy_is_still_used_when_present():
    # The lazy import must return the real module, not a stand-in.
    import vcr

    from neorx.experiments.capture import _vcr

    assert _vcr() is vcr
