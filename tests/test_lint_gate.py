"""The paths CI hard-gates must stay clean locally too.

CI fails the build if any whitelisted file is not ruff-clean and
ruff-formatted. Nothing enforced that before a push, so a formatting
regression in `src/neorx/experiments/record.py` went out three times in a
row -- three red builds for one wrapped line, each discovered only after
the push.

The whitelist is read from `.github/workflows/ci.yml` rather than copied
here. The workflow carries a comment telling whoever edits it to keep two
lists in sync by hand; a third hand-maintained copy in the test suite
would be the same hazard again, and this way the test cannot pass against
a whitelist CI no longer uses.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _gate_steps():
    """Every hard-gate ruff step in the workflow, as (job, name, files)."""
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    steps = []
    for job, spec in workflow["jobs"].items():
        for step in spec.get("steps", []):
            name = step.get("name", "")
            if "hard gate" not in name:
                continue
            run = step.get("run", "")
            files = [t for t in run.replace("\\", " ").split() if t.endswith(".py")]
            if files:
                steps.append((job, name, files))
    return steps


def test_the_workflow_still_has_hard_gates():
    # If the gates are renamed or removed, every test below would pass
    # vacuously while enforcing nothing.
    steps = _gate_steps()
    assert steps, "no hard-gate ruff steps found in ci.yml"
    assert any("format" in name for _job, name, _f in steps), (
        "no ruff FORMAT hard gate found; the check gate alone does not "
        "catch a reformatted line"
    )


def test_every_hard_gate_list_is_identical():
    """ci.yml says these lists are kept in sync by hand.

    Two jobs and two kinds of check share one whitelist. Drift between
    them means a file is gated in one job and not another, which reads as
    "gated" to anyone glancing at the workflow.
    """
    steps = _gate_steps()
    lists = {tuple(sorted(files)) for _job, _name, files in steps}
    assert len(lists) == 1, (
        "hard-gate whitelists have drifted:\n"
        + "\n".join(f"  {job} / {name}: {len(f)} files" for job, name, f in steps)
    )


def test_every_whitelisted_path_exists():
    # A path that no longer exists is silently not gated: ruff skips
    # missing files with an error but the step can still pass on others.
    _job, _name, files = _gate_steps()[0]
    missing = [f for f in files if not (REPO_ROOT / f).exists()]
    assert not missing, f"whitelisted paths do not exist: {missing}"


@pytest.mark.parametrize("mode", ["check", "format"])
def test_the_hard_gated_paths_are_clean(mode):
    """Run what CI runs, before the push rather than after it."""
    _job, _name, files = _gate_steps()[0]
    argv = [sys.executable, "-m", "ruff"]
    argv += ["check"] if mode == "check" else ["format", "--check"]
    result = subprocess.run(
        [*argv, *files], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"ruff {mode} fails on a CI hard-gated path -- this is what the "
        f"build will report:\n{result.stdout}{result.stderr}"
    )
