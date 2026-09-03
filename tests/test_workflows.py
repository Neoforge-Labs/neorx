"""Every filesystem path referenced in .github/workflows/*.yml must exist.

Workflow YAML isn't exercised by the rest of the test suite, so a path
that moves in a refactor (e.g. neorx/ -> src/neorx/) can go stale and
only fail the next time the workflow actually runs. This is a cheap
static guard against that class of defect.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Deliberately narrow: only matches tokens anchored to a known repo
# directory (with a literal "/", so it can't misfire on dotted Python
# module names like "neorx.core" or CLI invocations like "neorx run HIV"),
# or a whitelisted config-file extension.
PATH_TOKEN_RE = re.compile(
    r"\bsrc/neorx(?:/[\w./*-]*)?"
    r"|\bneorx/[\w./*-]+"
    r"|\btests/[\w./*-]*"
    r"|\bscripts/[\w./*-]+"
    r"|\b[\w.-]+\.(?:toml|cff)\b"
)

# Tokens that look like repo paths but legitimately aren't, at test time.
# Excluded explicitly (with reason) rather than loosening the regex.
EXCLUDED_TOKENS = {
    # publish.yml greps the *built wheel's* zip listing for this string.
    # hatchling's `packages = ["src/neorx"]` strips the "src/" prefix
    # when packaging, so inside the wheel the package lives at
    # "neorx/...", not "src/neorx/...". It's a wheel-internal path, not
    # a repo-relative one.
    "neorx/genmol/assets/molvae_chembl36.pt",
}


def _workflow_files():
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def _path_tokens(text):
    tokens = []
    for match in PATH_TOKEN_RE.finditer(text):
        token = match.group(0).rstrip("\"',.:;)")
        if token and token not in EXCLUDED_TOKENS:
            tokens.append(token)
    return tokens


def test_workflows_directory_has_files():
    # Guards the parametrization below from silently checking nothing.
    assert _workflow_files(), "no workflow files found under .github/workflows"


@pytest.mark.parametrize("workflow_path", _workflow_files(), ids=lambda p: p.name)
def test_workflow_paths_exist(workflow_path):
    tokens = _path_tokens(workflow_path.read_text())
    missing = [t for t in tokens if not (REPO_ROOT / t).exists()]
    assert not missing, f"{workflow_path.name} references missing paths: {missing}"
