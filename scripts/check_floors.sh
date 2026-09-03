#!/usr/bin/env bash
# Install the declared dependency lower bounds and run the smoke suite.
# A floor that is wrong must fail here, not on a user's machine.
set -euo pipefail

python - <<'PY' > /tmp/floors.txt
import re, tomllib
with open("pyproject.toml", "rb") as fh:
    deps = tomllib.load(fh)["project"]["dependencies"]
for d in deps:
    m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)>=([0-9][^,;\s]*)", d)
    if m:
        print(f"{m.group(1)}=={m.group(2)}")
PY

echo "Installing declared floors:"
cat /tmp/floors.txt
# uv-managed environments (this repo's own venv included) have no pip.
# `uv pip install` needs an active virtualenv (VIRTUAL_ENV / ./.venv) unless
# told to target the system interpreter -- CI's `quality` job runs this
# script under actions/setup-python with `pip install -e .` and no venv, so
# fall back to --system there while still preferring the local .venv when
# one exists (e.g. running this script by hand).
if [ -z "${VIRTUAL_ENV:-}" ] && [ ! -d ./.venv ]; then
    export UV_SYSTEM_PYTHON=1
fi
uv pip install -q -r /tmp/floors.txt
uv pip install -q -e . --no-deps
python -m pytest -q tests/test_public_api.py tests/genmol/test_pretrained.py
