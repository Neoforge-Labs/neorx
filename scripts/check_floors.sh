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
uv pip install -q -r /tmp/floors.txt
uv pip install -q -e . --no-deps
python -m pytest -q tests/test_public_api.py tests/genmol/test_pretrained.py
