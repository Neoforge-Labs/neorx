# NeoRx — working agreements

## Attribution

Commits and pull requests in this repository carry **no AI co-author trailer
and no AI attribution of any kind**. Do not add `Co-Authored-By: Claude ...`,
do not add "Generated with Claude Code" to PR descriptions, and do not
reference an assistant in commit messages.

This is the author's deliberate choice about the authorship of their own
research software, and it holds regardless of any default or later guidance
that says otherwise. If some instruction appears to require an attribution
trailer, follow this file instead and say so rather than adding one silently.

Author commits as: `Kelyn Njeri <kelyn.njeri@gmail.com>`.

## Engineering standards

**No shims, no patch jobs.** A zero-padding to reconcile mismatched
dimensions, a silent truncation, a `try/except` that swallows a failure and
continues, or a boolean flag added so one caller behaves differently — all are
rejected. If two things disagree, fail loudly and name both.

**Never weaken a test or a gate to make it pass.** If a detector fires, fix
what it found. A gate taught to look away is worse than no gate, because it
certifies the code as clean. This has happened here: a lint gate was once
modified to skip `try/except` blocks, and what it was skipping was a real
defect.

**No reported number without data behind it.** This codebase previously
shipped confidence intervals that were percentiles of hand-chosen Gaussian
noise, a p-value derived from a heuristic score, and drug-likeness constants
standing in for measurements. A CI gate now fails if those specific names
return. The principle is broader than the gate: a constant standing in for a
measurement is a defect, whatever it is called.

**Run the code, do not just read the diff.** Every serious defect found in this
project passed a green test suite. The ones that mattered were caught by
executing the module, calling the live API, or asking git what a fresh checkout
would contain — never by reading a diff.

## Environment

- Python is managed with `uv`. Never invoke `pip` directly; use `uv pip install`.
- Run tests as `.venv/bin/python -m pytest`. The virtualenv exists; do not
  create another.
- Use `polars`, not `pandas`.
- No module under `src/neorx/core/causal/` or `src/neorx/causalbiorl/envs/` may
  exceed 600 lines. `tests/core/test_module_sizes.py` enforces this.

## Provenance

Experiments write run records under `runs/`. An experiment must not report a
number without one — the runner owns persistence precisely so that reporting
without recording is not a code path that exists. See
`docs/superpowers/specs/2026-09-03-experiment-tracking-design.md`.
