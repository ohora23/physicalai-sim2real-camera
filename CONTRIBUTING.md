# Contributing / development rules

These rules are the default for every change.

1. **Branch + PR per task.** No direct pushes to `main`. One feature branch per plan task
   (`r1-batch`, `fix-...`), a PR whose body carries the acceptance criteria, the test output and any
   bench table, a review by a second party, then merge.
2. **Tests first, unit tests included.** Every task ships tests in `tests/test_<module>.py` that
   (a) pin the acceptance criteria with explicit tolerances and (b) unit-test the pure functions it
   adds (formulas, parsers, packing helpers) independently of the integration path. Bit-exact or
   physics-based checks stay the standard for pipelines. `make cov` must not drop below the
   `fail_under` in `pyproject.toml` (90 % of statements + branches); new modules aim for ≥ 90 % too. The total
   relies on subprocess tracing (`patch = ["subprocess"]`): if that silently stops working, `dmabuf.py` and
   the example CLIs drop out and the total falls to ~81 %, so a sudden gate failure after a coverage or
   pytest-cov upgrade usually means that, not lost tests.
3. **Docstrings and comments.** Every public module, class, function and method has a Google-style
   docstring (ruff `D` rules enforce it); non-obvious numerics carry an inline comment with the
   reason or the measurement; deliberate simplifications are marked `# ponytail: ...` with the
   ceiling and the upgrade path.
4. **Lint gate.** `make lint` (ruff E/F/W/I/D) is clean before commit; `make fix` handles the
   mechanical part.
5. **Pre-push gate = CI.** GitHub-hosted runners have no GPU/EGL, so `.githooks/pre-push` runs ruff
   and the full suite with coverage locally (`git config core.hooksPath .githooks` once per clone).
   `SKIP_PREPUSH=1` only with a stated reason in the PR.
6. **Reproducibility.** Benches write CSVs under `out/` (gitignored) and the numbers that end up in
   docs name the GPU condition (idle vs shared).

Run: `make test`, `make cov`, `make lint`. Tests also run standalone: `MUJOCO_GL=egl .venv/bin/python tests/test_x.py`.
