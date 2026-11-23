# Developer entry points (see CONTRIBUTING.md). GPU + EGL required for the full suite.
PY := .venv/bin/python
export MUJOCO_GL ?= egl
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

.PHONY: test lint cov fix
test:        ## full suite (pytest collection of tests/test_*.py)
	$(PY) -m pytest tests
cov:         ## suite + line/branch coverage (fails under 90 %)
	$(PY) -m pytest tests --cov --cov-report=term-missing:skip-covered --cov-report=html:out/htmlcov
lint:        ## ruff: errors, import order, docstrings (google style) on public API
	.venv/bin/ruff check physicalai_sim2real_camera tests examples scripts
fix:         ## ruff autofix (imports, trivial docstring formatting)
	.venv/bin/ruff check --fix physicalai_sim2real_camera tests examples scripts
