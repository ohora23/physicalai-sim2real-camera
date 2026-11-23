#!/usr/bin/env bash
# Build the Axon SDK (third_party/axon) with Python bindings against this
# repo's venv. Idempotent: safe to re-run, reuses the existing build dir.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AXON_SRC="$REPO_ROOT/third_party/axon"
AXON_BUILD="$AXON_SRC/build"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

if [ ! -d "$AXON_SRC" ]; then
    echo "error: $AXON_SRC not found (clone the Axon SDK into third_party/axon first)" >&2
    exit 1
fi

# apt's pybind11-dev (2.11.1) predates NumPy 2.0 C-API support (landed in
# pybind11 2.12) — with numpy>=2 it silently corrupts dtype itemsize, which
# zeroes the strides on every TensorView.data returned to Python (a
# zero-stride "broadcast" view: reads always return byte 0). Use a
# venv-local pybind11>=2.12 instead of the system package.
if ! "$PYTHON_BIN" -c "import pybind11" >/dev/null 2>&1; then
    echo "installing pybind11>=2.12 into .venv (numpy 2.x compat)"
    uv pip install --python "$PYTHON_BIN" "pybind11>=2.12"
fi
PYBIND11_DIR="$("$PYTHON_BIN" -c "import pybind11; print(pybind11.get_cmake_dir())")"

CMAKE_ARGS=(
    -S "$AXON_SRC" -B "$AXON_BUILD"
    -DCMAKE_BUILD_TYPE=Release
    -DAXON_BUILD_PYTHON=ON
    -DPython3_EXECUTABLE="$PYTHON_BIN"
    -Dpybind11_DIR="$PYBIND11_DIR"
)

# Enable CUDA VMM accelerator pool only if the toolkit is present.
if grep -q "AXON_WITH_CUDA" "$AXON_SRC/CMakeLists.txt" && command -v nvcc >/dev/null 2>&1; then
    echo "CUDA toolkit found ($(command -v nvcc)); enabling AXON_WITH_CUDA"
    CMAKE_ARGS+=(-DAXON_WITH_CUDA=ON)
else
    echo "CUDA toolkit not found; building without AXON_WITH_CUDA"
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$AXON_BUILD" -j

echo "done. PYTHONPATH=$AXON_BUILD/python"
