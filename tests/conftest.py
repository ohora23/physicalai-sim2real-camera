"""Fixtures for test files whose ``__main__`` block builds objects the pytest functions also need.

``test_calib_photo.py`` and ``test_calibrate.py`` are written to run standalone
(``python tests/test_x.py``) with their ``__main__`` block constructing shared objects
(``stacks``/``dark_mean_lin``, a scratch ``tmp`` directory) and passing them positionally.
These fixtures build the same objects for the pytest path, reusing each file's own helpers
so both entry points stay in sync.
"""

import tempfile

import pytest


@pytest.fixture(scope="module")
def _ptc_data_cache():
    """Compute ``(stacks, dark_mean_lin)`` once per module via ``test_calib_photo._ptc_data``."""
    from test_calib_photo import _ptc_data
    return _ptc_data()


@pytest.fixture
def stacks(_ptc_data_cache):
    """PTC stacks: list of (n_frames, H, W, 3) uint8 arrays, one per illumination level."""
    return _ptc_data_cache[0]


@pytest.fixture
def dark_mean_lin(_ptc_data_cache):
    """Mean linear (post-gamma) dark-frame level, dimensionless in [0, 1]."""
    return _ptc_data_cache[1]


@pytest.fixture(scope="module")
def tmp():
    """Scratch directory shared by every test in the module, cleaned up after (like the file's __main__)."""
    with tempfile.TemporaryDirectory() as d:
        yield d
