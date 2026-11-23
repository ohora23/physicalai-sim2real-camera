"""R3: pure helpers of scripts/bench_batch.py (no GPU) + one tiny GPU smoke case (skipped without CUDA)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from bench_batch import COLS, csv_row, markdown_table, median_row, r1_mem_bound

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False

ROW = dict(n=16, width=640, height=480, depth=1, ring=4, render_into_ms=3.14159, process_ms=2.5, serial_ms=5.64159,
           pipelined_ms=4.8, ms_per_camera=0.3, alloc_mib=123.456, peak_mib=200.04)
REF = dict(n=1, width=640, height=480, depth=1, ring=0, render_into_ms=None, process_ms=None, serial_ms=0.8,
           pipelined_ms=0.6, ms_per_camera=0.6, alloc_mib=10.0, peak_mib=20.0)


def test_r1_mem_bound():
    # by hand: per = 10*20*7 + 4*5*6*3 + 5*6*4 = 1400 + 360 + 120 = 1880; 1.1 * (1000 + 2*1880) = 5236
    assert r1_mem_bound(1000, 2, 10, 20, 5, 6, 4) == pytest.approx(5236)
    # depth adds 4 B/px raw Z + ring*4 B/px depth ring: 800 + 480 = 1280 per camera
    assert r1_mem_bound(1000, 2, 10, 20, 5, 6, 4, depth=True) == pytest.approx(1.1 * (1000 + 2 * 3160))
    # earlier measurement: n=64 640x480 (render 764x574), ring 1 era: 337.4 + 64 x 4.98 MiB -> 721.6 MiB
    per = 574 * 764 * 7 + 480 * 640 * 7
    assert per / 2**20 == pytest.approx(4.98, abs=0.01)
    assert r1_mem_bound(337.4 * 2**20, 64, 574, 764, 480, 640, 1) / 2**20 == pytest.approx(721.6, abs=0.2)
    print("T1 r1_mem_bound: hand numbers, depth terms, R1 721.6 MiB")


def test_csv_row():
    line = csv_row(ROW)
    assert line == "16,640,480,1,4,3.142,2.500,5.642,4.800,0.300,123.5,200.0,"  # no window key -> empty last field
    assert len(line.split(",")) == len(COLS)
    assert csv_row(REF) == "1,640,480,1,0,,,0.800,0.600,0.600,10.0,20.0,"  # None -> empty field
    assert csv_row(dict(ROW, window=1)).endswith(",123.5,200.0,1")  # window column last, as an int
    print("T2 csv_row: column order, 3/1 decimals, None -> empty")


def test_markdown_table():
    t = markdown_table([REF, ROW]).splitlines()
    assert len(t) == 4 and t[1] == "|" + "---|" * 14
    assert all(line.count("|") == 15 for line in t)
    assert t[2].startswith("| 1 (SimCamera) | 640×480 | yes | – | – | – | – | 0.80 | 0.60 | 0.600 | 1.00 | 1.00 |")
    # pipelined 4.8/16 / 0.6 = 0.50; serial 5.64159/16 / 0.8 = 0.44
    assert "| 4.80 | 0.300 | 0.50 | 0.44 |" in t[3]
    lone = markdown_table([ROW]).splitlines()[2]  # no reference at that resolution/depth
    assert "| 0.300 | – | – |" in lone
    # a window row compares only against the window reference, and shows its window config
    win = markdown_table([REF, dict(ROW, window=1)]).splitlines()[3]
    assert "| N4B8 |" in win and "| 0.300 | – | – |" in win
    print("T3 markdown_table: reference row, batching ratio, missing reference")


@pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
def test_bench_smoke_cuda():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from bench_batch import bench_batch, bench_single
    row, (cb, h_r, w_r) = bench_batch(2, 640, 480, True, 1, steps=1)
    assert row["n"] == 2 and row["ring"] == 1 and row["alloc_mib"] > 0
    assert row["serial_ms"] > 0 and row["pipelined_ms"] > 0
    assert row["alloc_mib"] * 2**20 <= r1_mem_bound(cb, 2, h_r, w_r, 480, 640, 1, depth=True)
    ref = bench_single(640, 480, False, steps=1)
    assert ref["ring"] == 0 and ref["serial_ms"] > 0 and ref["pipelined_ms"] > 0
    print(f"T4 smoke: n=2 depth ring=1 {row['pipelined_ms']:.2f} ms/step, {row['alloc_mib']:.1f} MiB")


def test_median_row():
    rows = [dict(REF, serial_ms=1.0, pipelined_ms=3.0, alloc_mib=1.0),
            dict(REF, serial_ms=2.0, pipelined_ms=1.0, alloc_mib=2.0),
            dict(REF, serial_ms=9.0, pipelined_ms=2.0, alloc_mib=3.0)]
    m = median_row(rows)
    assert m["serial_ms"] == 2.0 and m["pipelined_ms"] == 2.0 and m["alloc_mib"] == 3.0  # medians; memory: last run
    assert m["render_into_ms"] is None  # None columns stay None
    print("T4 median_row: timing medians, last-run memory, None kept")


if __name__ == "__main__":
    test_r1_mem_bound()
    test_csv_row()
    test_markdown_table()
    if HAVE_CUDA:
        test_bench_smoke_cuda()
    print("OK")
    test_median_row()
