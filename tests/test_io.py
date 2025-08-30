# tests/test_io.py
from pathlib import Path
import numpy as np
from src.utils.io_airfrans import infer_bbox_from_coords

def test_infer_bbox_from_coords():
    coords = np.array([[0.0, 1.0], [2.0, -1.0], [1.0, 0.5]])
    xmin, xmax, ymin, ymax = infer_bbox_from_coords(coords)
    assert xmin == 0.0 and xmax == 2.0
    assert ymin == -1.0 and ymax == 1.0
