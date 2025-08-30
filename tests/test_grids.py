# tests/test_grids.py
import numpy as np
from src.utils.grids import make_uniform_grid, flatten_grid, mask_from_sdf

def test_make_uniform_grid_and_flatten():
    X, Y = make_uniform_grid(0.0, 1.0, -1.0, 1.0, 5, 3)
    assert X.shape == (3, 5) and Y.shape == (3, 5)
    pts = flatten_grid(X, Y)
    assert pts.shape == (3*5, 2)
    # simple mask from sdf
    sdf = Y.copy()
    m = mask_from_sdf(sdf, inside_is_fluid=True)
    assert m.shape == Y.shape
    assert m.dtype == np.float32
