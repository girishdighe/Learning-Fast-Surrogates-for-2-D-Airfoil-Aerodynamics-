# tests/test_conformal.py
import torch
from src.conformal.icp_scalar import fit_icp_scalar, predict_interval_scalar

def test_icp_scalar_global():
    torch.manual_seed(0)
    y = torch.randn(100)
    yhat = y + 0.1*torch.randn(100)
    icp = fit_icp_scalar(y, yhat, target_cov=0.9)
    lo, hi = predict_interval_scalar(yhat, icp)
    cover = ((y >= lo) & (y <= hi)).float().mean().item()
    assert 0.8 <= cover <= 1.0  # crude bound
