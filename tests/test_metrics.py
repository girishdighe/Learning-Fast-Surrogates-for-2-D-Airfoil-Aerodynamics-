# tests/test_metrics.py
import torch
from src.metrics.field_metrics import compute_field_kpis

def test_field_kpis_shapes():
    B, H, W = 2, 8, 10
    y = torch.zeros(B, 3, H, W)
    yhat = torch.ones(B, 3, H, W) * 0.5
    kpis = compute_field_kpis(yhat, y, want_ssim=True)
    # Check presence of keys
    for k in ["rmse_u","mae_u","linf_u","rmse_v","rmse_p","rmse_Umag","ssim_Umag"]:
        assert any(k.startswith(prefix) for prefix in kpis.keys()) or k in kpis
