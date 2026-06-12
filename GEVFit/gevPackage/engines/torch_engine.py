# gevPackage/engines/torch.py
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def gev_nll_loss(weights, mu, sigma, xi, y):
    """
    PyTorch implementation of the GEV Negative Log-Likelihood.
    """
    # 0. Shape Handling
    if y.ndim == 1:
        y = y.view(-1, 1).expand_as(mu)

    # 1. Standardize (add epsilon to sigma)
    z = (y - mu) / (sigma + 1e-12)

    # 2. Gumbel Term
    nll_gumbel = torch.log(sigma + 1e-12) + z + torch.exp(-z)

    # 3. GEV Term (Stability Logic)
    # Replace xi ~ 0 with 1.0 to avoid div/0, then mask later
    xi_safe = torch.where(torch.abs(xi) < 1e-7, torch.ones_like(xi), xi)

    op_term = 1 + xi_safe * z
    valid_domain = op_term > 0
    
    # Mask inputs to log/power to prevent NaNs in gradients
    op_term_safe = torch.where(valid_domain, op_term, torch.ones_like(op_term))
    log_op = torch.log(op_term_safe)
    inv_xi = 1.0 / xi_safe
    
    t_gev = torch.exp(-inv_xi * log_op)
    nll_gev = torch.log(sigma + 1e-12) + (1.0 + inv_xi) * log_op + t_gev
    
    # Soft Infinity penalty
    penalty = 1e9
    nll_gev = torch.where(valid_domain, nll_gev, torch.tensor(penalty, device=y.device, dtype=y.dtype))

    # 4. Switch
    use_gumbel = torch.abs(xi) < 1e-5
    nll_point = torch.where(use_gumbel, nll_gumbel, nll_gev)

    # 5. Weighted Sum
    return torch.sum(weights * nll_point)

class GEVLoss(nn.Module):
    def forward(self, weights, mu, sigma, xi, y):
        return gev_nll_loss(weights, mu, sigma, xi, y)