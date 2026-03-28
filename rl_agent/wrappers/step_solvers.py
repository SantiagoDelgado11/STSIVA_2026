import torch
import numpy as np
from typing import Callable, Dict, Any

from algos.ddnm import DDNM
from algos.diffpir import DiffPIR, conjugate_gradient
from algos.dps import DPS
from utils.SPC_model import SPCModel


def ddnm_step(x_t: torch.Tensor, y: torch.Tensor, t: int, model: torch.nn.Module, spc_model: SPCModel, ddnm_params: DDNM) -> torch.Tensor:
    """
    Executes a single step of the DDNM algorithm.
    """
    device = x_t.device
    t_tensor = torch.tensor([t], dtype=torch.long, device=device)
    alpha_hat = ddnm_params.alpha_hat[t_tensor].view(-1, 1, 1, 1)
    sqrt_alpha_hat = torch.sqrt(alpha_hat)
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
    alpha_hat_prev = ddnm_params.alpha_hat_prev[t_tensor].view(-1, 1, 1, 1)

    A_psudo_inverse_y = spc_model.pseudo_inverse(y)

    with torch.no_grad():
        predicted_noise = model(x_t, t_tensor)

    x_0_t = (x_t - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
    X_hat_0_t = x_0_t - spc_model.pseudo_inverse(spc_model.forward_pass(x_0_t)) + A_psudo_inverse_y

    c1 = (1 - alpha_hat_prev).sqrt() * ddnm_params.eta
    c2 = (1 - alpha_hat_prev).sqrt() * ((1 - ddnm_params.eta**2) ** 0.5)

    x_t_minus_1 = torch.sqrt(alpha_hat_prev) * X_hat_0_t + c1 * torch.randn_like(x_t) + c2 * predicted_noise

    return x_t_minus_1.detach()


def diffpir_step(x_t: torch.Tensor, y: torch.Tensor, t: int, model: torch.nn.Module, spc_model: SPCModel, diffpir_params: DiffPIR) -> torch.Tensor:
    """
    Executes a single step of the DiffPIR algorithm.
    Assuming skip_type = "uniform", iter_num = noise_steps, so no skipping.
    """
    device = x_t.device
    t_step = t
    vec_t = torch.tensor([t_step] * x_t.shape[0], device=device)
    
    # In diffpir, "i" goes from 0 to noise_steps - 1.
    # Where i = noise_steps - 1 - t
    i = diffpir_params.noise_steps - 1 - t
    
    # Calculate sigmas and rhos as diffpir does at initialization
    sigmas = []
    sigma_ks = []
    rhos = []
    for j in range(diffpir_params.noise_steps):
        sigmas.append(diffpir_params.reduced_alpha_cumprod[diffpir_params.noise_steps - 1 - j])
        sigma_ks.append((diffpir_params.sqrt_1m_alphas_cumprod[j] / diffpir_params.sqrt_alphas_cumprod[j]))
        rhos.append(diffpir_params.lambda_ * (diffpir_params.sigma**2) / (sigma_ks[j] ** 2))

    rhos = torch.tensor(rhos, device=device)
    sigmas = torch.tensor(sigmas, device=device)
    
    with torch.no_grad():
        model_out = diffpir_params.p_sample(x_t, vec_t, model)
        z = model_out["pred_xstart"]

    rho_t = rhos[i]
    b = spc_model.transpose_pass(y) + rho_t * z

    def A_fn(v):
        return spc_model.transpose_pass(spc_model.forward_pass(v)) + rho_t * v

    if i < diffpir_params.noise_steps - 1:
        # PPO requires no_grad context broadly, but Conjugate Gradient needs gradient math? 
        # Wait, CG in diffpir uses PyTorch tensor ops without autograd if it's not tracing. Diffpir uses .detach()
        with torch.enable_grad():
            x0 = conjugate_gradient(A_fn, b, x0=z, n_iter=diffpir_params.cg_iters).detach()
    else:
        x0 = z

    if i < diffpir_params.noise_steps - 1:
        t_im1 = t - 1
        if t_im1 < 0:
            t_im1 = 0
            
        eps = (x_t - diffpir_params.sqrt_alphas_cumprod[t] * x0) / diffpir_params.sqrt_1m_alphas_cumprod[t]
        eta_sigma = (
            diffpir_params.eta
            * diffpir_params.sqrt_1m_alphas_cumprod[t_im1]
            / diffpir_params.sqrt_1m_alphas_cumprod[t]
            * torch.sqrt(diffpir_params.beta[t])
        )
        x_t_minus_1 = (
            diffpir_params.sqrt_alphas_cumprod[t_im1] * x0
            + np.sqrt(1 - diffpir_params.zeta)
            * (
                torch.sqrt(diffpir_params.sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma**2) * eps
                + eta_sigma * torch.randn_like(x_t)
            )
            + np.sqrt(diffpir_params.zeta) * diffpir_params.sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x_t)
        )
    else:
        x_t_minus_1 = x_t # Or final sample, but usually done at i==noise_steps-1

    return x_t_minus_1.detach()


def dps_step(x_t: torch.Tensor, y: torch.Tensor, t: int, model: torch.nn.Module, spc_model: SPCModel, dps_params: DPS) -> torch.Tensor:
    """
    Executes a single step of the DPS algorithm.
    """
    device = x_t.device
    
    # Make a copy of x_t that requires grad
    x_prev = x_t.detach().clone().requires_grad_(True)
    time_tensor = torch.tensor([t] * x_prev.shape[0], device=device)
    
    # Forward pass through UNet doesn't need grad for model weights, but needs it for x_prev
    # Actually, DPS needs grad wrt x_prev, so model weights shouldn't get grad
    # Pytorch requires_grad behavior allows this if model is eval and its params False
    out = dps_params.p_sample(x=x_prev, t=time_tensor, model=model)
    
    # Conditioning
    norm_grad, _ = dps_params.grad_and_value(
        x_prev=x_prev, 
        x_0_hat=out["pred_xstart"], 
        measurement=y, 
        forward_pass=spc_model.forward_pass
    )
    
    x_t_minus_1 = out["sample"] - norm_grad * dps_params.scale

    return x_t_minus_1.detach()
