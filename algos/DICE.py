import torch
from tqdm import tqdm
from utils.ddpm import get_named_beta_schedule
import numpy as np
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)


class DICE:
    """Diffusion implicit consistent equilibrium (DICE) sampler."""

    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
        schedule_name="cosine",
        channels=1,
        rho=0.9,
        mu=0.1,
        skip_type="uniform",
        iter_num=1000,
    ):
        """Initialize the sampler.

        Parameters
        ----------
        noise_steps : int
            Number of diffusion steps.
        beta_start, beta_end : float
            Noise schedule range.
        img_size : int
            Spatial size of the image.
        device : str
            Device on which to run.
        schedule_name : str
            Noise schedule type.
        channels : int
            Number of image channels.
        rho : float
            Rho
        mu : float
            Relaxation parameter for the outer update.
        skip_type : str
            Strategy to subsample diffusion steps.
        iter_num : int
            Number of diffusion iterations.
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule_name = schedule_name
        self.channels = channels

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.skip_type = skip_type
        self.iter_num = iter_num
        self.skip = self.noise_steps // self.iter_num if self.iter_num else 1
        self.rho = rho
        self.mu = mu
        self.alpha_hat_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_hat[:-1]])

    def prepare_noise_schedule(self):
        if self.schedule_name == "cosine":
            return torch.tensor(
                get_named_beta_schedule("cosine", self.noise_steps, self.beta_end).copy(),
                dtype=torch.float32,
            )

        if self.schedule_name == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _make_schedule(self):
        if not self.iter_num:
            return list(range(self.noise_steps - 1, 0, -1))
        if self.skip_type == "uniform":
            seq = [i * self.skip for i in range(self.iter_num)]
            if self.skip > 1:
                seq.append(self.noise_steps - 1)
        elif self.skip_type == "quad":
            seq = np.sqrt(np.linspace(0, self.noise_steps**2, self.iter_num))
            seq = [int(s) for s in list(seq)]
            seq[-1] = seq[-1] - 1
        else:
            seq = list(range(self.noise_steps))
        return sorted(set(seq), reverse=True)

    def sample(
        self,
        model,
        y,
        transpose_pass,
        forward_pass,
        CG_iter,
        CE_iter,
        ground_truth=None,
        track_metrics=False,
        track_consensus=False,
    ):

        x = torch.randn((1, self.channels, self.img_size, self.img_size)).to(self.device)

        seq = self._make_schedule()
        pbar = tqdm(seq, position=0)

        metrics = {
            "psnr": [],
            "ssim": [],
            "consistency": [],
            "disagreement": [],
            "fp_error": [],
            "fp_error_history": [],
        }

        if track_metrics and ground_truth is not None:
            SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            PSNR = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        for i in pbar:
            t = (torch.ones(1) * i).long().to(self.device)
            W = torch.stack([x.clone().detach(), x.clone().detach()], dim=0)
            MU = [self.mu, 1 - self.mu]

            fp_err = None
            step_history = []
            for K in range(CE_iter):
                prev_w = W.clone()
                W_prima = 2 * self.F_func(W, forward_pass, transpose_pass, y, CG_iter, i, model) - W
                W_prima = 2 * self.G_func(W_prima, MU) - W_prima
                W = (1 - self.rho) * W + self.rho * W_prima
                fp_err = torch.linalg.norm(W - prev_w)
                if track_consensus:
                    step_history.append(fp_err.item())
            if track_consensus:
                metrics["fp_error_history"].append(step_history)

            x0 = W[0] * MU[0] + W[1] * MU[1]

            ############################################################

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            alpha_hat_prev = self.alpha_hat_prev[t]
            x = torch.sqrt(alpha_hat_prev) * x0 + torch.sqrt(1 - alpha_hat_prev) * noise
            ############################################################

            difference = y - forward_pass(x)
            norm = torch.linalg.norm(difference)

            if track_metrics and ground_truth is not None:
                with torch.inference_mode():
                    x_eval = (x + 1) / 2
                    gt_eval = (ground_truth + 1) / 2
                    x_eval = x_eval.clamp(0, 1)
                    gt_eval = gt_eval.clamp(0, 1)
                    psnr_val = PSNR(x_eval, gt_eval).item()
                    ssim_val = SSIM(x_eval, gt_eval).item()
                devices = []
                if isinstance(self.device, str) and self.device.startswith("cuda"):
                    devices = [torch.device(self.device)]
                with torch.random.fork_rng(devices=devices):
                    with torch.enable_grad():
                        F_out = self.F_func(W, forward_pass, transpose_pass, y, CG_iter, i, model)
                    F_out = F_out.detach()
                    G_out = self.G_func(W, MU)
                disagree = torch.linalg.norm(F_out - G_out).item()
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["consistency"].append(norm.item())
                metrics["disagreement"].append(disagree)
                metrics["fp_error"].append(fp_err.item())

            pbar.set_description(f"Sampling - Step {i} - Consistency: {norm.item():.4f} - FP err: {fp_err.item():.4f}")

        if track_metrics and ground_truth is not None:
            return x, metrics
        return x

    def F_func(self, W, forward_pass, transpose_pass, y, CG_iter, i, model):
        return torch.stack(
            [
                self.F_t(forward_pass, transpose_pass, y, CG_iter, W[0], i),
                self.H_t(model, i, W[1]),
            ],
            dim=0,
        )

    def G_func(self, W, MU):
        return torch.stack(
            [
                W[0] * MU[0] + W[1] * MU[1],
                W[0] * MU[0] + W[1] * MU[1],
            ],
            dim=0,
        )

    def F_t(self, forward_pass, transpose_pass, y, itera, x, i):
        x = x.clone()
        t = (torch.ones(1) * i).long().to(self.device)
        lambda_t = float((1 - self.alpha_hat[t]) / self.alpha_hat[t])

        def A_fn(u):
            return transpose_pass(forward_pass(u)) + u * lambda_t

        b = transpose_pass(y) + x * lambda_t

        x = conjugate_gradient(A_fn, b, x0=x, n_iter=itera).detach()

        return x

    def H_t(self, model, i, x):
        model.eval()
        with torch.no_grad():
            t = torch.tensor([i], device=self.device).long()
            predicted_noise = model(x, t)

            alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
            sqrt_alpha_hat = torch.sqrt(alpha_hat)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

            x = (x - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
            x = x.clamp(-1.0, 1.0)

        return x


def conjugate_gradient(
    apply_A,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    n_iter: int = 40,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Early‑stopping batch CG that preserves autograd for ``apply_A``."""

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - apply_A(x)
    p = r.clone()
    rs_old = torch.sum(r * r, dim=list(range(1, r.ndim)), keepdim=True)

    for _ in range(n_iter):
        Ap = apply_A(p)
        alpha = rs_old / (torch.sum(p * Ap, dim=list(range(1, r.ndim)), keepdim=True) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r, dim=list(range(1, r.ndim)), keepdim=True)
        if torch.sqrt(rs_new.mean()) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x
