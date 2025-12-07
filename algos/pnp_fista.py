import torch
from tqdm import tqdm
import math


class PnPFISTA:
    """Plug-and-Play FISTA with a deep denoiser."""

    def __init__(
        self,
        denoiser: torch.nn.Module,
        max_iter: int = 50,
        step_size: float = 1.0,
        denoiser_strength: float = 1.0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.denoiser = denoiser.to(device)
        self.max_iter = max_iter
        self.step_size = step_size
        self.denoiser_strength = denoiser_strength
        self.device = device

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the denoiser assuming it predicts noise."""
        with torch.inference_mode():
            noise = self.denoiser(x)
        return x - self.denoiser_strength * noise

    def sample(self, y: torch.Tensor, forward_pass, transpose_pass) -> torch.Tensor:
        """Run PnP-FISTA starting from the back-projection of ``y``."""
        x = transpose_pass(y)
        v = x.clone()
        t = 1.0

        pbar = tqdm(range(self.max_iter))
        for i in pbar:
            v = v.detach().requires_grad_(True)
            with torch.enable_grad():
                grad = transpose_pass(forward_pass(v) - y)
            grad = grad.detach()
            x_next = self.denoise((v - self.step_size * grad).detach())

            t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
            v = (x_next + ((t - 1) / t_next) * (x_next - x)).detach()

            diff = y - forward_pass(x)
            norm = torch.linalg.norm(diff)
            pbar.set_description(f"FISTA Step {i} - Distance: {norm.item():.4f}")

            x, t = x_next.detach(), t_next

        return x
