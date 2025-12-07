import torch
import torch.nn as nn
from typing import Optional

from . import radon


class CTModel(nn.Module):
    """CT forward and backprojection operator with autograd-based adjoint."""

    def __init__(
        self,
        im_size: int,
        sampling_ratio: float,
        num_angles: int = 180,
        sampling_method: str = "uniform",
        mask: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
    ) -> None:
        super().__init__()

        self.im_size = im_size
        self.sampling_ratio = sampling_ratio
        self.num_angles = num_angles

        # Full angular grid; keep inclusive linspace to match prior behavior.
        full_angles = torch.linspace(0, 180, num_angles, dtype=torch.float32)
        self.theta = full_angles.to(device="cuda")

        # Build (or accept) the view mask
        if mask is not None:
            if mask.numel() != num_angles:
                raise ValueError("mask must have length equal to num_angles")
            self.mask = mask.float().to(device="cuda")
            self.apply_mask = apply_mask
        else:
            n_select = max(1, int(round(num_angles * sampling_ratio)))
            if sampling_method in ["uniform", "uniform_angle"]:
                indices = torch.linspace(0, num_angles - 1, n_select)
                indices = indices.round().long()
            elif sampling_method in ["non_uniform", "random"]:
                indices = torch.randperm(num_angles)[:n_select]
                indices, _ = torch.sort(indices)
            elif sampling_method in ["limited_angle", "limited"]:
                indices = torch.arange(n_select)
            else:
                raise ValueError("sampling_method must be 'uniform' or 'non_uniform'")
            mask_full = torch.zeros(num_angles, dtype=torch.float32)
            mask_full[indices] = 1.0
            self.mask = mask_full
            self.apply_mask = apply_mask

        # Forward operator (parallel-beam)
        self.radon_op = radon.Radon(
            in_size=im_size,
            theta=self.theta,
            circle=False,
            parallel_computation=True,
            device="cuda",
        )

        # Keep an FBP instance for quick reconstructions
        self.iradon_FBP = radon.IRadon(
            in_size=im_size,
            theta=self.theta,
            circle=False,
            use_filter=True,
            parallel_computation=True,
            device="cuda",
        )

    # ------------------------------------------------------------------
    def _mask_view(self, sino: torch.Tensor) -> torch.Tensor:
        """Apply the angular mask (diagonal operator M)."""
        if self.apply_mask:
            return sino * self.mask.view(1, 1, 1, -1).to(device=sino.device, dtype=sino.dtype)
        return sino

    # ------------------------------------------------------------------
    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward Radon transform (with view mask if enabled)."""
        sinogram = self.radon_op(x)
        return self._mask_view(sinogram)

    # ------------------------------------------------------------------
    def transpose_pass(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the *exact adjoint* (backprojection) of the *masked* forward operator.

        Forward is M A; adjoint is A^T M. Since M is diagonal, masking 'y' suffices.
        """
        y_masked = self._mask_view(y)
        B, C = y_masked.shape[:2]
        x_shape = (B, C, self.im_size, self.im_size)
        return radon.radon_adjoint_autograd(y_masked, self.radon_op, x_shape)

    # ------------------------------------------------------------------
    def FBP(self, y: torch.Tensor) -> torch.Tensor:
        """Filtered backprojection (analytic FBP) for convenience."""
        y_masked = self._mask_view(y)
        return self.iradon_FBP(y_masked)

    # ------------------------------------------------------------------
    def pseudoinverse_cgls(
        self,
        y: torch.Tensor,
        max_iter: int = 5,
        tol: float = 1e-6,
        x0: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ):
        """
        Solve x ≈ argmin ||M A x - M y||_2^2 via CGLS using autograd-based adjoint.
        This computes the Moore–Penrose action A^+ on masked data (if mask is enabled).
        """
        B, C = y.shape[:2]
        x_shape = (B, C, self.im_size, self.im_size)
        the_mask = self.mask if self.apply_mask else None

        if x0 is None:
            x0 = self.FBP(y)
        x, history = radon.cgls_pseudoinverse(
            b=y,
            radon_op=self.radon_op,
            x_shape=x_shape,
            max_iter=max_iter,
            tol=tol,
            x0=x0,
            verbose=verbose,
            mask=the_mask,
        )
        return x
