import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Torch helpers
# ---------------------------------------------------------------------
if torch.__version__ > "1.2.0":
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode="bilinear": F.grid_sample(input, grid, align_corners=True, mode=mode)
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample


@torch.no_grad()
def _zeros_like_shape(shape, dtype, device):
    # helper that does NOT record grad history for the tensor creation itself
    return torch.zeros(shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------
# Autograd-based linear algebra hooks
# ---------------------------------------------------------------------
def radon_forward(x, radon_op):
    """A(x) using your Radon module."""
    return radon_op(x)


def radon_adjoint_autograd(v, radon_op, x_shape):
    """
    Exact adjoint A^T(v) computed via autograd for the *same* discrete operator as radon_op.

    Parameters
    ----------
    v        : sinogram-shaped tensor [B, C, N_det, N_ang]
    radon_op : instance of your Radon class
    x_shape  : [B, C, H, W]

    Returns
    -------
    A^T v with shape [B, C, H, W]
    """
    # Note: do NOT wrap this in torch.no_grad(); we need a local graph here.
    x_dummy = torch.zeros(x_shape, dtype=v.dtype, device=v.device, requires_grad=True)
    y = radon_op(x_dummy)  # y = A x_dummy
    (adjoint,) = torch.autograd.grad(
        outputs=y,
        inputs=x_dummy,
        grad_outputs=v,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    return adjoint


def _as_mask_view(mask, device, dtype):
    """Return mask shaped/broadcastable as [1,1,1,N_ang] or None."""
    if mask is None:
        return None
    if mask.ndim == 1:
        return mask.view(1, 1, 1, -1).to(device=device, dtype=dtype)
    # Try to broadcast as-is; fall back to last-dim view if possible
    if mask.ndim == 4:
        return mask.to(device=device, dtype=dtype)
    raise ValueError("mask must be 1-D length N_ang or broadcastable to [1,1,1,N_ang]")


def cgls_pseudoinverse(
    b,
    radon_op,
    x_shape,
    max_iter=50,
    tol=1e-6,
    x0=None,
    verbose=False,
    mask: torch.Tensor = None,
):
    """
    Compute x ≈ A^+ b by CGLS, using only A and A^T.
    If 'mask' is provided, we solve with the masked operator M A:
        minimize || M A x - M b ||_2^2.

    Parameters
    ----------
    b         : sinogram [B, C, N_det, N_ang]
    radon_op  : Radon operator A
    x_shape   : [B, C, H, W]
    max_iter  : iterations
    tol       : relative tolerance on ||A^T r||
    x0        : optional initial guess, same shape as x_shape
    verbose   : print progress
    mask      : (optional) 1-D [N_ang] or broadcastable to [1,1,1,N_ang]

    Returns
    -------
    x         : reconstruction [B, C, H, W]
    history   : dict with norms for monitoring
    """
    device, dtype = b.device, b.dtype
    M = _as_mask_view(mask, device, dtype)

    def apply_M(sino):
        return sino if M is None else sino * M

    # Initialize
    x = _zeros_like_shape(x_shape, dtype, device) if x0 is None else x0.clone().detach()

    # r0 = M b - M A x0
    Ax = apply_M(radon_forward(x, radon_op))
    r = apply_M(b) - Ax

    # s0 = A^T (M r0)  [since r is already masked, re-apply M for safety]
    s = radon_adjoint_autograd(apply_M(r), radon_op, x_shape)
    p = s.clone()
    gamma = (s * s).sum()

    history = {"norm_At_r": [gamma.sqrt().item()]}

    for it in range(max_iter):
        # qk = M A pk
        q = apply_M(radon_forward(p, radon_op))
        denom = (q * q).sum().clamp_min(1e-12)

        alpha = gamma / denom
        x = x + alpha * p
        r = r - alpha * q

        s = radon_adjoint_autograd(apply_M(r), radon_op, x_shape)
        gamma_new = (s * s).sum()

        rel = (gamma_new.sqrt() / history["norm_At_r"][0]).item()
        history["norm_At_r"].append(gamma_new.sqrt().item())
        if verbose:
            print(f"[CGLS] iter {it+1:02d}: ||A^T r|| = {gamma_new.sqrt().item():.4e} (rel {rel:.4e})")
        if rel < tol:
            break

        beta = gamma_new / gamma
        p = s + beta * p
        gamma = gamma_new

    return x, history


# ---------------------------------------------------------------------
# Geometry & filters (unchanged)
# ---------------------------------------------------------------------
def fan_beam_grid(theta, image_size, fan_parameters, dtype=torch.float, device="cpu"):
    scale_factor = 2.0 / (image_size * fan_parameters["pixel_spacing"])
    n_detector_pixels = fan_parameters["n_detector_pixels"]
    source_radius = fan_parameters["source_radius"] * scale_factor
    detector_radius = fan_parameters["detector_radius"] * scale_factor
    detector_spacing = fan_parameters["detector_spacing"] * scale_factor
    detector_length = detector_spacing * (n_detector_pixels - 1)

    R = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=dtype,
        device=device,
    )
    base_grid = affine_grid(R, torch.Size([1, 1, n_detector_pixels, image_size]))
    x_vals = base_grid[0, 0, :, 0]
    dist_scaling = 0.5 * detector_length * (x_vals + source_radius) / (source_radius + detector_radius)
    base_grid[:, :, :, 1] *= dist_scaling[None, None, :]
    base_grid = base_grid.reshape(-1, 2)
    rot_matrix = torch.tensor(
        [[theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]],
        dtype=dtype,
        device=device,
    )
    base_grid = base_grid @ rot_matrix.T
    base_grid = base_grid.reshape(1, n_detector_pixels, image_size, 2).transpose(1, 2)
    return base_grid


# constants
SQRT2 = (2 * torch.ones(1)).sqrt()


def fftfreq(n):
    val = 1.0 / n
    results = torch.zeros(n)
    N = (n - 1) // 2 + 1
    p1 = torch.arange(0, N)
    results[:N] = p1
    p2 = torch.arange(-(n // 2), 0)
    results[N:] = p2
    return results * val


def deg2rad(x):
    return x * 4 * torch.ones(1, device=x.device, dtype=x.dtype).atan() / 180


class AbstractFilter(nn.Module):
    def __init__(self, device="cpu", dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        is_3d = len(x.shape) == 5
        out = torch.empty_like(x)
        input_size = x.shape[dim]
        projection_size_padded = max(64, int(2 ** (2 * torch.tensor(input_size)).float().log2().ceil()))
        pad_width = projection_size_padded - input_size

        f = self._get_fourier_filter(projection_size_padded).to(x.device)
        fourier_filter = self.create_filter(f)
        if dim == 2 or dim == -2:
            fourier_filter = fourier_filter.unsqueeze(-1)

        if is_3d:
            B, C, H, A, N = x.shape
            for i in range(H):
                out[:, :, i] = self.filter(x[:, :, i], fourier_filter, pad_width, dim)
        else:
            out[:] = self.filter(x, fourier_filter, pad_width, dim)
        return out.contiguous()

    def filter(self, x: torch.Tensor, fourier_filter: torch.Tensor, pad_width: int, dim: int = 3) -> torch.Tensor:
        input_size = x.shape[dim]
        if dim == 3 or dim == -1:
            padded_tensor = F.pad(x, (0, pad_width, 0, 0))
        elif dim == 2 or dim == -2:
            padded_tensor = F.pad(x, (0, 0, 0, pad_width))

        projection = torch.fft.rfft(padded_tensor, dim=dim) * fourier_filter
        result = torch.fft.irfft(projection, dim=dim)

        if dim == 2 or dim == -2:
            return result[:, :, :input_size, :]
        elif dim == 3 or dim == -1:
            return result[:, :, :, :input_size]

    def _get_fourier_filter(self, size):
        n = torch.cat([torch.arange(1, size / 2 + 1, 2), torch.arange(size / 2 - 1, 0, -2)])
        f = torch.zeros(size, dtype=self.dtype, device=self.device)
        f[0] = 0.25
        f[1::2] = -1 / (torch.pi * n) ** 2
        fourier_filter = torch.fft.rfft(f, dim=-1)
        return 2 * fourier_filter

    def create_filter(self, f):
        raise NotImplementedError


class RampFilter(AbstractFilter):
    def __init__(self, **kwargs):
        super(RampFilter, self).__init__(**kwargs)

    def create_filter(self, f):
        return f


# ---------------------------------------------------------------------
# Radon / IRadon (unchanged numerics; we will not use IRadon for adjoint)
# ---------------------------------------------------------------------
class Radon(nn.Module):
    r"""Sparse Radon transform operator."""

    def __init__(
        self,
        in_size=None,
        theta=None,
        circle=False,
        parallel_computation=True,
        fan_beam=False,
        fan_parameters=None,
        dtype=torch.float,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.dtype = dtype
        self.parallel_computation = parallel_computation
        self.fan_beam = fan_beam
        self.fan_parameters = fan_parameters
        if fan_beam:
            if self.fan_parameters is None:
                self.fan_parameters = {}
            if "pixel_spacing" not in self.fan_parameters:
                assert in_size is not None, "Either input size or pixel spacing have to be given"
                self.fan_parameters["pixel_spacing"] = 0.5 / in_size
            if "source_radius" not in self.fan_parameters:
                self.fan_parameters["source_radius"] = 57.5
            if "detector_radius" not in self.fan_parameters:
                self.fan_parameters["detector_radius"] = 57.5
            if "n_detector_pixels" not in self.fan_parameters:
                self.fan_parameters["n_detector_pixels"] = 258
            if "detector_spacing" not in self.fan_parameters:
                self.fan_parameters["detector_spacing"] = 0.077
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle).to(device)
            if self.parallel_computation:
                self.all_grids_par = torch.cat([self.all_grids[i] for i in range(len(self.theta))], 2)

    def forward(self, x):
        N, C, W, H = x.shape
        assert W == H, "Input image must be square"

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle, device=x.device)
            if self.parallel_computation:
                self.all_grids_par = torch.cat([self.all_grids[i] for i in range(len(self.theta))], 2)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        if self.circle:
            yax = 2 * (torch.arange(W, dtype=torch.float, device=x.device)[None, :].expand(W, -1)[None, None, :, :]) / (W - 1) - 1.0
            xax = yax.transpose(-2, -1)
            mask = (xax**2 + yax**2 <= 1).to(torch.float)
            x = x * mask

        N, C, W, _ = x.shape

        if self.parallel_computation:
            rotated_par = grid_sample(x, self.all_grids_par.repeat(N, 1, 1, 1).to(x.device))
            out = rotated_par.sum(2).reshape(N, C, len(self.theta), -1).transpose(-2, -1)
        else:
            out = torch.zeros(
                N,
                C,
                self.all_grids[0].shape[-2],
                len(self.theta),
                device=x.device,
                dtype=self.dtype,
            )
            for i in range(len(self.theta)):
                rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
                out[..., i] = rotated.sum(2)
        return out

    def _create_grids(self, angles, grid_size, circle, device="cpu"):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            if self.fan_beam:
                all_grids.append(fan_beam_grid(theta, grid_size, self.fan_parameters, dtype=self.dtype, device=device))
            else:
                R = torch.tensor(
                    [[[theta.cos(), theta.sin(), 0], [-theta.sin(), theta.cos(), 0]]],
                    dtype=self.dtype,
                    device=device,
                )
                all_grids.append(affine_grid(R, torch.Size([1, 1, grid_size, grid_size])))
        return torch.stack(all_grids)


class IRadon(nn.Module):
    r"""Inverse sparse Radon transform operator (FBP when use_filter=True)."""

    def __init__(
        self,
        in_size=None,
        theta=None,
        circle=False,
        use_filter=True,
        out_size=None,
        parallel_computation=True,
        dtype=torch.float,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.circle = circle
        self.device = device
        self.theta = theta if theta is not None else torch.arange(180).to(self.device)
        self.out_size = out_size
        self.in_size = in_size
        self.parallel_computation = parallel_computation
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle).to(self.device)
            if self.parallel_computation:
                self.all_grids_par = torch.cat([self.all_grids[i] for i in range(len(self.theta))], 2)
        self.filter = RampFilter(dtype=self.dtype, device=self.device) if use_filter else lambda x: x

    def forward(self, x, filtering=True):
        it_size = x.shape[2]
        ch_size = x.shape[1]
        if self.in_size is None:
            self.in_size = int((it_size / SQRT2).floor()) if not self.circle else it_size
        if self.ygrid is None or self.xgrid is None or self.all_grids is None:
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)
            if self.parallel_computation:
                self.all_grids_par = torch.cat([self.all_grids[i] for i in range(len(self.theta))], 2)

        x = self.filter(x) if filtering else x

        if self.parallel_computation:
            reco = grid_sample(x, self.all_grids_par.repeat(x.shape[0], 1, 1, 1))
            reco = reco.reshape(x.shape[0], ch_size, it_size, len(self.theta), it_size)
            reco = reco.sum(-2)
        else:
            reco = torch.zeros(
                x.shape[0],
                ch_size,
                it_size,
                it_size,
                device=self.device,
                dtype=self.dtype,
            )
            for i_theta in range(len(self.theta)):
                reco += grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1))

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid**2 + self.ygrid**2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.0

        reco = reco * torch.pi / (2 * len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size) // 2
            reco = F.pad(reco, (pad, pad, pad, pad))
        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype, device=self.device)
        return torch.meshgrid(unitrange, unitrange, indexing="ij")

    def _XYtoT(self, theta):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin()
        return T

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for i_theta in range(len(angles)):
            X = torch.ones(grid_size, dtype=self.dtype, device=self.device).view(-1, 1).repeat(1, grid_size) * i_theta * 2.0 / (len(angles) - 1) - 1.0
            Y = self._XYtoT(angles[i_theta])
            all_grids.append(torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1).unsqueeze(0))
        return torch.stack(all_grids)


# Legacy: kept for compatibility, but no longer used for the adjoint
class ApplyRadon(torch.autograd.Function):
    @staticmethod
    def forward(x, radon, iradon, adjoint):
        if adjoint:
            return iradon(x, filtering=False) / torch.pi * (2 * len(iradon.theta))
        else:
            return radon(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.radon = inputs[1]
        ctx.iradon = inputs[2]
        ctx.adjoint = inputs[3]

    @staticmethod
    def backward(ctx, grad_output):
        return (
            ApplyRadon.apply(grad_output, ctx.radon, ctx.iradon, not ctx.adjoint),
            None,
            None,
            None,
        )
