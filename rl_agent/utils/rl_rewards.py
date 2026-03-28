import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class RewardCalculator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
    def normalize_tensor(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalizes UNet outputs from [-1, 1] to [0, 1] for torchmetrics.
        """
        img_norm = (img + 1.0) / 2.0
        return img_norm.clamp(0.0, 1.0)

    def calculate_reward(self, x_current: torch.Tensor, x_next: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Computes the incremental reward.
        Reward = (PSNR_next - PSNR_current) + lambda * (SSIM_next - SSIM_current)
        """
        with torch.no_grad():
            x_curr_norm = self.normalize_tensor(x_current)
            x_next_norm = self.normalize_tensor(x_next)
            gt_norm = self.normalize_tensor(ground_truth)
            
            psnr_curr = self.psnr(x_curr_norm, gt_norm)
            psnr_next = self.psnr(x_next_norm, gt_norm)
            
            ssim_curr = self.ssim(x_curr_norm, gt_norm)
            ssim_next = self.ssim(x_next_norm, gt_norm)
            
            delta_psnr = psnr_next - psnr_curr
            delta_ssim = ssim_next - ssim_curr
            
            # Giving SSIM a weight since its scale is [0, 1] while PSNR is usually [10, 40]
            # e.g., weight 50 scales a 0.02 SSIM improvement to 1.0. 
            reward = delta_psnr + 50.0 * delta_ssim
            
        return reward
