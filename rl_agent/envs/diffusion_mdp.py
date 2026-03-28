import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from rl_agent.wrappers.step_solvers import ddnm_step, diffpir_step, dps_step
from rl_agent.utils.rl_rewards import RewardCalculator


class DiffusionMDPEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Each step represents one denoising timestep going from t = T-1 down to 0.
    The agent chooses among DDNM (0), DiffPIR (1), and DPS (2).
    """

    def __init__(
        self,
        unet_model,
        spc_model,
        ddnm_params,
        diffpir_params,
        dps_params,
        device="cuda",
        noise_steps=1000,
        img_size=32,
    ):
        super(DiffusionMDPEnv, self).__init__()

        self.device = device
        self.unet_model = unet_model
        self.spc_model = spc_model
        
        self.ddnm_params = ddnm_params
        self.diffpir_params = diffpir_params
        self.dps_params = dps_params
        
        self.noise_steps = noise_steps
        self.img_size = img_size
        
        self.reward_calculator = RewardCalculator(device=self.device)

        # Action: 0=DDNM, 1=DiffPIR, 2=DPS
        self.action_space = spaces.Discrete(3)

        # Observation
        # Note: image_xt and image_y are back-projected views 
        self.observation_space = spaces.Dict(
            {
                "image_xt": spaces.Box(low=-np.inf, high=np.inf, shape=(3, img_size, img_size), dtype=np.float32),
                "image_y": spaces.Box(low=-np.inf, high=np.inf, shape=(3, img_size, img_size), dtype=np.float32),
                "time": spaces.Box(low=0, high=noise_steps, shape=(1,), dtype=np.float32),
            }
        )

        self.dataset = CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        
        self.current_t = self.noise_steps - 1
        self.current_xt = None
        self.current_y = None
        self.ground_truth = None
        self.back_projected_y = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             torch.manual_seed(seed)
             np.random.seed(seed)
             
        idx = np.random.randint(0, len(self.dataset))
        gt_img = self.dataset[idx][0].unsqueeze(0).to(self.device) # (1, 3, 32, 32)
        self.ground_truth = gt_img * 2.0 - 1.0 # normalize to [-1, 1]
        
        # Original measurement
        with torch.no_grad():
            self.current_y = self.spc_model.forward_pass(self.ground_truth)
            # DiffPIR adds noise to measurement if required, assuming noise_level_img=0 here as agent shouldn't add noise themselves
            self.back_projected_y = self.spc_model.transpose_pass(self.current_y)
        
        self.current_t = self.noise_steps - 1
        self.current_xt = torch.randn_like(self.ground_truth, device=self.device)

        return self._get_obs(), {}

    def _get_obs(self):
        obs = {
            "image_xt": self.current_xt.squeeze(0).cpu().numpy().astype(np.float32),
            "image_y": self.back_projected_y.squeeze(0).cpu().numpy().astype(np.float32),
            "time": np.array([self.current_t], dtype=np.float32)
        }
        return obs

    def step(self, action):
        old_xt = self.current_xt.clone()

        if action == 0:
            next_xt = ddnm_step(
                x_t=self.current_xt,
                y=self.current_y,
                t=self.current_t,
                model=self.unet_model,
                spc_model=self.spc_model,
                ddnm_params=self.ddnm_params
            )
        elif action == 1:
            next_xt = diffpir_step(
                x_t=self.current_xt,
                y=self.current_y,
                t=self.current_t,
                model=self.unet_model,
                spc_model=self.spc_model,
                diffpir_params=self.diffpir_params
            )
        elif action == 2:
            next_xt = dps_step(
                x_t=self.current_xt,
                y=self.current_y,
                t=self.current_t,
                model=self.unet_model,
                spc_model=self.spc_model,
                dps_params=self.dps_params
            )
        else:
             raise ValueError("Invalid action")

        # Step time
        self.current_t -= 1
        
        # Calculate Reward
        reward_tensor = self.reward_calculator.calculate_reward(self.current_xt, next_xt, self.ground_truth)
        reward = reward_tensor.item()
        
        self.current_xt = next_xt
        
        done = self.current_t < 0
        truncated = False
        
        info = {
            "action": action,
            "reward": reward
        }
        
        return self._get_obs(), reward, done, truncated, info

