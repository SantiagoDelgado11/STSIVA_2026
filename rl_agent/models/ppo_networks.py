import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class DiffusionFeatureExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for dict observations.
    Extracts features from spatial inputs (image_xt, image_y) using a CNN
    and continuous inputs (time) using an MLP.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We call the parent constructor specifying the final feature dimension
        super().__init__(observation_space, features_dim)
        
        # Extract shapes
        xt_shape = observation_space.spaces["image_xt"].shape
        y_shape = observation_space.spaces["image_y"].shape
        
        in_channels = xt_shape[0] + y_shape[0]
        
        # Image feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # eg. 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # eg. 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate resulting shape dynamically
        with torch.no_grad():
            dummy_xt = torch.zeros(1, *xt_shape)
            dummy_y = torch.zeros(1, *y_shape)
            dummy_img = torch.cat([dummy_xt, dummy_y], dim=1)
            n_flatten = self.cnn(dummy_img).shape[1]
            
        # Action embedding: 4 tokens translated to one-hot by SB3, projected to 16 dims
        self.action_embed = nn.Linear(4, 16)
        
        # Scalars embedding: Time (1) + Noise (1) + Var (1) + ActEmb (16) = 19
        self.scalars_mlp = nn.Sequential(
            nn.Linear(19, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combine image and scalar features
        self.fc = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # observations is a dictionary of tensors
        x_t = observations["image_xt"]
        y = observations["image_y"]
        t = observations["time"]
        noise_level = observations["noise_level"]
        img_var = observations["img_variance"]
        
        # SB3 automatically converts Discrete spaces in Dicts to one-hot floats
        # Re-allow flattening to counteract SB3's RolloutBuffer adding an extra dimension (batch, 1, 4) during train vs collect_rollouts
        act_emb = self.action_embed(observations["prev_action"]).flatten(start_dim=1)
        
        # Forward pass for spatial inputs
        x_cat = torch.cat([x_t, y], dim=1)
        cnn_features = self.cnn(x_cat)
        
        # Forward pass for scalars
        scalars = torch.cat([t, noise_level, img_var, act_emb], dim=1)
        scalar_features = self.scalars_mlp(scalars)
        
        # Combine and pass through final linear layer
        combined = torch.cat([cnn_features, scalar_features], dim=1)
        return self.fc(combined)

# Standard SB3 API allows you to just pass CustomFeatureExtractor to policy_kwargs 
# when doing MlpPolicy or MultiInputPolicy. We don't strictly need to define CustomActorCriticPolicy 
# unless we override the MLP builder itself, but we can provide it for clarity if wanted.
