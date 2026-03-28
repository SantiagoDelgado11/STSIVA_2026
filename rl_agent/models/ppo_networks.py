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
            
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combine image and time features
        self.fc = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # observations is a dictionary of tensors
        x_t = observations["image_xt"]
        y = observations["image_y"]
        t = observations["time"]
        
        # Forward pass for spatial inputs
        x_cat = torch.cat([x_t, y], dim=1)
        cnn_features = self.cnn(x_cat)
        
        # Forward pass for time
        time_features = self.time_mlp(t)
        
        # Combine and pass through final linear layer
        combined = torch.cat([cnn_features, time_features], dim=1)
        return self.fc(combined)

# Standard SB3 API allows you to just pass CustomFeatureExtractor to policy_kwargs 
# when doing MlpPolicy or MultiInputPolicy. We don't strictly need to define CustomActorCriticPolicy 
# unless we override the MLP builder itself, but we can provide it for clarity if wanted.
