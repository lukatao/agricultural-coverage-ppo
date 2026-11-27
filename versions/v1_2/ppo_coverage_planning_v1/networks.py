"""Lightweight multi-branch CNN + MLP feature extractor for Stable-Baselines3 PPO.

Architecture:
  - local_map (5, H_loc, W_loc)  -> 3-layer strided CNN -> Linear -> 64-D (when enabled)
  - global_map (6, H_glob, W_glob) -> 3-layer strided CNN -> Linear -> 64-D (when enabled)
  - proprio + capabilities (23-D) -> 2-layer MLP -> 64-D
  - progress + obstacles (62-D)   -> 2-layer MLP -> 64-D
  Concatenated features (192-D or 256-D) feed SB3's Actor and Critic MLPs ([128, 128]).
"""
from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


def _make_map_cnn(in_channels: int, h: int, w: int, out_dim: int) -> nn.Sequential:
    conv = nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, h, w)
        flat_dim = int(conv(dummy).shape[1])
    return nn.Sequential(
        conv,
        nn.Linear(flat_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
    )


class AgCoverageFeaturesExtractor(BaseFeaturesExtractor):
    """Multi-modal CNN+MLP encoder for Dict observations from PPOCoverageEnv."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        local_dim: int = 64,
        global_dim: int = 64,
        state_dim: int = 64,
        sensor_dim: int = 64,
    ) -> None:
        spaces = observation_space.spaces
        self.has_local = "local_map" in spaces
        self.has_global = "global_map" in spaces
        self.has_global_untreated = "global_untreated" in spaces

        total_dim = state_dim + sensor_dim
        if self.has_local:
            total_dim += local_dim
        if self.has_global:
            total_dim += global_dim

        super().__init__(observation_space, features_dim=total_dim)

        if self.has_local:
            c, h, w = spaces["local_map"].shape
            self.local_cnn = _make_map_cnn(c, h, w, local_dim)
        else:
            self.local_cnn = None

        if self.has_global:
            c, h, w = spaces["global_map"].shape
            self.global_cnn = _make_map_cnn(c, h, w, global_dim)
        else:
            self.global_cnn = None

        prop_dim = int(spaces["proprio"].shape[0])
        cap_dim = int(spaces["capabilities"].shape[0])
        self.state_mlp = nn.Sequential(
            nn.Linear(prop_dim + cap_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
            nn.ReLU(),
        )

        prog_dim = int(spaces["progress"].shape[0])
        obs_dim = int(spaces["obstacles"].shape[0])
        cue_dim = int(spaces["global_untreated"].shape[0]) if self.has_global_untreated else 0
        self.sensor_mlp = nn.Sequential(
            nn.Linear(prog_dim + obs_dim + cue_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, sensor_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.local_cnn is not None:
            parts.append(self.local_cnn(observations["local_map"]))
        if self.global_cnn is not None:
            parts.append(self.global_cnn(observations["global_map"]))

        state_in = torch.cat([observations["proprio"], observations["capabilities"]], dim=1)
        parts.append(self.state_mlp(state_in))

        sensor_parts = [observations["progress"], observations["obstacles"]]
        if self.has_global_untreated:
            sensor_parts.append(observations["global_untreated"])
        sensor_in = torch.cat(sensor_parts, dim=1)
        parts.append(self.sensor_mlp(sensor_in))
        return torch.cat(parts, dim=1)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
