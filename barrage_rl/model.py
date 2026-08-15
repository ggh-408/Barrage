"""Barrage 像素智能体的轻量 CNN Actor-Critic 网络。"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


def _orthogonal_init(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """共享CNN提取画面特征，分别输出动作概率和状态价值。"""

    def __init__(
        self,
        frame_stack: int,
        observation_size: int,
        action_count: int,
        model_version: int = 3,
    ) -> None:
        super().__init__()
        self.model_version = int(model_version)
        encoder_input_channels = frame_stack
        if self.model_version == 1:
            # 仅用于加载和评估旧权重。
            convolution_layers = [
                _orthogonal_init(
                    nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4), 2 ** 0.5
                ),
                nn.ReLU(),
                _orthogonal_init(
                    nn.Conv2d(32, 64, kernel_size=4, stride=2), 2 ** 0.5
                ),
                nn.ReLU(),
                _orthogonal_init(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1), 2 ** 0.5
                ),
                nn.ReLU(),
            ]
        elif self.model_version == 2:
            # 先用较小卷积核保留微小目标，再逐级降采样。
            convolution_layers = [
                _orthogonal_init(
                    nn.Conv2d(frame_stack, 32, kernel_size=5, stride=2), 2 ** 0.5
                ),
                nn.ReLU(),
                _orthogonal_init(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2), 2 ** 0.5
                ),
                nn.ReLU(),
                _orthogonal_init(
                    nn.Conv2d(64, 64, kernel_size=3, stride=2), 2 ** 0.5
                ),
                nn.ReLU(),
            ]
        elif self.model_version == 3:
            # Add explicit frame differences, and extract full-resolution features
            # before downsampling so that tiny, fast targets remain visible.
            encoder_input_channels = frame_stack * 2 - 1
            convolution_layers = [
                _orthogonal_init(
                    nn.Conv2d(
                        encoder_input_channels,
                        32,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    ),
                    2 ** 0.5,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                _orthogonal_init(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2), 2 ** 0.5
                ),
                nn.ReLU(),
                _orthogonal_init(
                    nn.Conv2d(64, 64, kernel_size=3, stride=2), 2 ** 0.5
                ),
                nn.ReLU(),
            ]
        else:
            raise ValueError("不支持的模型版本: %d" % self.model_version)

        self.encoder = nn.Sequential(*convolution_layers, nn.Flatten())

        # 自动计算展平尺寸，避免为不同观测分辨率保存无意义参数
        with torch.no_grad():
            sample = torch.zeros(
                1, encoder_input_channels, observation_size, observation_size
            )
            feature_size = self.encoder(sample).shape[1]

        self.hidden = nn.Sequential(
            _orthogonal_init(nn.Linear(feature_size, 512), 2 ** 0.5),
            nn.ReLU(),
        )
        self.actor = _orthogonal_init(nn.Linear(512, action_count), 0.01)
        self.critic = _orthogonal_init(nn.Linear(512, 1), 1.0)

    def _features(self, observation: torch.Tensor) -> torch.Tensor:
        # 轨迹以uint8保存节约内存，只在进入GPU网络时归一化
        observation = observation.float().div_(255.0)
        if self.model_version == 3:
            frame_differences = observation[:, 1:] - observation[:, :-1]
            observation = torch.cat((observation, frame_differences), dim=1)
        return self.hidden(self.encoder(observation))

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(observation)).squeeze(-1)

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(observation)
        logits = self.actor(features)
        distribution = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = distribution.sample()
        log_probability = distribution.log_prob(action)
        entropy = distribution.entropy()
        value = self.critic(features).squeeze(-1)
        return action, log_probability, entropy, value
