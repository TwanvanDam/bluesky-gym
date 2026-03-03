import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from scripts.config import FeatureExtractorConfig


class CombinedExtractor(BaseFeaturesExtractor):
    """Expected observation_spaces:
    1D Vectors
    2D Maps
    Is able to be used with one observation or a batch.
    """
    def __init__(self, observation_space: gym.spaces.Dict, config: FeatureExtractorConfig):
        super().__init__(observation_space, features_dim=1)
        self.config = config
        self.map_keys = []
        self.vector_keys = []

        self.cnn: None | nn.Module = None

        for key, subspace in observation_space.spaces.items():
            if len(subspace.shape) == 2:
                self.map_keys.append(key)
            else:
                self.vector_keys.append(key)

        self.build_cnn(len(self.map_keys))

        map_shape = observation_space.spaces[self.map_keys[0]].shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, len(self.map_keys), *map_shape)
            cnn_out = self.cnn(dummy_input)
            cnn_flatten_dim = cnn_out.view(cnn_out.size(0), -1).shape[1]

        self._features_dim = cnn_flatten_dim + sum(observation_space.spaces[key].shape[0] for key in self.vector_keys)

    def build_cnn(self, number_of_maps: int) -> None:
        cnn_layers = []
        self.config.layers[0].conv.in_channels = number_of_maps

        for layer in self.config.layers:
            if layer.conv:
                cnn_layers.append(nn.Conv2d(in_channels=layer.conv.in_channels,
                                            out_channels=layer.conv.out_channels,
                                            kernel_size=layer.conv.kernel_size,
                                            stride=layer.conv.stride,
                                            padding=layer.conv.padding))
            if layer.pooling:
                match layer.pooling.type:
                    case "max":
                        pool = nn.MaxPool2d
                    case "avg":
                        pool = nn.AvgPool2d
                    case _:
                        msg = f"{layer.pooling.type} is not supported, please try 'max' or 'avg'"
                        raise ValueError(msg)
                cnn_layers.append(pool(kernel_size=layer.pooling.kernel_size,
                                       stride=layer.pooling.stride,
                                       padding=layer.pooling.padding))

            if layer.activation:
                match layer.activation:
                    case "ReLU":
                        cnn_layers.append(nn.ReLU())
                    case "Tanh" :
                        cnn_layers.append(nn.Tanh())
                    case "Sigmoid":
                        cnn_layers.append(nn.Sigmoid())
                    case _:
                        msg = f"{layer.activation} is not supported, please try 'ReLU', 'Tanh' or 'Sigmoid'"
                        raise ValueError(msg)
        self.cnn = nn.Sequential(*cnn_layers)

    def forward(self, observations: dict) -> torch.Tensor:
        encoded = []

        if self.map_keys:
            map_tensors = [observations[key] for key in self.map_keys]
            if map_tensors[0].dim() == 2:
                map_tensors = [tensor.unsqueeze(0).unsqueeze(0) for tensor in map_tensors]
            elif map_tensors[0].dim() == 3:
                map_tensors = [tensor.unsqueeze(1) for tensor in map_tensors]

            map_tensors = torch.cat(map_tensors, dim=1)
            cnn_output = self.cnn(map_tensors)
            encoded.append(torch.flatten(cnn_output, start_dim=1))

        for key in self.vector_keys:
            vector = observations[key]
            if vector.dim() == 1:
                vector = vector.unsqueeze(0)
            encoded.append(torch.flatten(vector, start_dim=1))

        return torch.cat(encoded, dim=1)