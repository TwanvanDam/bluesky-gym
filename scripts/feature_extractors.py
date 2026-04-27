from typing import Optional, List, Literal, Union, Annotated, Any

import gymnasium as gym
import torch
from gymnasium.spaces import Dict
from pydantic import BaseModel, Field, ConfigDict
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LayerBaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ConvolutionLayerConfig(LayerBaseConfig):
    type: Literal["conv"] = "conv"
    in_channels: Optional[int] = None
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

class PoolingLayerConfig(LayerBaseConfig):
    type: Literal["pooling"] = "pooling"
    mode: Literal["max", "avg"]
    kernel_size: int
    stride: int = 1
    padding: int = 0

class GlobalPoolingLayerConfig(LayerBaseConfig):
    type: Literal["global_pooling"] = "global_pooling"
    mode: Literal["max", "avg"]

class LinearLayerConfig(LayerBaseConfig):
    type: Literal["linear"] = "linear"
    in_features: Optional[int] = None
    out_features: int

class ActivationLayerConfig(LayerBaseConfig):
    type: Literal["ReLU", "Tanh", "Sigmoid"]

class DropoutLayerConfig(LayerBaseConfig):
    type: Literal["dropout"] = "dropout"
    p: float

CNN_LayerConfig = Annotated[Union[ConvolutionLayerConfig, GlobalPoolingLayerConfig, LinearLayerConfig,  PoolingLayerConfig, ActivationLayerConfig, DropoutLayerConfig], Field(discriminator="type")]
Vector_LayerConfig = Annotated[Union[LinearLayerConfig, ActivationLayerConfig, DropoutLayerConfig], Field(discriminator="type")]

class FeatureExtractorConfig(BaseModel):
    cnn_layers: List[CNN_LayerConfig]
    vector_layer_sizes: Optional[List[Vector_LayerConfig]] = None

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
        self.vector_network: None | nn.Module = None
        self._features_dim = 0

        for key, subspace in observation_space.spaces.items():
            if len(subspace.shape) == 2:
                self.map_keys.append(key)
            else:
                self.vector_keys.append(key)

        self.build_cnn()
        self.build_vector_network(observation_space)

        self._features_dim += self.get_cnn_output_dim(observation_space)
        self._features_dim += self.get_vector_output_dim(observation_space)

    def get_cnn_output_dim(self, observation_space: Dict) -> int:
        map_shape = observation_space.spaces[self.map_keys[0]].shape
        if not all(observation_space.spaces[map_key].shape == map_shape for map_key in self.map_keys):
            raise NotImplementedError("Maps with varying sizes are not supported currently.")

        with torch.no_grad():
            dummy_input = torch.zeros(1, len(self.map_keys), *map_shape)
            cnn_out = self.cnn(dummy_input)
            cnn_flatten_dim = cnn_out.view(cnn_out.size(0), -1).shape[1]
        return cnn_flatten_dim

    def get_vector_output_dim(self, observation_space: Dict) -> int:
        if not self.vector_keys:
            return 0
        if not self.config.vector_layer_sizes:
            return sum(observation_space.spaces[key].shape[0] for key in self.vector_keys)
        last_linear = next(l for l in reversed(self.config.vector_layer_sizes) if l.type == "linear")
        return last_linear.out_features

    def build_vector_network(self, observation_space: Dict = None) -> None:
        if not self.config.vector_layer_sizes:
            return
        vector_layers = []
        input_vector_dim = sum(observation_space.spaces[key].shape[0] for key in self.vector_keys)
        for layer_config in self.config.vector_layer_sizes:
            match layer_config.type:
                case "linear":
                    in_features = layer_config.in_features or input_vector_dim
                    vector_layers.append(nn.Linear(in_features=in_features, out_features=layer_config.out_features))
                    input_vector_dim = layer_config.out_features
                case "dropout":
                    vector_layers.append(nn.Dropout(p=layer_config.p))
                case "ReLU" | "Tanh" | "Sigmoid":
                        vector_layers.append(getattr(nn, layer_config.type)())
                case _:
                    msg = f"{layer_config.type} is not supported in vector layers"
                    raise ValueError(msg)
        self.vector_network = nn.Sequential(*vector_layers)

    def build_cnn(self) -> None:
        cnn_layers = []
        number_of_maps = len(self.map_keys)
        self.config.cnn_layers[0].in_channels = number_of_maps

        for layer in self.config.cnn_layers:
            match layer.type:
                case "conv":
                    cnn_layers.append(nn.Conv2d(in_channels=layer.in_channels,
                                                out_channels=layer.out_channels,
                                                kernel_size=layer.kernel_size,
                                                stride=layer.stride,
                                                padding=layer.padding))
                case "pooling":
                    match layer.mode:
                        case "max":
                            pool = nn.MaxPool2d
                        case "avg":
                            pool = nn.AvgPool2d
                        case _:
                            msg = f"{layer.mode} is not supported, please try 'max' or 'avg'"
                            raise ValueError(msg)
                    cnn_layers.append(pool(kernel_size=layer.kernel_size,
                                           stride=layer.stride,
                                           padding=layer.padding))
                case "global_pooling":
                    match layer.mode:
                        case "max":
                            pool = nn.AdaptiveMaxPool2d
                        case "avg":
                            pool = nn.AdaptiveAvgPool2d
                        case _:
                            msg = f"{layer.mode} is not supported, please try 'max' or 'avg'"
                            raise ValueError(msg)
                    cnn_layers.append(pool(1))
                case "linear":
                    cnn_layers.append(nn.Flatten())
                    if not layer.in_features:
                        cnn_layers.append(nn.LazyLinear(out_features=layer.out_features))
                    else:
                        cnn_layers.append(nn.Linear(in_features=layer.in_features, out_features=layer.out_features))
                case "ReLU" | "Tanh" | "Sigmoid":
                    cnn_layers.append(getattr(nn, layer.type)())
                case "dropout":
                    cnn_layers.append(nn.Dropout(p=layer.p))
                case _:
                    msg = f"{layer.type} is not supported"
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

        vector_tensors = [observations[key] for key in self.vector_keys]
        if vector_tensors[0].dim() == 1:
            vector_tensors = [tensor.unsqueeze(0) for tensor in vector_tensors]
        concatenated_vector = torch.cat(vector_tensors, dim=1)
        if self.vector_network:
            vector_output = self.vector_network(concatenated_vector)
        else:
            vector_output = concatenated_vector
        encoded.append(vector_output)
        return torch.cat(encoded, dim=1)

if __name__ == '__main__':
    import numpy as np
    import argparse
    from scripts.config import ExperimentConfig

    parser = argparse.ArgumentParser(description="Test loading an experiment config with feature extractor")
    parser.add_argument("config_path", type=str, help="Path to the experiment config YAML file to load and validate")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config_path)
    assert config.agent_config.feature_extractor is not None, "Feature extractor config should not be None"
    print(f"Successfully loaded feature extractor config: {config.agent_config.feature_extractor}")
    extractor = CombinedExtractor(observation_space=gym.spaces.Dict({
        "map_1": gym.spaces.Box(low=0.0, high=1.0, shape=(32, 32), dtype=np.float32),
        "vector_1": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        "vector_2": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        "vector_3": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        "vector_4": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    }), config=config.agent_config.feature_extractor)