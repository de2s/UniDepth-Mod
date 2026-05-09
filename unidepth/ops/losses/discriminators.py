import math
from abc import ABC, abstractmethod

from aim import Image, Distribution
import torch
import torch.nn as nn

from unidepth.utils.geometric import matrix_to_angles
from unidepth.utils.logger import getAimRun
from unidepth.utils.visualization import colorize
from unidepth.ops.losses.mlp import MLP


class AbstractTransformEstimator(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def getTransformParameters(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        log: bool = False,
        step: int = -1
    ) -> torch.Tensor:
        """This method estimates an equivariant transformation based on input transformation"""
        raise NotImplementedError
    

class TransformEstimatorSO3(AbstractTransformEstimator):

    def __init__(self, feature_dim, transform_generator_config):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = transform_generator_config

        # Create Transform Discriminator
        self.rotation_estimator = MLP(
            layer_sizes = [1024, 2048, 1024, 512, 3],
            init_weights = 0.0,
            init_std_devs = 1e-3,
            init_bias = 0.0
        )

    def getTransformParameters(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        log: bool = False,
        step: int = -1
    ):
        
        return self.rotation_estimator(torch.cat((feature0, feature1), dim=1))


class TransformEstimatorSE3(AbstractTransformEstimator):

    def __init__(self, feature_dim, transform_generator_config):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = transform_generator_config

        # Create Transform Discriminator
        self.rotation_estimator = MLP(
            layer_sizes = [1024, 2048, 1024, 512, 6],
            init_weights = 0.0,
            init_std_devs = 1e-3,
            init_bias = 0.0
        )

    def getTransformParameters(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        log: bool = False,
        step: int = -1
    ):
        
        return self.rotation_estimator(torch.cat((feature0, feature1), dim=1))
