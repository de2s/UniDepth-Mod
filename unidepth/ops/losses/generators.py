import math
from abc import ABC, abstractmethod

from aim import Image, Distribution
import torch
import torch.nn as nn

from unidepth.utils.geometric import matrix_to_angles
from unidepth.utils.logger import getAimRun
from unidepth.utils.visualization import colorize
from unidepth.ops.losses.monotonic_mlp import MonotonicMLP


class AbstractGenerator(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def getTransformation(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        log: bool = False,
        step: int = -1
    ) -> torch.Tensor:
        """This method estimates an equivariant transformation based on input transformation"""
        raise NotImplementedError
    

class GeneratorSO3toSOn(AbstractGenerator):

    def __init__(self, feature_dim, transform_generator_config):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = transform_generator_config

        # Create Equvariant Rotation Estimator
        self.rotation_estimator = MonotonicMLP(
            layer_sizes = [3, 128, 512, 130816],
            init_weights = 0.0,
            init_std_devs = 1e-3,
            init_bias = [0.0, 0.0, math.pi]
        )

    def getTransformation(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        log: bool = False,
        step: int = -1
    ):
        # Add batch dimension
        if rotation.dim() == 2:
            rotation = rotation.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)

        # Compute input angles
        alpha, beta, gamma = matrix_to_angles(rotation)
        input_angles = torch.stack([alpha, beta, gamma], dim=1)

        # Generate ouput angles (clip between 0-2*pi)
        input_angles = (input_angles + math.pi) % (2 * math.pi)
        generator_angles = self.rotation_estimator(input_angles)
        generator_angles = generator_angles % (2 * math.pi) - math.pi

        # Check number of angles
        num_samples = generator_angles.shape[0]
        num_parameters = generator_angles.shape[1]
        num_angles = self.feature_dim * (self.feature_dim - 1) // 2
        assert num_parameters == num_angles

        # Generate rotation matrix
        generator_matrix = torch.zeros((num_samples, self.feature_dim, self.feature_dim), device=generator_angles.device)

        # Get indices for upper triangle and fill upper and lower parts
        rows, cols = torch.triu_indices(self.feature_dim, self.feature_dim, offset=1)
        rows = rows[None, :].expand(num_samples, num_angles)
        cols = cols[None, :].expand(num_samples, num_angles)
        sample_indexes = torch.arange(num_samples, device=generator_matrix.device)[:, None]
        generator_matrix[sample_indexes, rows, cols] = generator_angles
        generator_matrix[sample_indexes, cols, rows] = -generator_angles

        # Generate rotation matrix
        output_rotation = torch.matrix_exp(generator_matrix)

        # Log relative pose angles and distances
        if log:
            run = getAimRun()
            run.track(
                {
                    "train/equivariance/relative_pose/mean_alpha": alpha.mean(),
                    "train/equivariance/relative_pose/mean_beta": beta.mean(),
                    "train/equivariance/relative_pose/mean_gamma": gamma.mean(),
                    "train/equivariance/relative_pose/std_alpha": alpha.std(),
                    "train/equivariance/relative_pose/std_beta": beta.std(),
                    "train/equivariance/relative_pose/std_gamma": gamma.std(),
                },
                context={"subset": "train"},
                step=step
            )
            run.track(Distribution(generator_matrix.detach().cpu().numpy()), name=f"train/equivariance/generator_matrix_distribution", context={"subset": "train"}, step=step)
            run.track(Distribution(output_rotation.detach().cpu().numpy()), name=f"train/equivariance/rotation_matrix_distribution", context={"subset": "train"}, step=step)
            run.track(Image(colorize(output_rotation[torch.randint(0, output_rotation.size(0), (1,)).item()].detach().cpu().numpy(), cmap="gray")), name="train/equivariance/rotation_matrix", context={"subset": "train"}, step=step)

        return output_rotation


class GeneratorSE3toSEn(AbstractGenerator):

    def __init__(self, feature_dim, transform_generator_config):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = transform_generator_config

        self.feature_dim = feature_dim
        self.config = transform_generator_config

        # Create Equvariant Rotation Estimator
        self.rotation_estimator = MonotonicMLP(
            layer_sizes = [3, 128, 512, 130816],
            init_weights = 0.0,
            init_std_devs = 1e-3,
            init_bias = [0.0, 0.0, math.pi]
        )

        # Create Equivariant Translation Estimator 
        self.translation_estimator = MonotonicMLP(
            layer_sizes =  [3, 128, 512],
            init_weights = 0.0,
            init_std_devs = 1e-3,
            init_bias = [0.0, 200.0]
        )


    def getTransformation(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        log: bool = False,
        step: int = -1
    ):
        # Add batch dimension
        if rotation.dim() == 2:
            rotation = rotation.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
            
        # Compute input angles
        alpha, beta, gamma = matrix_to_angles(rotation)
        input_angles = torch.stack([alpha, beta, gamma], dim=1)

        # Generate ouput angles (clip between 0-2*pi)
        input_angles = (input_angles + math.pi) % (2 * math.pi)
        generator_angles = self.rotation_estimator(input_angles)
        generator_angles = generator_angles % (2 * math.pi) - math.pi

        # Check number of angles
        num_samples = generator_angles.shape[0]
        num_parameters = generator_angles.shape[1]
        num_angles = self.feature_dim * (self.feature_dim - 1) // 2
        assert num_parameters == num_angles

        # Generate rotation matrix
        generator_matrix = torch.zeros((num_samples, self.feature_dim, self.feature_dim), device=generator_angles.device)

        # Get indices for upper triangle and fill upper and lower parts
        rows, cols = torch.triu_indices(self.feature_dim, self.feature_dim, offset=1)
        rows = rows[None, :].expand(num_samples, num_angles)
        cols = cols[None, :].expand(num_samples, num_angles)
        sample_indexes = torch.arange(num_samples, device=generator_matrix.device)[:, None]
        generator_matrix[sample_indexes, rows, cols] = generator_angles
        generator_matrix[sample_indexes, cols, rows] = -generator_angles

        # Generate rotation matrix
        output_rotation = torch.matrix_exp(generator_matrix)

        # Generate translation
        translation = translation + 200.0
        output_translation = self.translation_estimator(translation)
        output_translation = output_translation - 200.0

        # Log relative pose angles and distances
        if log:
            run = getAimRun()
            mean_x, mean_y, mean_z = translation.mean(dim=0)
            std_x, std_y, std_z = translation.std(dim=0)
            run.track(
                {
                    "train/equivariance/relative_pose/mean_alpha": alpha.mean(),
                    "train/equivariance/relative_pose/mean_beta": beta.mean(),
                    "train/equivariance/relative_pose/mean_gamma": gamma.mean(),
                    "train/equivariance/relative_pose/std_alpha": alpha.std(),
                    "train/equivariance/relative_pose/std_beta": beta.std(),
                    "train/equivariance/relative_pose/std_gamma": gamma.std(),
                    "train/equivariance/relative_pose/mean_x": mean_x,
                    "train/equivariance/relative_pose/mean_y": mean_y,
                    "train/equivariance/relative_pose/mean_z": mean_z,
                    "train/equivariance/relative_pose/std_x": std_x,
                    "train/equivariance/relative_pose/std_y": std_y,
                    "train/equivariance/relative_pose/std_z": std_z,
                },
                context={"subset": "train"},
                step=step
            )
            run.track(Distribution(generator_matrix.detach().cpu().numpy()), name=f"train/equivariance/generator_matrix_distribution", context={"subset": "train"}, step=step)
            run.track(Distribution(output_rotation.detach().cpu().numpy()), name=f"train/equivariance/rotation_matrix_distribution", context={"subset": "train"}, step=step)
            run.track(Distribution(output_translation.detach().cpu().numpy()), name=f"train/equivariance/translation_vector_distribution", context={"subset": "train"}, step=step)
            run.track(Image(colorize(output_rotation[torch.randint(0, output_rotation.size(0), (1,)).item()].detach().cpu().numpy(), cmap="gray")), name="train/equivariance/transformation_matrix", context={"subset": "train"}, step=step)

        # Convert to homogeneous matrices
        homogeneous_transformation = torch.zeros(num_samples, self.feature_dim + 1, self.feature_dim + 1, device=output_rotation.device)
        homogeneous_transformation[:, :self.feature_dim, :self.feature_dim] = output_rotation
        homogeneous_transformation[:, :self.feature_dim, self.feature_dim] = output_translation
        homogeneous_transformation[:, self.feature_dim, self.feature_dim] = 1.0

        return homogeneous_transformation


class GeneratorSE3toGLn(AbstractGenerator):

    def __init__(self, feature_dim, transform_generator_config):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = transform_generator_config

        self.feature_dim = feature_dim
        self.config = transform_generator_config

        # Create Equvariant Rotation Estimator
        self.transformation_estimator = MonotonicMLP(
            # SO(n): [3, 128, 512, 130816]
            # GL(n): [3, 128, 512, 262144]
            layer_sizes = [3, 128, 512, 262144],
            init_weights = 0.0,
            init_std_devs = 1e-3,
            init_bias = [0.0, 0.0, 3 * math.pi]
        )

        # Create Equivariant Translation Estimator 
        self.translation_estimator = MonotonicMLP(
            layer_sizes =  [3, 128, 512],
            init_weights = 0.0,
            init_std_devs = 1e-3,
            init_bias = [0.0, 200.0]
        )


    def getTransformation(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        log: bool = False,
        step: int = -1
    ):
        # Add batch dimension
        if rotation.dim() == 2:
            rotation = rotation.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
            
        # Compute input angles
        alpha, beta, gamma = matrix_to_angles(rotation)
        input_angles = torch.stack([alpha, beta, gamma], dim=1)

        # Generate ouput coefficients (clip between +-3*pi)
        input_angles = input_angles + 3 * math.pi
        generator_values = self.transformation_estimator(input_angles)
        generator_values = generator_values - 3 * math.pi

        # Check number of angles
        num_samples = generator_values.shape[0]
        num_parameters = generator_values.shape[1]
        num_values = self.feature_dim * self.feature_dim
        assert num_parameters == num_values

        # Generate rotation matrix
        generator_matrix = generator_values.reshape(-1, 512, 512)

        # Generate rotation matrix
        output_transformation = torch.matrix_exp(generator_matrix)

        # Generate translation
        translation = translation + 200.0
        output_translation = self.translation_estimator(translation)
        output_translation = output_translation - 200.0

        # Log relative pose angles and distances
        if log:
            run = getAimRun()
            mean_x, mean_y, mean_z = translation.mean(dim=0)
            std_x, std_y, std_z = translation.std(dim=0)
            run.track(
                {
                    "train/equivariance/relative_pose/mean_alpha": alpha.mean(),
                    "train/equivariance/relative_pose/mean_beta": beta.mean(),
                    "train/equivariance/relative_pose/mean_gamma": gamma.mean(),
                    "train/equivariance/relative_pose/std_alpha": alpha.std(),
                    "train/equivariance/relative_pose/std_beta": beta.std(),
                    "train/equivariance/relative_pose/std_gamma": gamma.std(),
                    "train/equivariance/relative_pose/mean_x": mean_x,
                    "train/equivariance/relative_pose/mean_y": mean_y,
                    "train/equivariance/relative_pose/mean_z": mean_z,
                    "train/equivariance/relative_pose/std_x": std_x,
                    "train/equivariance/relative_pose/std_y": std_y,
                    "train/equivariance/relative_pose/std_z": std_z,
                },
                context={"subset": "train"},
                step=step
            )
            run.track(Distribution(generator_matrix.detach().cpu().numpy()), name=f"train/equivariance/generator_matrix_distribution", context={"subset": "train"}, step=step)
            run.track(Distribution(output_transformation.detach().cpu().numpy()), name=f"train/equivariance/transformation_matrix_distribution", context={"subset": "train"}, step=step)
            run.track(Distribution(output_translation.detach().cpu().numpy()), name=f"train/equivariance/translation_vector_distribution", context={"subset": "train"}, step=step)
            run.track(Image(colorize(output_transformation[torch.randint(0, output_transformation.size(0), (1,)).item()].detach().cpu().numpy(), cmap="gray")), name="train/equivariance/transformation_matrix", context={"subset": "train"}, step=step)

        # Convert to homogeneous matrices
        homogeneous_transformation = torch.zeros(num_samples, self.feature_dim + 1, self.feature_dim + 1, device=output_transformation.device)
        homogeneous_transformation[:, :self.feature_dim, :self.feature_dim] = output_transformation
        homogeneous_transformation[:, :self.feature_dim, self.feature_dim] = output_translation
        homogeneous_transformation[:, self.feature_dim, self.feature_dim] = 1.0

        return homogeneous_transformation