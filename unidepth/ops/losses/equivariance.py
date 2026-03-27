from typing import List

import torch
import torch.nn as nn

from unidepth.utils.geometric import invert_matrix
import unidepth.ops.losses.generators as generators

from .utils import FNS

class EquivarianceLoss(nn.Module):

    def __init__(
            self, 
            weight: float, 
            transform_generator_type: str,
            transform_generator_config: dict, 
            feature_dim: int, 
            output_fn: str = "sqrt", 
            eps: float = 1e-5, 
            change_base: bool = False, 
            inverse_change_base: bool = False,
    ):
        super().__init__()

        # Save config
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.transform_generator_type = transform_generator_type
        self.transform_generator_config = transform_generator_config
        self.feature_dim = feature_dim
        self.change_base = change_base
        self.inverse_change_base = inverse_change_base
        self.eps: float = eps

        # Validate change of base parameters
        if change_base == False and inverse_change_base == True:
            raise ValueError(f"Invalid configuration: 'inverse_change_base' only can be True if 'change_base' is True.")
        
        # Create change of base matrix
        if self.change_base:
            self.change_base_generator = nn.Parameter(torch.zeros(self.feature_dim,self.feature_dim))

        # Create transform generator
        cls = getattr(generators, transform_generator_type)
        self.transform_generator = cls(feature_dim, transform_generator_config)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        list_features0: List[torch.Tensor | None], list_features1: List[torch.Tensor | None],
        list_points3d0: List[torch.Tensor | None], list_points3d1: List[torch.Tensor | None],
        list_poses0: List[torch.Tensor | None], list_poses1: List[torch.Tensor | None],
        list_flips0: List[torch.Tensor | None], list_flips1: List[torch.Tensor | None],
        log = False,
        step = -1
    ) -> dict:
        
        # Copy log variable because this function should only log the first iteration
        should_log = log

        # Create change of basis matrixes
        change_base = torch.matrix_exp(self.change_base_generator) if self.change_base else None
        change_base_inv = torch.matrix_exp(-self.change_base_generator) if self.inverse_change_base else None

        # Iterate pairs of samples
        features0, features1 = [], []
        # transformations = []
        for (
            feature0, feature1, 
            points3d0, points3d1, 
            pose0, pose1, 
            flip0, flip1
         ) in zip(
            list_features0, list_features1, 
            list_points3d0, list_points3d1, 
            list_poses0, list_poses1, 
            list_flips0, list_flips1
        ):
            
            # Check that both images were rotated together
            assert flip0 == flip1

            # Skip if there are not matched features for the pair of frames
            if feature0 == None and feature1 == None:
                continue

            # Verify shapes
            assert feature0.shape == feature1.shape and points3d0.shape == points3d1.shape and pose0.shape == pose1.shape and flip0 == flip1

            # Get the number of samples
            num_samples = feature0.shape[0]

            # Change of base for the feature vectors
            feature0_transformed = torch.matmul(change_base, feature0.T).T if change_base else feature0
            feature1_transformed = torch.matmul(change_base, feature1.T).T if change_base else feature1

            # Compute relative poses
            pose0_inv = invert_matrix(pose0)
            pose0_rel = pose0_inv @ pose1
            pose1_inv = invert_matrix(pose1)
            pose1_rel = pose1_inv @ pose0

            # Transform features
            transformation0 = self.transform_generator.getTransformation(pose0_rel[:3, :3], pose0_rel[:3, 3], should_log, step)
            if transformation0.shape[1] == feature0_transformed.shape[1] and transformation0.shape[1] == feature0_transformed.shape[1]:
                # Normal transformation
                feature0_transformed = torch.einsum('bij,bi->bj', transformation0, feature0_transformed)
            elif transformation0.shape[1] == feature0_transformed.shape[1] + 1 and transformation0.shape[1] == feature0_transformed.shape[1] + 1:
                # Homogeneous transformation
                ones = torch.ones(num_samples, 1, device=feature0_transformed.device, dtype=feature0_transformed.dtype)
                feature0_transformed = torch.cat([feature0_transformed, ones], dim=1)
                feature0_transformed = torch.einsum('bij,bi->bj', transformation0, feature0_transformed)
                feature0_transformed = feature0_transformed[:, :-1]
            #transformations.append(transformation0)
            transformation1 = self.transform_generator.getTransformation(pose1_rel[:3, :3], pose1_rel[:3, 3], should_log, step)
            if transformation1.shape[1] == feature1_transformed.shape[1] and transformation1.shape[1] == feature1_transformed.shape[1]:
                feature1_transformed = torch.einsum('bij,bi->bj', transformation1, feature1_transformed)
            elif transformation1.shape[1] == feature1_transformed.shape[1] + 1 and transformation1.shape[1] == feature1_transformed.shape[1] + 1:
                # Homogeneous transformation
                ones = torch.ones(num_samples, 1, device=feature1_transformed.device, dtype=feature1_transformed.dtype)
                feature1_transformed = torch.cat([feature1_transformed, ones], dim=1)
                feature1_transformed = torch.einsum('bij,bi->bj', transformation1, feature1_transformed)
                feature1_transformed = feature1_transformed[:, :-1]
            #transformations.append(transformation1)

            # Revert change of base for the feature vectors
            feature0_transformed = torch.matmul(change_base_inv, feature0_transformed.T).T if change_base_inv else feature0_transformed
            feature1_transformed = torch.matmul(change_base_inv, feature1_transformed.T).T if change_base_inv else feature1_transformed

            # Aggregate results
            features0.append(feature0_transformed)
            features1.append(feature1)
            features0.append(feature1_transformed)
            features1.append(feature0)

            # Log only the first iteration
            should_log = False

        # Compute aggregated losses
        if len(features0) > 0 and len(features1) > 0:
            features0 = torch.cat(features0, dim=0)
            features1 = torch.cat(features1, dim=0)
            loss = self.loss(features0, features1.detach())
        else:
            # If Does not contribute to gradients if image pair is skipped
            print("Warning: Empty loss")
            # HACK: Create a zero that stays in the computational graph to avoid exceptions
            device = "cuda"
            dummy = torch.zeros(1, device=device)
            for p in self.parameters():
                dummy = dummy + (p.sum() * 0.0)
            loss = dummy

        return loss

    def loss(
        self,
        features0: torch.Tensor,
        features1: torch.Tensor,
    ) -> torch.Tensor:
        
        # Compute cosine similarity for each pair
        dot = torch.sum(features0 * features1, dim=1) 
        norm0 = torch.norm(features0, dim=1)
        norm1 = torch.norm(features1, dim=1)
        cos_theta = dot / (norm0 * norm1 + self.eps) 

        # Equivariant loss
        equivariant_loss = 1 - torch.abs(cos_theta)
        return self.output_fn(equivariant_loss + self.eps)
    
    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            transform_generator_type = config["transform_generator_type"],
            transform_generator_config = config["transform_generator_config"], 
            feature_dim=config["feature_dim"],
            output_fn=config["output_fn"],
            change_base=config["change_base"],
            inverse_change_base=config["inverse_change_base"]
        )
        return obj
