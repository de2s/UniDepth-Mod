import math
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from einops import rearrange
from aim import Image, Distribution
from e3nn.o3 import Irreps
from unidepth.utils.geometric import generate_rays, invert_matrix, matrix_to_angles, gather_nearest_patch_center_samples
from unidepth.utils.logger import getAimRun, log_equivariant_loss_artifacts, save_point_cloud
from unidepth.utils.visualization import colorize
from unidepth.ops.metrics import bbox_intersections, voxelized_intersections

from .utils import FNS, masked_mean
from .utils import FNS


class EqSelfDistill2(nn.Module):

    def __init__(self, weight: float, transform_generation: str, representation_type: str, identity_penalty_factor: float, identity_penalty_sigma: float, feature_dim: int, output_fn: str = "sqrt",  eps: float = 1e-5, change_base: bool = False, inverse_change_base: bool = False, use_gt: bool = True, init_std_dev: float = 1e-3, max_patches_match: int = -1):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.transform_generation = transform_generation
        self.representation_type = representation_type
        self.identity_penalty_factor = identity_penalty_factor
        self.identity_penalty_sigma = identity_penalty_sigma
        self.feature_dim = feature_dim
        self.change_base = change_base
        self.inverse_change_base = inverse_change_base
        self.eps: float = eps
        self.use_gt = use_gt
        if use_gt == False:
            raise RuntimeError("THIS METHOD WAS MOVED TO MATCHER AND HAS NOT BEEN IMPLEMENTED")
        self.init_std_dev = init_std_dev
        self.max_patches_match = max_patches_match

        # Validate parameters
        if change_base == False and inverse_change_base == True:
            raise ValueError(f"Invalid configuration: 'inverse_change_base' only can be True if 'change_base' is True.")

        if self.change_base:
            self.change_base_generator = nn.Parameter(torch.zeros(self.feature_dim,self.feature_dim))

        if self.transform_generation == "SO(3)->Irreps":
            self.transform_generator = GeneratorSO3toIrreps(representation_type)
        elif self.transform_generation == "SO(3)->SO(n)":
            self.transform_generator = GeneratorSO3toSOn(self.feature_dim, self.init_std_dev)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        list_features0: List[torch.Tensor | None], list_features1: List[torch.Tensor | None],
        list_encoder_features0: List[List[torch.Tensor] | None], list_encoder_features1: List[List[torch.Tensor] | None],
        list_points3d0: List[torch.Tensor | None], list_points3d1: List[torch.Tensor | None],
        list_poses0: List[torch.Tensor | None], list_poses1: List[torch.Tensor | None],
        list_flips0: List[torch.Tensor | None], list_flips1: List[torch.Tensor | None],
        log = False,
        step = -1
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:

        # Copy log variable because this function should only log the first iteration
        device = "cuda"
        should_log = log

        # Create change of basis matrixes
        change_base = torch.matrix_exp(self.change_base_generator) if self.change_base else None
        change_base_inv = torch.matrix_exp(-self.change_base_generator) if self.inverse_change_base else None

        # Iterate pairs of samples
        features0, features1 = [], []
        transformations = []
        for feature0, feature1, points3d0, points3d1, pose0, pose1, flip0, flip1 in zip(list_features0, list_features1, list_points3d0, list_points3d1, list_poses0, list_poses1, list_flips0, list_flips1):

            # Check that both images were rotated together
            assert flip0 == flip1
        
            if feature0 == None and feature1 == None:
                continue

            # Verify shapes
            assert feature0.shape == feature1.shape and points3d0.shape == points3d1.shape and pose0.shape == pose1.shape and flip0 == flip1

            # HACK: Truncate the number of shapes to prevent Out of Memory errors
            if self.max_patches_match != -1:
                selected = torch.randperm(feature0.size(0))[:self.max_patches_match] 
                feature0 = feature0[selected]
                feature1 = feature1[selected]
                points3d0 = points3d0[selected]
                points3d1 = points3d1[selected]

            # Change of base for the feature vectors
            feature0_transformed = torch.matmul(change_base, feature0.T).T if change_base else feature0
            feature1_transformed = torch.matmul(change_base, feature1.T).T if change_base else feature1

            # Get the midpoints between matches
            points3d = (points3d0 + points3d1) / 2.0

            # Vector 1: direction of 3D point in camera coordinates
            extrinsics0 = invert_matrix(pose0)
            dir1_cam0 = (extrinsics0 @ torch.cat([points3d, torch.ones(points3d.shape[0], 1, device=points3d.device)], dim=1).T).T[:, :3]
            extrinsics1 = invert_matrix(pose1)
            dir1_cam1 = (extrinsics1 @ torch.cat([points3d, torch.ones(points3d.shape[0], 1, device=points3d.device)], dim=1).T).T[:, :3]

            # Vector 2: direction of point - other camera location
            dir2_cam0 = dir1_cam0 - (extrinsics0 @ pose1[:, 3])[:3] 
            dir2_cam1 = dir1_cam1 - (extrinsics1 @ pose0[:, 3])[:3] 

            # Get rotation from vector 2 to vector 1
            rotations0 = self.rotation_matrices_from_vectors(dir1_cam0, dir2_cam0) 
            rotations1 = self.rotation_matrices_from_vectors(dir1_cam1, dir2_cam1)

            # Rotate vectors
            transformation0 = self.transform_generator.getTransformation(rotations0, should_log, step)
            feature0_transformed = torch.einsum('bij,bi->bj', transformation0, feature0_transformed)
            transformations.append(transformation0)
            transformation1 = self.transform_generator.getTransformation(rotations1, should_log, step)
            feature1_transformed = torch.einsum('bij,bi->bj', transformation1, feature1_transformed)
            transformations.append(transformation1)

            # Verify rotations
            dir2_rotated = torch.bmm(rotations0, dir2_cam0.unsqueeze(-1)).squeeze(-1) 
            dot = (dir2_rotated * dir1_cam0).sum(dim=1)  # [664]
            cos_angle = dot / (dir1_cam0.norm(dim=1) * dir2_cam0.norm(dim=1))
            print("Minimum cos(angle):", cos_angle.min().item())
            print("Maximum cos(angle):", cos_angle.max().item())

            dir1_rotated = torch.bmm(rotations1, dir2_cam1.unsqueeze(-1)).squeeze(-1) 
            dot = (dir1_rotated * dir1_cam1).sum(dim=1)  # [664]
            cos_angle = dot / (dir1_cam1.norm(dim=1) * dir2_cam1.norm(dim=1))
            print("Minimum cos(angle):", cos_angle.min().item())
            print("Maximum cos(angle):", cos_angle.max().item())

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

        # Compute aggregated loss
        if len(features0) > 0 and len(features1) > 0:
            features0 = torch.cat(features0, dim=0)
            features1 = torch.cat(features1, dim=0)
            loss = self.loss(features0, features1.detach())
        else:
            # If Does not contribute to gradients if image pair is skipped
            print("Warning: Empty loss")
            # HACK: Create a zero that stays in the computational graph to avoid exceptions
            dummy = torch.zeros(1, device=device)
            for p in self.parameters():
                dummy = dummy + (p.sum() * 0.0)
            loss = dummy

        # Compute identity penalty loss
        identity_penalty_loss = None
        if self.identity_penalty_factor != 0.0:
            if len(transformations) != 0:
                transformations = torch.stack(transformations, dim=0)
                identity_penalty_loss = self.identity_penalty_loss(transformations, self.identity_penalty_sigma)
            else:
                identity_penalty_loss = torch.zeros(1, device=device)

        return loss, identity_penalty_loss

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
        return self.output_fn(equivariant_loss + self.eps) # QUESTION: WHAT IS THIS FUNCTION FOR (SQRT)

    
    def identity_penalty_loss(
        self,
        transformations: torch.Tensor,
        sigma: float = 0.1,
    ) -> torch.Tensor:
 
        # Extract diagonals -> (B, N)
        diag = torch.diagonal(transformations, dim1=-2, dim2=-1)

        # Peak at 1, decay away from 1
        penalty = torch.exp(-((diag - 1.0) ** 2) / (2 * sigma ** 2))
        return penalty

    def rotation_matrices_from_vectors(self, vec1, vec2):
        """
        Compute rotation matrices that rotate vec2 to vec1
        vec1, vec2: [N, 3] tensors
        Returns: [N, 3, 3] rotation matrices
        """
        # Normalize vectors
        a = vec1 / vec1.norm(dim=1, keepdim=True)
        b = vec2 / vec2.norm(dim=1, keepdim=True)

        # Cross product = rotation axis
        v = torch.cross(b, a, dim=1)  # shape [N, 3]
        s = v.norm(dim=1, keepdim=True)  # sin(theta)
        c = (a * b).sum(dim=1, keepdim=True)  # cos(theta)

        # Skew-symmetric cross-product matrix K
        K = torch.zeros((vec1.shape[0], 3, 3), device=vec1.device)
        K[:, 0, 1] = -v[:, 2]
        K[:, 0, 2] =  v[:, 1]
        K[:, 1, 0] =  v[:, 2]
        K[:, 1, 2] = -v[:, 0]
        K[:, 2, 0] = -v[:, 1]
        K[:, 2, 1] =  v[:, 0]

        I = torch.eye(3, device=vec1.device).unsqueeze(0)  # [1, 3, 3]

        # Rodrigues formula: R = I + K + K^2 * (1-c)/s^2
        s_safe = s.clone()
        s_safe[s_safe < 1e-8] = 1e-8  # avoid division by zero for very small angles
        R = I + K + torch.bmm(K, K) * ((1 - c) / (s_safe**2)).unsqueeze(-1)

        return R
    
    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            transform_generation = config["transform_generation"],
            representation_type = config["representation_type"],
            identity_penalty_factor = config["identity_penalty_factor"],
            identity_penalty_sigma = config["identity_penalty_factor"], 
            feature_dim=config["feature_dim"],
            output_fn=config["output_fn"],
            change_base=config["change_base"],
            inverse_change_base=config["inverse_change_base"],
            use_gt = config["use_gt"],
            init_std_dev = config["init_std_dev"],
            max_patches_match = config["max_patches_match"],
        )
        return obj

class GeneratorSO3toIrreps(nn.Module):

    def __init__(self, representation_type: str):

        super().__init__()
        self.representation_type = representation_type

        # Generate irreps
        self.irreps = Irreps(representation_type)

    def getTransformation(self, rotation: torch.Tensor, log = False, step = -1):
        
        # Compute the block-diagonal rotation matrix in the irreps space
        alpha, beta, gamma = matrix_to_angles(rotation.cpu())
        R = self.irreps.D_from_angles(alpha, beta, gamma).to(rotation.device)

        # Log relative pose angles and distances
        if log:
            run = getAimRun()
            run.track(
                {
                    "train/equivariance/relative_pose/alpha_mean": alpha.mean(),
                    "train/equivariance/relative_pose/beta_mean": beta.mean(),
                    "train/equivariance/relative_pose/gamma_mean": gamma.mean(),
                    "train/equivariance/relative_pose/alpha_std": alpha.std(),
                    "train/equivariance/relative_pose/beta_std": beta.std(),
                    "train/equivariance/relative_pose/gamma_std": gamma.std(),
                },
                context={"subset": "train"},
                step=step
            )
            run.track(Image(colorize(R.detach().cpu().numpy(), cmap="gray")), name="train/equivariance/rotation_matrix", context={"subset": "train"}, step=step)
            run.track(Distribution(R.detach().cpu().numpy()), name=f"train/equivariance/rotation_matrix_distribution", context={"subset": "train"}, step=step)
            
        return R


class GeneratorSO3toSOn(nn.Module):

    def __init__(self, feature_dim, init_std_dev):

        super().__init__()
        self.feature_dim = feature_dim
        self.rotation_decoder = MonotonicAngleDecoder(init_std_dev=init_std_dev)

    def getTransformation(self, rotation: torch.Tensor, log = False, step = -1):

        # Compute input angles
        alpha, beta, gamma = matrix_to_angles(rotation)
        input_angles = torch.stack([alpha, beta, gamma], dim=1)
        generator_angles = self.rotation_decoder(input_angles)

        # Check number of angles
        num_samples = generator_angles.shape[0]
        num_parameters = generator_angles.shape[1]
        num_angles = self.feature_dim * (self.feature_dim - 1) // 2
        assert num_parameters == num_angles

        # Generate rotation matrix
        generator_matrix = torch.zeros((num_samples, self.feature_dim, self.feature_dim), device=generator_angles.device)

        # Get indices for upper triangle and fill upper and lower parts
        #rows, cols = torch.triu_indices(self.feature_dim, self.feature_dim, offset=1)
        #generator_matrix[rows, cols] = generator_angles
        #generator_matrix[cols, rows] = -generator_angles
        rows, cols = torch.triu_indices(self.feature_dim, self.feature_dim, offset=1)
        rows = rows[None, :].expand(num_samples, num_angles)
        cols = cols[None, :].expand(num_samples, num_angles)
        sample_indexes = torch.arange(num_samples, device=generator_matrix.device)[:, None]
        generator_matrix[sample_indexes, rows, cols] = generator_angles
        generator_matrix[sample_indexes, cols, rows] = -generator_angles

        # Generate rotation matrix
        rotation_matrix = torch.matrix_exp(generator_matrix)

        # Log relative pose angles and distances
        if log:
            run = getAimRun()
            run.track(
                {
                    "train/equivariance/relative_pose/alpha_mean": alpha.mean(),
                    "train/equivariance/relative_pose/beta_mean": beta.mean(),
                    "train/equivariance/relative_pose/gamma_mean": gamma.mean(),
                    "train/equivariance/relative_pose/alpha_std": alpha.std(),
                    "train/equivariance/relative_pose/beta_std": beta.std(),
                    "train/equivariance/relative_pose/gamma_std": gamma.std(),
                },
                context={"subset": "train"},
                step=step
            )
            run.track(Distribution(generator_matrix.detach().cpu().numpy()), name=f"train/equivariance/generator_matrix_distribution", context={"subset": "train"}, step=step)
            run.track(Distribution(rotation_matrix.detach().cpu().numpy()), name=f"train/equivariance/rotation_matrix_distribution", context={"subset": "train"}, step=step)
            run.track(Image(colorize(rotation_matrix[torch.randint(0, rotation_matrix.size(0), (1,)).item()].detach().cpu().numpy(), cmap="gray")), name="train/equivariance/rotation_matrix", context={"subset": "train"}, step=step)

        return rotation_matrix


class MonotonicAngleDecoder(nn.Module):

    def __init__(
        self, 
        init_value = math.pi,
        init_std_dev = 1e-3
    ):
        super().__init__()

        # --- Initialization ---
        ### TODO: FIXED DEVICE BREAKS DISTRIBUTED TRAINING
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 130816)

        # Small weights for hidden layers
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(fc.weight, mean=0.0, std=init_std_dev)
            ### TODO: INCREASE init_std_dev TO ALLOW MORE RANDOMESS ON INITIAL ROTATION ANGLES.
        
        # Hidden biases zero
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

        # Final bias = init_value for all outputs
        nn.init.constant_(self.fc3.bias, init_value)

    def forward(self, x):

        # Move the tensor to the interval [0 to 2*pi]
        x = (x + math.pi) % (2 * math.pi)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Move the tensor the the inverval [-pi to pi]
        x = x % (2 * math.pi) - math.pi

        return x


    
