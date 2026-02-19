import math
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple
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
        self.init_std_dev = init_std_dev
        self.max_patches_match = max_patches_match

        # Validate parameters
        if change_base == False and inverse_change_base == True:
            raise ValueError(f"Invalid configuration: 'inverse_change_base' only can be True if 'change_base' is True.")

        if self.change_base:
            self.change_base_generator = nn.Parameter(torch.zeros(self.feature_dim,self.feature_dim))
        else:
            self.register_buffer("change_base_generator", torch.zeros(self.feature_dim,self.feature_dim))

        if self.transform_generation == "SO(3)->Irreps":
            self.transform_generator = GeneratorSO3toIrreps(representation_type)
        elif self.transform_generation == "SO(3)->SO(n)":
            self.transform_generator = GeneratorSO3toSOn(self.feature_dim, self.init_std_dev)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        image: torch.Tensor, # Used only for debugging
        feature: torch.Tensor,
        encoder_features: list,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        estimated_depth: torch.Tensor,
        estimated_intrinsics: torch.Tensor,
        poses: torch.Tensor,
        mask: torch.Tensor,
        flips: torch.Tensor,
        patch_size=14,
        log = False,
        step = -1
    ) -> Tuple[torch.Tensor, torch.Tensor | None, dict]:
 
        # # Detach everything except features
        # image = image.detach()
        # #feature: torch.Tensor,
        # #encoder_features: list,
        # depth = depth.detach()
        # intrinsics = intrinsics.detach()
        # estimated_depth = estimated_depth.detach()
        # estimated_intrinsics = estimated_intrinsics.detach()
        # poses = poses.detach()
        # mask = mask.detach()
        # flips = flips.detach()

        # Copy log variable because this function should only log the first iteration
        should_log = log
        metrics = {}

        # Choose between ground truth or estimated data
        depth = depth if self.use_gt else estimated_depth
        mask = mask if self.use_gt else torch.ones_like(mask, dtype=torch.bool)
        K = intrinsics if self.use_gt else estimated_intrinsics

        # Calculate number of patches
        B, _, H, W = depth.shape
        device = depth.device
        assert W % patch_size == 0 and H % patch_size == 0
        num_patches_x, num_patches_y = W // patch_size, H // patch_size

        # Generate rays     
        rays_flat, _ = generate_rays(K, (H, W))

        # Retroproject points
        depth_flat = rearrange(depth, "b c h w -> b (h w) c")
        points3d_flat = torch.stack([
            rays_flat[..., 0] * depth_flat[..., 0] / rays_flat[..., 2],
            rays_flat[..., 1] * depth_flat[..., 0] / rays_flat[..., 2],
            depth_flat[..., 0]
        ], dim=-1)
        mask_flat = rearrange(mask, "b c h w -> b (h w) c")
        
        # Transform to global coordinates
        B, N, _ = points3d_flat.shape
        ones = torch.ones(B, N, 1, device=points3d_flat.device)
        points3d_flat = torch.cat([points3d_flat, ones], dim=-1)
        points3d_flat = torch.bmm(poses, points3d_flat.transpose(-2, -1))
        points3d_flat = points3d_flat.permute(0, 2, 1)[:, :, :3]   # Drop the homogeneous coordinate
        
        # Calculate intersection over union metrics
        box_ious, box_ioas, box_iobs = [], [], []
        vox_ious, vox_ioas, vox_iobs = [], [], []
        for i in range(B // 2):
            left, right = 2 * i, 2 * i + 1
            mask0_flat, mask1_flat = mask_flat[left], mask_flat[right]
            points3d0_flat, points3d1_flat = points3d_flat[left][mask0_flat.squeeze(-1)], points3d_flat[right][mask1_flat.squeeze(-1)]
            box_iou, box_ioa, box_iob = bbox_intersections(points3d0_flat, points3d1_flat)
            box_ious.append(box_iou); box_ioas.append(box_ioa); box_iobs.append(box_iob)
            vox_iou, vox_ioa, vox_iob = voxelized_intersections(points3d0_flat, points3d1_flat, factor=0.1)
            vox_ious.append(vox_iou); vox_ioas.append(vox_ioa); vox_iobs.append(vox_iob)
        box_ious = torch.cat([t.unsqueeze(0) for t in box_ious], dim=0); box_ioas = torch.cat([t.unsqueeze(0) for t in box_ioas], dim=0); box_iobs = torch.cat([t.unsqueeze(0) for t in box_iobs], dim=0)
        vox_ious = torch.cat([t.unsqueeze(0) for t in vox_ious], dim=0); vox_ioas = torch.cat([t.unsqueeze(0) for t in vox_ioas], dim=0); vox_iobs = torch.cat([t.unsqueeze(0) for t in vox_iobs], dim=0)

        # Log input point cloud
        if should_log:
            metrics["BoxIntersectionOverUnionSample"] = box_ious[0]
            metrics["VoxIntersectionOverUnionSample"] = vox_ious[0]
            save_point_cloud(
                f"artifacts/train/point_cloud_{step:06d}",
                torch.cat([points3d_flat[0], points3d_flat[1]], dim=0),
                torch.cat([
                    torch.tensor([0, 255, 0], dtype=torch.uint8).repeat(points3d_flat[0].shape[0], 1),
                    torch.tensor([255, 0, 0], dtype=torch.uint8).repeat(points3d_flat[1].shape[0], 1)
                    ],
                    dim=0
                )
            )

        # Choose the ray closest to the patch center
        rays = rearrange(rays_flat, "b (h w) c -> b c h w", h=H, w=W)
        points3d = rearrange(points3d_flat, "b (h w) c -> b c h w", h=H, w=W) 
        depth, rays, points3d, mask, points2d = gather_nearest_patch_center_samples(depth, rays, points3d, mask, patch_size)        

        # Flatten tensors                                     
        # depth_flat = rearrange(depth, "b c h w -> b (h w) c")
        # rays_flat = rearrange(rays, "b c h w -> b (h w) c")
        points3d_flat = rearrange(points3d, "b c h w -> b (h w) c")
        mask_flat = rearrange(mask, "b c h w -> b (h w) c")
        points2d_flat = rearrange(points2d, "b c h w -> b (h w) c")
        features_flat = rearrange(feature, "b c h w -> b (h w) c")
        
        # Create change of basis matrixes
        change_base = torch.matrix_exp(self.change_base_generator)
        change_base_inv = torch.matrix_exp(-self.change_base_generator)

        # Iterate samples
        features0, features1 = [], []
        transformations = []
        num_pairs = B // 2
        for i in range(num_pairs):
            left, right = 2 * i, 2 * i + 1
            image0, image1 = (image[left], image[right]) if image is not None else (None, None)
            points3d0, points3d1 = points3d_flat[left], points3d_flat[right]
            points2d0, points2d1 = points2d_flat[left], points2d_flat[right]
            feature0, feature1 = features_flat[left], features_flat[right]
            pose0, pose1 = poses[left], poses[right]
            mask0, mask1 = mask_flat[left], mask_flat[right]
            flip0, flip1 = flips[left], flips[right]

            # Check that both images were rotated together
            assert flip0 == flip1

            # Skip pair if cloud points doesn't intersect
            if vox_ious[i] < 0.2:
                print(f"Warning: Skip pair of frames because IoU is too low: {vox_ious[i]}")
                continue

            if self.use_gt:
                mask0, mask1 = mask0.squeeze(), mask1.squeeze()
                points3d0 = points3d0[mask0]
                points3d1 = points3d1[mask1]
                points2d0 = points2d0[mask0]
                points2d1 = points2d1[mask1]
                feature0 = feature0[mask0]
                feature1 = feature1[mask1]

            # Log number of points    
            if should_log:
                metrics["NumSelectedPointsSample"] = torch.tensor(points3d0.size(0) + points3d1.size(0))

            # Calculate matrix of distances between 3D points
            distances = torch.cdist(points3d0, points3d1, p=2)

            # Compute matching plan 
            reg = 0.05       # entropy regularization
            reg_m = 1.0      # mass relaxation for unbalanced OT
            A = torch.ones(points3d0.shape[0], device=points3d0.device)
            B = torch.ones(points3d1.shape[0], device=points3d1.device)
            A /= A.sum()
            B /= B.sum()            
            # defaults: method='sinkhorn', reg_type='kl', numItermax=1000, stopThr=1e-6, verbose=False
            plan = ot.unbalanced.sinkhorn_unbalanced(A, B, distances, reg, reg_m) # TODO: CHECK DISTANCE SQUARED
            
            # Calculate costs using unbalanced optimal transport
            matches_a = torch.argmax(plan, dim=1)  # For each A[i], best B[j]
            matches_b = torch.argmax(plan, dim=0)  # For each B[i], best A[j]

            # Get mutual matches
            matches_a_indices = torch.arange(matches_a.size(0), device=device)
            mutual_mask = matches_b[matches_a] == matches_a_indices  # only keep if mutual
            matches = torch.stack([matches_a_indices[mutual_mask], matches_a[mutual_mask]], dim=1)

            # Skip if there is not matches
            if len(matches) == 0:
                continue

            # Log artifacts if needed
            if should_log:

                print("Logging Artifacts.")

                log_equivariant_loss_artifacts(
                    image0=image0, image1=image1, 
                    mask0=rearrange(mask0, "(h w) -> h w", h=num_patches_y, w=num_patches_x), mask1=rearrange(mask1, "(h w) -> h w", h=num_patches_y, w=num_patches_x), 
                    feature0=feature0, feature1=feature1,
                    list_encoder_feature0=None, list_encoder_feature1=None, list_encoder_indexes=None,
                    points2d0=points2d0, points2d1=points2d1,
                    matches=matches,
                    step = step
                )
            
            # HACK: Truncate the number of shapes to prevent Out of Memory errors
            if self.max_patches_match != -1:
                selected = torch.randperm(matches.size(0))[:self.max_patches_match]
                matches = matches[selected]

            # Filter out not matched points
            feature0 = feature0[matches[:, 0]]
            feature1 = feature1[matches[:, 1]]
            points3d0 = points3d0[matches[:, 0]]
            points3d1 = points3d1[matches[:, 1]]

            # Track number of matched points
            if should_log:
                metrics["NumMatchedFeaturesSample"] = torch.tensor(feature0.size(0))

            # Change of base for the feature vectors
            feature0_transformed = torch.matmul(change_base, feature0.T).T
            feature1_transformed = torch.matmul(change_base, feature1.T).T

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
            if self.inverse_change_base:
                feature0_transformed = torch.matmul(change_base_inv, feature0_transformed.T).T
                feature1_transformed = torch.matmul(change_base_inv, feature1_transformed.T).T

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

        return loss, identity_penalty_loss, metrics

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


    
