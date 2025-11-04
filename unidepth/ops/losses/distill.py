import math
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange

from unidepth.utils.geometric import generate_rays, invert_matrix, matrix_to_angles, gather_nearest_patch_center_samples


from unidepth.utils.geometric import generate_rays, invert_matrix
from unidepth.utils.visualization import save_tensors
from unidepth.ops.metrics import bbox_intersections, voxelized_intersections

from .utils import FNS, masked_mean

class SelfDistill(nn.Module):
    def __init__(self, weight: float, output_fn: str = "sqrt", eps: float = 1e-5):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.eps: float = eps

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        intrinsics: torch.Tensor,
        mask: torch.Tensor,
        flips: torch.Tensor,
        downsample_ratio=14,
    ) -> torch.Tensor:
        chunks = input.shape[0] // 2
        mask = F.interpolate(mask.float(), size=input.shape[-2:], mode="nearest")

        iters = zip(
            input.chunk(chunks),
            mask.chunk(chunks),
            intrinsics.chunk(chunks),
            flips.chunk(chunks),
        )
        inputs0, inputs1, masks = [], [], []
        for i, (pair_input, pair_mask, pair_cam, pair_flip) in enumerate(iters):

            mask0, mask1 = pair_mask
            input0, input1 = pair_input
            cam0, cam1 = pair_cam
            flip0, flip1 = pair_flip

            fx_0 = cam0[0, 0] / downsample_ratio
            fx_1 = cam1[0, 0] / downsample_ratio
            cx_0 = cam0[0, 2] / downsample_ratio
            cx_1 = cam1[0, 2] / downsample_ratio
            cy_0 = cam0[1, 2] / downsample_ratio
            cy_1 = cam1[1, 2] / downsample_ratio

            # flip image
            if flip0 ^ flip1:
                input0 = torch.flip(input0, dims=(2,))
                mask0 = torch.flip(mask0, dims=(2,))
                cx_0 = input0.shape[-1] - cx_0

            # calc zoom
            zoom_x = float(fx_1 / fx_0)

            # apply zoom
            input0 = F.interpolate(
                input0.unsqueeze(0), scale_factor=zoom_x, mode="bilinear"
            ).squeeze(0)
            mask0 = F.interpolate(
                mask0.unsqueeze(0), scale_factor=zoom_x, mode="nearest"
            ).squeeze(0)

            # calc translation
            change_left = int(cx_1 - (cx_0 - 0.5) * zoom_x - 0.5)
            change_top = int(cy_1 - (cy_0 - 0.5) * zoom_x - 0.5)
            change_right = input1.shape[-1] - change_left - input0.shape[-1]
            change_bottom = input1.shape[-2] - change_top - input0.shape[-2]

            # apply translation
            pad_left = max(0, change_left)
            pad_right = max(0, change_right)
            pad_top = max(0, change_top)
            pad_bottom = max(0, change_bottom)

            crop_left = max(0, -change_left)
            crop_right = max(0, -change_right)
            crop_top = max(0, -change_top)
            crop_bottom = max(0, -change_bottom)

            input0 = F.pad(
                input0,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            mask0 = F.pad(
                mask0,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            input0 = input0[
                :,
                crop_top : input0.shape[-2] - crop_bottom,
                crop_left : input0.shape[-1] - crop_right,
            ]
            mask0 = mask0[
                :,
                crop_top : mask0.shape[-2] - crop_bottom,
                crop_left : mask0.shape[-1] - crop_right,
            ]

            mask = torch.logical_and(mask0, mask1)

            inputs0.append(input0)
            inputs1.append(input1)
            masks.append(mask)

        inputs0 = torch.stack(inputs0, dim=0)
        inputs1 = torch.stack(inputs1, dim=0)
        masks = torch.stack(masks, dim=0)
        loss1 = self.loss(inputs0, inputs1.detach(), masks)
        loss2 = self.loss(inputs1, inputs0.detach(), masks)
        return torch.cat([loss1, loss2], dim=0)

    def loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = masked_mean(
            (input - target).square().mean(dim=1), mask=mask, dim=[-2, -1]
        )
        return self.output_fn(loss + self.eps)

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
        )
        return obj

class GeneratorSO3(nn.Module):

    def __init__(self, representation_type: str):

        super().__init__()
        self.representation_type = representation_type

        # Generate irreps
        self.irreps = Irreps(representation_type)

    def getTransformation(self, pose1: torch.Tensor, pose2: torch.Tensor):

        # Compute relative pose
        pose0_inv = invert_matrix(pose0)
        pose_rel = pose0_inv @ pose1
        alpha, beta, gamma = matrix_to_angles(pose_rel[:3, :3])

        # Compute the block-diagonal rotation matrix in the irreps space
        R = self.irreps.D_from_angles(alpha, beta, gamma)
        return R

class GeneratorSOn(nn.Module):

    def __init__(self, feature_dim):

        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = feature_dim * (feature_dim - 1) / 2
        self.rotation_decoder = MonotonicAngleDecoder()

    def getTransformation(self, pose0: torch.Tensor, pose1: torch.Tensor):

        # Compute relative position
        pose0_inv = invert_matrix(pose0)
        pose_rel = pose0_inv @ pose1

        # Compute relative position
        # x, y, z = pose_rel[0, 3].item(), pose_rel[1, 3].item(), pose_rel[2, 3].item()

        # Compute input angles
        alpha, beta, gamma = matrix_to_angles(pose_rel[:3, :3])
        input_angles = torch.tensor([alpha, beta, gamma], device=pose_rel.device)
        generator_angles = self.rotation_decoder(input_angles)

        # Check number of angles
        nun_angles = self.feature_dim * (self.feature_dim - 1) // 2
        assert generator_angles.numel() == nun_angles

        # Generate rotation matrix
        generator_matrix = torch.zeros((self.feature_dim, self.feature_dim), device=generator_angles.device)

        # Get indices for upper triangle and fill upper and lower parts
        rows, cols = torch.triu_indices(self.feature_dim, self.feature_dim, offset=1)
        generator_matrix[rows, cols] = generator_angles
        generator_matrix[cols, rows] = -generator_angles

        # Generate rotation matrix
        rotation_matrix = torch.matrix_exp(generator_matrix)
        return rotation_matrix

class MonotonicAngleDecoder(nn.Module):

    def __init__(
        self, 
        init_value=math.pi
    ):
        super().__init__()

        # --- Initialization ---
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 130816)

        # Small weights for hidden layers
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(fc.weight, mean=0.0, std=1e-3)
        
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

class EqSelfDistill(nn.Module):

    def __init__(self, weight: float, symmetry_group: str, representation_type: str, feature_dim: int, output_fn: str = "sqrt",  eps: float = 1e-5, change_base: bool = False, inverse_change_base: bool = False, use_gt: bool = True):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.symmetry_group = symmetry_group
        self.representation_type = representation_type
        self.feature_dim = feature_dim
        self.change_base = change_base
        self.inverse_change_base = inverse_change_base
        self.eps: float = eps
        self.use_gt = use_gt

        # Validate parameters
        if change_base == False and inverse_change_base == True:
            raise ValueError(f"Invalid configuration: 'inverse_change_base' only can be True if 'change_base' is True.")

        if self.change_base:
            self.change_base_generator = nn.Parameter(torch.zeros(self.feature_dim,self.feature_dim))
        else:
            self.register_buffer("change_base_generator", torch.zeros(self.feature_dim,self.feature_dim))

        if self.symmetry_group == "SO(3)":
            self.transform_generator = GeneratorSO3(representation_type)
        elif self.symmetry_group == "SO(n)":
            self.transform_generator = GeneratorSOn(self.feature_dim)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        image: torch.Tensor, # Used only for debugging
        feature: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        estimated_depth: torch.Tensor,
        estimated_intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        mask: torch.Tensor,
        flips: torch.Tensor,
        patch_size=14,
    ) -> Tuple[torch.Tensor, dict]:

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
        points_flat = torch.stack([
            rays_flat[..., 0] * depth_flat[..., 0] / rays_flat[..., 2],
            rays_flat[..., 1] * depth_flat[..., 0] / rays_flat[..., 2],
            depth_flat[..., 0]
        ], dim=-1)
        mask_flat = rearrange(mask, "b c h w -> b (h w) c")
        
        # Transform to global coordinates
        B, N, _ = points_flat.shape
        ones = torch.ones(B, N, 1, device=points_flat.device)
        points_flat = torch.cat([points_flat, ones], dim=-1)
        points_flat = torch.bmm(extrinsics, points_flat.transpose(-2, -1))
        points_flat = points_flat.permute(0, 2, 1)[:, :, :3]   # Drop the homogeneous coordinate
        
        # Calculate intersection over union metrics
        box_ious, box_ioas, box_iobs = [], [], []
        vox_ious, vox_ioas, vox_iobs = [], [], []
        for i in range(B // 2):
            left, right = 2 * i, 2 * i + 1
            mask0_flat, mask1_flat = mask_flat[left], mask_flat[right]
            points0_flat, points1_flat = points_flat[left][mask0_flat.squeeze(-1)], points_flat[right][mask1_flat.squeeze(-1)]
            box_iou, box_ioa, box_iob = bbox_intersections(points0_flat, points1_flat)
            box_ious.append(box_iou); box_ioas.append(box_ioa); box_iobs.append(box_iob)
            vox_iou, vox_ioa, vox_iob = voxelized_intersections(points0_flat, points1_flat, factor=0.1)
            vox_ious.append(vox_iou); vox_ioas.append(vox_ioa); vox_iobs.append(vox_iob)
        box_ious = torch.cat([t.unsqueeze(0) for t in box_ious], dim=0); box_ioas = torch.cat([t.unsqueeze(0) for t in box_ioas], dim=0); box_iobs = torch.cat([t.unsqueeze(0) for t in box_iobs], dim=0)
        vox_ious = torch.cat([t.unsqueeze(0) for t in vox_ious], dim=0); vox_ioas = torch.cat([t.unsqueeze(0) for t in vox_ioas], dim=0); vox_iobs = torch.cat([t.unsqueeze(0) for t in vox_iobs], dim=0)
        print(f"IOU: {vox_ious}")

        # Choose the ray closest to the patch center
        rays = rearrange(rays_flat, "b (h w) c -> b c h w", h=H, w=W)
        points = rearrange(points_flat, "b (h w) c -> b c h w", h=H, w=W) 
        depth, rays, points, mask, chosen_mask = gather_nearest_patch_center_samples(depth, rays, points, mask, patch_size)        

        # Flatten tensors                                     
        # depth_flat = rearrange(depth, "b c h w -> b (h w) c")
        # rays_flat = rearrange(rays, "b c h w -> b (h w) c")
        points_flat = rearrange(points, "b c h w -> b (h w) c")
        mask_flat = rearrange(mask, "b c h w -> b (h w) c")
        features_flat = rearrange(feature, "b c h w -> b (h w) c")
        
        # Create change of basis matrixes
        change_base = torch.matrix_exp(self.change_base_generator)
        change_base_inv = torch.matrix_exp(-self.change_base_generator)

        # Iterate samples
        features0, features1 = [], []
        for i in range(B // 2):
            left, right = 2 * i, 2 * i + 1
            image0, image1 = (image[left], image[right]) if image is not None else (None, None)
            points0, points1 = points_flat[left], points_flat[right]
            feature0, feature1 = features_flat[left], features_flat[right]
            extrinsics0, extrinsics1 = extrinsics[left], extrinsics[right]
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
                points0 = points0[mask0]
                points1 = points1[mask1]
                feature0 = feature0[mask0]
                feature1 = feature1[mask1]

            # Calculate matrix of distances between 3D points
            distances = torch.cdist(points0, points1, p=2)

            # Compute matching plan 
            reg = 0.05       # entropy regularization
            reg_m = 1.0      # mass relaxation for unbalanced OT
            A = torch.ones(points0.shape[0], device=points0.device)
            B = torch.ones(points1.shape[0], device=points1.device)
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

            # TODO: SKIP IF THERE IS NOT ENOUGH MATCHES.

            # Filter out not matched points
            feature0 = feature0[matches[:, 0]]
            feature1 = feature1[matches[:, 1]]

            # Change of base for the feature vectors
            feature0_transformed = torch.matmul(change_base, feature0.T).T
            feature1_transformed = torch.matmul(change_base, feature1.T).T
            
            # Rotate vectors
            transformation0 = self.transform_generator.getTransformation(extrinsics0, extrinsics1)
            feature0_transformed = torch.matmul(transformation0, feature0_transformed.T).T
            transformation1 = self.transform_generator.getTransformation(extrinsics1, extrinsics0)
            feature1_transformed = torch.matmul(transformation1, feature1_transformed.T).T

            # Revert change of base for the feature vectors
            if self.inverse_change_base:
                feature0_transformed = torch.matmul(change_base_inv, feature0_transformed.T).T
                feature1_transformed = torch.matmul(change_base_inv, feature1_transformed.T).T

            # Aggregate results
            features0.append(feature0_transformed)
            features1.append(feature1)
            features0.append(feature1_transformed)
            features1.append(feature0)

        # Compute aggregated loss
        if len(features0) > 0 and len(features1) > 0:
            features0 = torch.cat(features0, dim=0)
            features1 = torch.cat(features1, dim=0)
            loss = self.loss(features0, features1.detach())
        else:
            # If Does not contribute to gradients if image pair is skipped
            print("Warning: Empty loss")
            # HACK: Create a zero that stays in the computational graph to avoid exceptions
            dummy = 0.0
            for p in self.parameters():
                dummy = dummy + (p.sum() * 0.0)
            loss = dummy

        return loss, metrics
    
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

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            symmetry_group = config["symmetry_group"],
            representation_type = config["representation_type"],
            feature_dim=config["feature_dim"],
            output_fn=config["output_fn"],
            change_base=config["change_base"],
            inverse_change_base=config["inverse_change_base"],
            use_gt = config["use_gt"]
        )
        return obj
    
class TeacherDistill(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        cross: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        assert output_fn in FNS
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.eps: float = eps
        self.cross = cross
        self.threshold = 0.05
        self.head_dim = 64  # hardcoded for vit

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        mask: torch.Tensor,
        # metas: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        B = student_features.shape[0]
        device = student_features.device
        chunks = student_features.shape[0] // 2

        mask = (
            F.interpolate(
                mask.float() + 1e-3, size=student_features.shape[-2:], mode="nearest"
            )
            > 0.5
        )

        # chunk features as self.head_dim
        student_features = rearrange(
            student_features, "b (n c) h w -> b c h w n", c=self.head_dim
        )
        teacher_features = rearrange(
            teacher_features, "b (n c) h w -> b c h w n", c=self.head_dim
        )
        student_tokens = rearrange(
            student_tokens, "b t (n c) -> b t c n", c=self.head_dim
        )
        teacher_tokens = rearrange(
            teacher_tokens, "b t (n c) -> b t c n", c=self.head_dim
        )

        distance = (
            (student_features - teacher_features)
            .square()
            .sum(dim=1, keepdim=True)
            .sqrt()
            .mean(dim=-1)
        )
        loss_features = masked_mean(distance, mask=mask, dim=[-2, -1])
        loss_features = self.output_fn(loss_features.clamp(min=self.eps)).squeeze(
            1, 2, 3
        )

        distance = (
            (student_tokens - teacher_tokens).square().sum(dim=-2).sqrt().mean(dim=-1)
        )
        loss_tokens = self.output_fn(distance.clamp(min=self.eps)).squeeze(1)

        return loss_features + 0.01 * loss_tokens

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            cross=config["cross"],
        )
        return obj
