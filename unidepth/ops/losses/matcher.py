import math
import ot
import torch
from typing import List, Tuple
from einops import rearrange
from unidepth.utils.geometric import generate_rays, gather_nearest_patch_center_samples
from unidepth.utils.logger import log_equivariant_loss_artifacts, save_point_cloud
from unidepth.ops.metrics import bbox_intersections, voxelized_intersections


@torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
def match_features(
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
    use_gt: bool = True,
    patch_size=14,
    log = False,
    step = -1
) -> Tuple[ List[torch.Tensor | None], List[torch.Tensor | None], List[List[torch.Tensor]], List[List[torch.Tensor]], List[torch.Tensor | None], List[torch.Tensor | None], List[torch.Tensor | None], List[torch.Tensor | None], List[torch.Tensor | None], List[torch.Tensor | None], dict]:

    # Copy log variable because this function should only log the first iteration
    should_log = log
    metrics = {}

    # Choose between ground truth or estimated data
    depth = depth if use_gt else estimated_depth
    mask = mask if use_gt else torch.ones_like(mask, dtype=torch.bool)
    K = intrinsics if use_gt else estimated_intrinsics

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
    list_encoder_features_flat = [rearrange(encoder_feature,  "b h w c -> b (h w) c") for encoder_feature in encoder_features]

    # Iterate samples
    list_features0, list_features1 = [], []
    list_encoder_features0, list_encoder_features1 = [], []
    list_points3d0, list_points3d1 = [], []
    list_poses0, list_poses1 = [], []
    list_flips0, list_flips1 = [], []
    num_pairs = B // 2
    for i in range(num_pairs):
        left, right = 2 * i, 2 * i + 1
        image0, image1 = (image[left], image[right]) if image is not None else (None, None)
        points3d0, points3d1 = points3d_flat[left], points3d_flat[right]
        points2d0, points2d1 = points2d_flat[left], points2d_flat[right]
        feature0, feature1 = features_flat[left], features_flat[right]
        list_encoder_feature0, list_encoder_feature1  = [encoders_feature_flat[left] for encoders_feature_flat in list_encoder_features_flat], [encoder_features_flat[right] for encoder_features_flat in list_encoder_features_flat]
        pose0, pose1 = poses[left], poses[right]
        mask0, mask1 = mask_flat[left], mask_flat[right]
        flip0, flip1 = flips[left], flips[right]

        # Check that both images were rotated together
        assert flip0 == flip1

        # Skip pair if cloud points doesn't intersect
        if vox_ious[i] < 0.2:
            print(f"Warning: Skip pair of frames because IoU is too low: {vox_ious[i]}")
            list_features0.append(None)
            list_features1.append(None)
            list_encoder_features0.append(None)
            list_encoder_features1.append(None)
            list_points3d0.append(None)
            list_points3d1.append(None)
            list_poses0.append(None)
            list_poses1.append(None)
            list_flips0.append(None)
            list_flips1.append(None)
            continue

        if use_gt:
            mask0, mask1 = mask0.squeeze(), mask1.squeeze()
            points3d0 = points3d0[mask0]
            points3d1 = points3d1[mask1]
            points2d0 = points2d0[mask0]
            points2d1 = points2d1[mask1]
            feature0 = feature0[mask0]
            feature1 = feature1[mask1]
            list_encoder_feature0 = [encoder_feature0[mask0] for encoder_feature0 in list_encoder_feature0]
            list_encoder_feature1 = [encoder_feature1[mask1] for encoder_feature1 in list_encoder_feature1]

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
            print(f"Warning: Skip pair of frames because there are no matches.")
            list_features0.append(None)
            list_features1.append(None)
            list_encoder_features0.append(None)
            list_encoder_features1.append(None)
            list_points3d0.append(None)
            list_points3d1.append(None)
            list_poses0.append(None)
            list_poses1.append(None)
            list_flips0.append(None)
            list_flips1.append(None)
            continue

        # Log artifacts if needed
        if should_log:

            print("Logging Artifacts.")

            log_equivariant_loss_artifacts(
                image0=image0, image1=image1, 
                mask0=rearrange(mask0, "(h w) -> h w", h=num_patches_y, w=num_patches_x), mask1=rearrange(mask1, "(h w) -> h w", h=num_patches_y, w=num_patches_x), 
                feature0=feature0, feature1=feature1,
                list_encoder_feature0=list_encoder_feature0, list_encoder_feature1=list_encoder_feature0,
                points2d0=points2d0, points2d1=points2d1,
                matches=matches,
                step = step
            )

        # Filter out not matched points
        feature0 = feature0[matches[:, 0]]
        feature1 = feature1[matches[:, 1]]
        points3d0 = points3d0[matches[:, 0]]
        points3d1 = points3d1[matches[:, 1]]
        list_encoder_feature0 = [encoder_feature0[matches[:, 0]] for encoder_feature0 in list_encoder_feature0]
        list_encoder_feature1 = [encoder_feature1[matches[:, 1]] for encoder_feature1 in list_encoder_feature1]

        # Track number of matched points
        if should_log:
            metrics["NumMatchedFeaturesSample"] = torch.tensor(feature0.size(0))

        # Check shapes
        assert feature0.shape == feature1.shape and points3d0.shape == points3d1.shape
        for encoder_feature_a, encoder_feature_b in zip(list_encoder_feature0, list_encoder_feature1):
            assert encoder_feature_a.shape == encoder_feature_b.shape

        list_features0.append(feature0)
        list_features1.append(feature1)
        list_encoder_features0.append(list_encoder_feature0)
        list_encoder_features1.append(list_encoder_feature1)
        list_points3d0.append(points3d0)
        list_points3d1.append(points3d1)
        list_poses0.append(pose0)
        list_poses1.append(pose1)
        list_flips0.append(flip0)
        list_flips1.append(flip1)

    # Check lengths
    assert len(list_features0) == len(list_features1) and len(list_encoder_features0) == len(list_encoder_features1) and len(list_points3d0) == len(list_points3d1) and len(list_poses0) == len(list_poses1) and len(list_flips0) == len(list_flips1)
    
    return list_features0, list_features1, list_encoder_features0, list_encoder_features1, list_points3d0, list_points3d1, list_poses0, list_poses1, list_flips0, list_flips1, metrics