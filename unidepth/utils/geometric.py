"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Tuple

import torch
from torch.nn import functional as F

from e3nn.o3 import xyz_to_angles, angles_to_matrix


@torch.jit.script
def generate_rays(
    camera_intrinsics: torch.Tensor, image_shape: Tuple[int, int], noisy: bool = False
):
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (3, H*W)
    ray_directions = F.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    # pitch = torch.asin(ray_directions[..., 1])
    # roll = torch.atan2(ray_directions[..., 0], - ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles


# Define function to calculate inverse matrix
#@torch.jit.script
def invert_matrix(T):

    num_dimmensions = T.dim()
    if num_dimmensions == 2:
        T = T.unsqueeze(0)

    W, H = T.shape[-2:]
    if W == 3 and H == 3:
        R_inv = T.transpose(-2,-1)
        value  = R_inv
    elif W == 4 and H == 4:
        R = T[:, :3, :3]              # Rotation part
        t = T[:, :3, 3]               # Translation part
        t = t.unsqueeze(-1) 
        R_inv = R.transpose(-2, -1)   # Transpose of rotation matrix
        t_inv = -torch.bmm(R_inv, t)  # Inverted translation
        T_inv = torch.zeros_like(T)
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3] = t_inv.squeeze(-1)
        T_inv[:, 3, 3] = 1.0
        value = T_inv
    else:
        raise RuntimeError("Shape not supported")
    
    if num_dimmensions == 2:
        value = value.squeeze(0)
    return value

@torch.jit.script
def spherical_zbuffer_to_euclidean(spherical_tensor: torch.Tensor) -> torch.Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract zbuffer depth

    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    # =>
    # r = z / sin(phi) / cos(theta)
    # y = z / (sin(phi) / cos(phi)) / cos(theta)
    # x = z * sin(theta) / cos(theta)
    x = z * torch.tan(theta)
    y = z / torch.tan(phi) / torch.cos(theta)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


@torch.jit.script
def spherical_to_euclidean(spherical_tensor: torch.Tensor) -> torch.Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    r = spherical_tensor[..., 2]  # Extract radius
    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    x = r * torch.sin(phi) * torch.sin(theta)
    y = r * torch.cos(phi)
    z = r * torch.cos(theta) * torch.sin(phi)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


@torch.jit.script
def euclidean_to_spherical(spherical_tensor: torch.Tensor) -> torch.Tensor:
    x = spherical_tensor[..., 0]  # Extract polar angle
    y = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract radius
    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(x / r, z / r)
    phi = torch.acos(y / r)

    euclidean_tensor = torch.stack((theta, phi, r), dim=-1)
    return euclidean_tensor


@torch.jit.script
def euclidean_to_spherical_zbuffer(euclidean_tensor: torch.Tensor) -> torch.Tensor:
    pitch = torch.asin(euclidean_tensor[..., 1])
    yaw = torch.atan2(euclidean_tensor[..., 0], euclidean_tensor[..., -1])
    z = euclidean_tensor[..., 2]  # Extract zbuffer depth
    euclidean_tensor = torch.stack((pitch, yaw, z), dim=-1)
    return euclidean_tensor


@torch.jit.script
def unproject_points(
    depth: torch.Tensor, camera_intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Unprojects a batch of depth maps to 3D point clouds using camera intrinsics.

    Args:
        depth (torch.Tensor): Batch of depth maps of shape (B, 1, H, W).
        camera_intrinsics (torch.Tensor): Camera intrinsic matrix of shape (B, 3, 3).

    Returns:
        torch.Tensor: Batch of 3D point clouds of shape (B, 3, H, W).
    """
    batch_size, _, height, width = depth.shape
    device = depth.device

    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    pixel_coords = torch.stack((x_coords, y_coords), dim=-1)  # (H, W, 2)

    # Get homogeneous coords (u v 1)
    pixel_coords_homogeneous = torch.cat(
        (pixel_coords, torch.ones((height, width, 1), device=device)), dim=-1
    )
    pixel_coords_homogeneous = pixel_coords_homogeneous.permute(2, 0, 1).flatten(
        1
    )  # (3, H*W)
    # Apply K^-1 @ (u v 1): [B, 3, 3] @ [3, H*W] -> [B, 3, H*W]
    unprojected_points = torch.matmul(
        torch.inverse(camera_intrinsics), pixel_coords_homogeneous
    )  # (B, 3, H*W)
    unprojected_points = unprojected_points.view(
        batch_size, 3, height, width
    )  # (B, 3, H, W)
    unprojected_points = unprojected_points * depth  # (B, 3, H, W)
    return unprojected_points


@torch.jit.script
def project_points(
    points_3d: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
    image_shape: Tuple[int, int],
) -> torch.Tensor:
    # Project 3D points onto the image plane via intrinsics (u v w) = (x y z) @ K^T
    points_2d = torch.matmul(points_3d, intrinsic_matrix.transpose(1, 2))

    # Normalize projected points: (u v w) -> (u / w, v / w, 1)
    points_2d = points_2d[..., :2] / points_2d[..., 2:]

    points_2d = points_2d.int()

    # points need to be inside the image (can it diverge onto all points out???)
    valid_mask = (
        (points_2d[..., 0] >= 0)
        & (points_2d[..., 0] < image_shape[1])
        & (points_2d[..., 1] >= 0)
        & (points_2d[..., 1] < image_shape[0])
    )

    # Calculate the flat indices of the valid pixels
    flat_points_2d = points_2d[..., 0] + points_2d[..., 1] * image_shape[1]
    flat_indices = flat_points_2d.long()

    # Create depth maps and counts using scatter_add, (B, H, W)
    depth_maps = torch.zeros(
        [points_3d.shape[0], *image_shape], device=points_3d.device
    )
    counts = torch.zeros([points_3d.shape[0], *image_shape], device=points_3d.device)

    # Loop over batches to apply masks and accumulate depth/count values
    for i in range(points_3d.shape[0]):
        valid_indices = flat_indices[i, valid_mask[i]]
        depth_maps[i].view(-1).scatter_add_(
            0, valid_indices, points_3d[i, valid_mask[i], 2]
        )
        counts[i].view(-1).scatter_add_(
            0, valid_indices, torch.ones_like(points_3d[i, valid_mask[i], 2])
        )

    # Calculate mean depth for each pixel in each batch
    mean_depth_maps = depth_maps / counts.clamp(min=1.0)
    return mean_depth_maps.reshape(-1, 1, *image_shape)  # (B, 1, H, W)


@torch.jit.script
def downsample(data: torch.Tensor, downsample_factor: int = 2):
    N, _, H, W = data.shape
    data = data.view(
        N,
        H // downsample_factor,
        downsample_factor,
        W // downsample_factor,
        downsample_factor,
        1,
    )
    data = data.permute(0, 1, 3, 5, 2, 4).contiguous()
    data = data.view(-1, downsample_factor * downsample_factor)
    data_tmp = torch.where(data == 0.0, 1e5 * torch.ones_like(data), data)
    data = torch.min(data_tmp, dim=-1).values
    data = data.view(N, 1, H // downsample_factor, W // downsample_factor)
    data = torch.where(data > 1000, torch.zeros_like(data), data)
    return data


@torch.jit.script
def flat_interpolate(
    flat_tensor: torch.Tensor,
    old: Tuple[int, int],
    new: Tuple[int, int],
    antialias: bool = True,
    mode: str = "bilinear",
) -> torch.Tensor:
    if old[0] == new[0] and old[1] == new[1]:
        return flat_tensor
    tensor = flat_tensor.view(flat_tensor.shape[0], old[0], old[1], -1).permute(
        0, 3, 1, 2
    )  # b c h w
    tensor_interp = F.interpolate(
        tensor,
        size=(new[0], new[1]),
        mode=mode,
        align_corners=False,
        antialias=antialias,
    )
    flat_tensor_interp = tensor_interp.view(
        flat_tensor.shape[0], -1, new[0] * new[1]
    ).permute(
        0, 2, 1
    )  # b (h w) c
    return flat_tensor_interp.contiguous()


@torch.jit.script
def dilate(image, kernel_size: int | tuple[int, int]):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    device, dtype = image.device, image.dtype
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    kernel = torch.ones((1, 1, *kernel_size), dtype=torch.float32, device=image.device)
    dilated_image = F.conv2d(image.float(), kernel, padding=padding, stride=1)
    dilated_image = torch.where(
        dilated_image > 0,
        torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device),
    )
    return dilated_image.to(dtype)


@torch.jit.script
def erode(image, kernel_size: int | tuple[int, int]):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    device, dtype = image.device, image.dtype
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    kernel = torch.ones((1, 1, *kernel_size), dtype=torch.float32, device=image.device)
    eroded_image = F.conv2d(image.float(), kernel, padding=padding, stride=1)
    eroded_image = torch.where(
        eroded_image == (kernel_size[0] * kernel_size[1]),
        torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device),
    )
    return eroded_image.to(dtype)


@torch.jit.script
def iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    device = mask1.device

    # Ensure the masks are binary (0 or 1)
    mask1 = mask1.to(torch.bool)
    mask2 = mask2.to(torch.bool)

    # Compute intersection and union
    intersection = torch.sum(mask1 & mask2).to(torch.float32)
    union = torch.sum(mask1 | mask2).to(torch.float32)

    # Compute IoU
    iou = intersection / union.clip(min=1.0)

    return iou


# Our own version of `e3nn.o3.matrix_to_angles(R)` to avoid max tolerance errors
def matrix_to_angles(R):

    assert torch.allclose(torch.det(R), R.new_tensor(1), rtol=1e-02, atol=1e-06), "Invalid Rotation Matrix"
    x = R @ R.new_tensor([0.0, 1.0, 0.0])
    a, b = xyz_to_angles(x)
    R = angles_to_matrix(a, b, torch.zeros_like(a)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c


def gather_nearest_patch_center_samples(depth, rays, points, mask, patch_size):
        
        B, _, H, W = rays.shape
        num_patches_y, num_patches_x = H // patch_size, W // patch_size

        # Extract patches
        depth_patches = F.unfold(depth, kernel_size=patch_size, stride=patch_size)
        ray_patches = F.unfold(rays, kernel_size=patch_size, stride=patch_size)
        point_patches = F.unfold(points, kernel_size=patch_size, stride=patch_size) 
        mask_patches = F.unfold(mask.float(), kernel_size=patch_size, stride=patch_size)

        depth_patches = depth_patches.view(B, 1, patch_size * patch_size, num_patches_y * num_patches_x)
        ray_patches = ray_patches.view(B, 3, patch_size * patch_size, num_patches_y * num_patches_x)
        point_patches = point_patches.view(B, 3, patch_size * patch_size, num_patches_y * num_patches_x) 
        mask_patches = mask_patches.view(B, 1, patch_size * patch_size, num_patches_y * num_patches_x)

        # Pixel-center coordinates
        y, x = torch.meshgrid(
            torch.arange(patch_size, device=rays.device),
            torch.arange(patch_size, device=rays.device),
            indexing="ij"
        )
        coords = torch.stack([y + 0.5, x + 0.5], dim=-1).reshape(-1, 2).float()
        center = torch.tensor([patch_size / 2, patch_size / 2], device=rays.device).view(1, 2)

        # Distance to patch center
        dists = torch.norm(coords - center, dim=-1)
        sorted_indices = torch.argsort(dists)

        # Sort by distance
        depth_sorted = depth_patches[:, :, sorted_indices, :]
        rays_sorted = ray_patches[:, :, sorted_indices, :]
        points_sorted = point_patches[:, :, sorted_indices, :]  
        mask_sorted = mask_patches[:, :, sorted_indices, :]

        valid = mask_sorted.bool()
        patch_has_valid = valid.any(dim=2, keepdim=True)

        # Safe argmax to select nearest valid values
        nearest_idx = valid.float().argmax(dim=2, keepdim=True)

        # Gather depths
        choosen_depths = depth_sorted.gather(2, nearest_idx).squeeze(2)
        choosen_depths = choosen_depths.view(B, 1, num_patches_y, num_patches_x)

        # Gather rays
        chosen_rays = rays_sorted.gather(2, nearest_idx.expand(-1, 3, -1, -1)).squeeze(2)
        chosen_rays = chosen_rays.view(B, 3, num_patches_y, num_patches_x)

        # Gather rays
        chosen_points = points_sorted.gather(2, nearest_idx.expand(-1, 3, -1, -1)).squeeze(2)
        chosen_points = chosen_points.view(B, 3, num_patches_y, num_patches_x)

        # Return a new mask for patches
        patches_mask = patch_has_valid.view(B, 1, num_patches_y, num_patches_x)

        # Compute patch offsets
        patch_y = torch.arange(num_patches_y, device=rays.device) * patch_size
        patch_x = torch.arange(num_patches_x, device=rays.device) * patch_size
        patch_y_grid, patch_x_grid = torch.meshgrid(patch_y, patch_x, indexing="ij")
        nearest_idx_flat = nearest_idx.view(B, num_patches_y, num_patches_x)

        # Convert to patch-local coordinates
        y_coords = nearest_idx_flat // patch_size
        x_coords = nearest_idx_flat % patch_size

        # Add patch offsets (broadcasting works)
        y_coords = y_coords + patch_y_grid.unsqueeze(0)
        x_coords = x_coords + patch_x_grid.unsqueeze(0)
        selected_points = torch.stack((x_coords, y_coords), dim=1)

        return choosen_depths, chosen_rays, chosen_points, patches_mask, selected_points