import torch


def bbox_intersections(points0, points1):

    # Bounding boxes: min and max corners
    min0, _ = points0.min(dim=0)
    max0, _ = points0.max(dim=0)
    min1, _ = points1.min(dim=0)
    max1, _ = points1.max(dim=0)

    # Intersection box
    inter_min = torch.maximum(min0, min1)
    inter_max = torch.minimum(max0, max1)

    # Clamp intersection dimensions to >= 0
    inter_dims = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_dims.prod()

    # Volumes of individual boxes
    vol0 = (max0 - min0).prod()
    vol1 = (max1 - min1).prod()

    # Union
    union_vol = vol0 + vol1 - inter_vol

    # IoU (avoid division by zero)
    iou = inter_vol / union_vol.clamp(min=1e-8)
    ioa = inter_vol / vol0.clamp(min=1e-8)
    iob = inter_vol / vol1.clamp(min=1e-8)

    return iou, ioa, iob


def voxelized_intersections(points0, points1, factor=0.02):


    # Compute global min/max
    min_coords = torch.minimum(points0.min(dim=0).values, points1.min(dim=0).values)
    max_coords = torch.maximum(points0.max(dim=0).values, points1.max(dim=0).values)
    voxel_size = (max_coords - min_coords).max() * factor

    # Shift points so min is at origin
    points0_shifted = points0 - min_coords
    points1_shifted = points1 - min_coords

    # Convert to voxel coordinates
    vox0 = torch.floor(points0_shifted / voxel_size).long()
    vox1 = torch.floor(points1_shifted / voxel_size).long()
    
    # Flatten voxel coordinates into unique IDs
    max_coords_vox = torch.cat([vox0, vox1], dim=0).max(dim=0).values + 1
    ids_a = vox0[:, 0] + vox0[:, 1] * max_coords_vox[0] + vox0[:, 2] * max_coords_vox[0] * max_coords_vox[1]
    ids_b = vox1[:, 0] + vox1[:, 1] * max_coords_vox[0] + vox1[:, 2] * max_coords_vox[0] * max_coords_vox[1]

    # Compute intersection and union
    ua = torch.unique(ids_a)
    ub = torch.unique(ids_b)
    inter = torch.sum(torch.isin(ua, ub)).float()
    union = ua.numel() + ub.numel() - inter
    iou = inter / union if union > 0 else torch.tensor(0.0)
    ioa = inter / ua.numel() if ua.numel() > 0 else torch.tensor(0.0)
    iob = inter / ub.numel() if ub.numel() > 0 else torch.tensor(0.0)

    return iou, ioa, iob


