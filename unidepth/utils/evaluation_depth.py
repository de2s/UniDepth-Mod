from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch3d.loss import chamfer_distance

from unidepth.utils.chamfer_distance import ChamferDistance


chamfer_cls = ChamferDistance()


def chamfer_dist(tensor1, tensor2):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    return (torch.sqrt(dist1) + torch.sqrt(dist2)) / 2


def auc(tensor1, tensor2, thresholds):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    # compute precision recall
    precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
    recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
    auc_value = torch.trapz(
        torch.tensor(precisions, device=tensor1.device),
        torch.tensor(recalls, device=tensor1.device),
    )
    return auc_value


def delta(tensor1, tensor2, exponent):
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < 1.25**exponent).to(torch.float32).mean()


def tau(tensor1, tensor2, perc):
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < (1.0 + perc)).to(torch.float32).mean()


def ssi(tensor1, tensor2):
    stability_mat = 1e-9 * torch.eye(2, device=tensor1.device)
    tensor2_one = torch.stack(
        [tensor2.detach(), torch.ones_like(tensor2).detach()], dim=1
    )
    scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (
        tensor2_one.T @ tensor1.unsqueeze(1)
    )
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return tensor2 * scale + shift


def si(tensor1, tensor2):
    return tensor2 * torch.median(tensor1) / torch.median(tensor2)


def arel(tensor1, tensor2):
    tensor2 = tensor2 * torch.median(tensor1) / torch.median(tensor2)
    return (torch.abs(tensor1 - tensor2) / tensor1).mean()


def d_auc(tensor1, tensor2):
    exponents = torch.linspace(0.01, 5.0, steps=100, device=tensor1.device)
    deltas = [delta(tensor1, tensor2, exponent) for exponent in exponents]
    return torch.trapz(torch.tensor(deltas, device=tensor1.device), exponents) / 5.0


def f1_score(tensor1, tensor2, thresholds):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    # compute precision recall
    precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
    recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
    precisions = torch.tensor(precisions, device=tensor1.device)
    recalls = torch.tensor(recalls, device=tensor1.device)
    f1_thresholds = 2 * precisions * recalls / (precisions + recalls)
    f1_thresholds = torch.where(
        torch.isnan(f1_thresholds), torch.zeros_like(f1_thresholds), f1_thresholds
    )
    f1_value = torch.trapz(f1_thresholds) / len(thresholds)
    return f1_value

def filter_by_occupancy_grid_iou(tensor1, tensor2, num_div=100.0, dilate=1):

    min1, _ = tensor1.min(dim=0)
    max1, _ = tensor1.max(dim=0)
    
    min2, _ = tensor2.min(dim=0)
    max2, _ = tensor2.max(dim=0)

    # 1. Compute bounding box and voxel size
    grid_min = torch.min(min1, min2)
    grid_max = torch.max(max1, max2)
    voxel_size = torch.dist(grid_max, grid_min) / num_div

    EPS = 1e-8
    if voxel_size < EPS:
        print(f"Voxel size close to zero: {voxel_size}")
        return None, None

    # 2. Convert points to voxel indices
    def points_to_voxels(points):
        indices = ((points - grid_min) / voxel_size).long()
        return indices
    
    vox1 = points_to_voxels(tensor1)
    vox2 = points_to_voxels(tensor2)
    
    # 3. Create occupancy grids
    grid_size = ((grid_max - grid_min) / voxel_size).ceil().long()
    if (grid_size == 0).any():
        print(f"Empty grid size: {grid_size}")
        return None, None
    
    if (vox1.min(dim=0)[0] < 0).any() or ((grid_size - vox1.max(dim=0)[0]) <= 0).any() or (vox2.min(dim=0)[0] < 0).any() or ((grid_size - vox2.max(dim=0)[0]) <= 0).any():
        print(f"Voxel indices outside gridsize range:")
        print(f"gridsize: {grid_size}")
        print(f"vox1: {vox1.min(dim=0)[0]} - {vox1.max(dim=0)[0]}")
        print(f"vox1: {vox2.min(dim=0)[0]} - {vox2.max(dim=0)[0]}")
        return None, None
    
    occ1 = torch.zeros(grid_size.tolist(), dtype=torch.bool, device=tensor1.device)
    occ2 = torch.zeros(grid_size.tolist(), dtype=torch.bool, device=tensor1.device)
    
    occ1[vox1[:,0], vox1[:,1], vox1[:,2]] = True
    occ2[vox2[:,0], vox2[:,1], vox2[:,2]] = True
   
    # 4. Dilate occupancy grids by 1 voxel using max pooling (cheaper than conv)
    kernel_size = 2 * dilate + 1
    if kernel_size > 1:
        occ1 = F.max_pool3d(occ1[None, None].float(), kernel_size=kernel_size, stride=1, padding=dilate)[0, 0].bool()
        occ2 = F.max_pool3d(occ2[None, None].float(), kernel_size=kernel_size, stride=1, padding=dilate)[0, 0].bool()
    
    # 6. Filter points based on intersection of occupancy gri
    intersection = occ1 & occ2
    if intersection.sum() == 0:
        print("Empty intersection")
        return None, None

    mask1 = intersection[vox1[:,0], vox1[:,1], vox1[:,2]]
    mask2 = intersection[vox2[:,0], vox2[:,1], vox2[:,2]]
    
    # Return none if all points are masked out
    if mask1.sum() == 0 or mask2.sum() == 0:
        print(f"Empty mask:")
        print(f"Mask1: {mask1.sum()}")
        print(f"Mask2: {mask1.sum()}")
        return None, None
    
    tensor1 = tensor1[mask1]
    tensor2 = tensor2[mask2]
    
    return tensor1, tensor2

def consistency(tensors, masks=None): #Shapes: torch.Size([307200, 3]), torch.Size([307200, 3])

    # Check dimensions
    assert tensors.shape[0] == 2
    if masks is not None:
        assert masks.shape[0] == 2

    # Filter masked points if mask is provided
    tensor1 = tensors[0][masks[0]] if masks is not None else tensors[0]
    tensor2 = tensors[1][masks[1]] if masks is not None else tensors[1]

    # Fallback if no points survived (avoid empty Chamfer distance)
    if tensor1.shape[0] == 0 or tensor2.shape[0] == 0:
        return None

    # Filter points that are not registered in both images
    tensor1, tensor2 = filter_by_occupancy_grid_iou(tensor1, tensor2, num_div=100, dilate=1)

    # Fallback if no points survived (avoid empty Chamfer distance)
    if tensor1 == None or tensor2 == None or tensor1.shape[0] == 0 or tensor2.shape[0] == 0:
        return None
    
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)

    return chamfer_distance(tensor1, tensor2, batch_reduction="mean", point_reduction="mean")[0]

DICT_METRICS = {
    "d1": partial(delta, exponent=1.0),
    "d2": partial(delta, exponent=2.0),
    "d3": partial(delta, exponent=3.0),
    "rmse": lambda gt, pred: torch.sqrt(((gt - pred) ** 2).mean()),
    "rmselog": lambda gt, pred: torch.sqrt(
        ((torch.log(gt) - torch.log(pred)) ** 2).mean()
    ),
    "arel": lambda gt, pred: (torch.abs(gt - pred) / gt).mean(),
    "sqrel": lambda gt, pred: (((gt - pred) ** 2) / gt).mean(),
    "log10": lambda gt, pred: torch.abs(torch.log10(pred) - torch.log10(gt)).mean(),
    "silog": lambda gt, pred: 100 * torch.std(torch.log(pred) - torch.log(gt)).mean(),
    "medianlog": lambda gt, pred: 100 * (torch.log(pred) - torch.log(gt)).median().abs(),
    "d_auc": d_auc,
    "tau": partial(tau, perc=0.03),
}


DICT_METRICS_3D = {
    "MSE_3d": lambda gt, pred, thresholds: torch.norm(gt - pred, dim=0, p=2),
    "chamfer": lambda gt, pred, thresholds: chamfer_dist(
        gt.unsqueeze(0).permute(0, 2, 1), pred.unsqueeze(0).permute(0, 2, 1)
    ),
    "F1": lambda gt, pred, thresholds: f1_score(
        gt.unsqueeze(0).permute(0, 2, 1),
        pred.unsqueeze(0).permute(0, 2, 1),
        thresholds=thresholds,
    ),
}

DICT_METRICS_D = {
    "a1": lambda gt, pred: (torch.maximum((gt / pred), (pred / gt)) > 1.25**1.0).to(
        torch.float32
    ),
    "abs_rel": lambda gt, pred: (torch.abs(gt - pred) / gt),
}

DICT_METRICS_SEQUENCE = {
    "gts_consistency": lambda gts, pred, mask: consistency(gts, mask),
    "pred_consistenc_masked": lambda gts, pred, mask: consistency(pred, mask),
    "pred_consistency_unmasked": lambda gts, pred, mask: consistency(pred),
}

def eval_depth(
    gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, max_depth=None
):
    summary_metrics = defaultdict(list)
    preds = F.interpolate(preds, gts.shape[-2:], mode="bilinear")
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        if max_depth is not None:
            mask = mask & (gt <= max_depth)
        for name, fn in DICT_METRICS.items():
            if name in ["tau", "d1", "arel"]:
                for rescale_fn in ["ssi", "si"]:
                    summary_metrics[f"{name}_{rescale_fn}"].append(
                        fn(gt[mask], eval(rescale_fn)(gt[mask], pred[mask]))
                    )
            summary_metrics[name].append(fn(gt[mask], pred[mask]).mean())
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def eval_3d(
    gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, thresholds=None
):
    summary_metrics = defaultdict(list)
    ratio = min(
        1.0, (240 * 320 / masks.sum()) ** 0.5
    )  # rescale to avoid OOM during eval, FIXME
    h_max, w_max = int(gts.shape[-2] * ratio), int(gts.shape[-1] * ratio)
    gts = F.interpolate(gts, size=(h_max, w_max), mode="nearest-exact")
    preds = F.interpolate(preds, size=(h_max, w_max), mode="nearest-exact")
    masks = F.interpolate(
        masks.float(), size=(h_max, w_max), mode="nearest-exact"
    ).bool()
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        if not torch.any(mask):
            continue
        for name, fn in DICT_METRICS_3D.items():
            summary_metrics[name].append(
                fn(gt[:, mask.squeeze()], pred[:, mask.squeeze()], thresholds).mean()
            )
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}

def eval_sequence(
    gts: torch.Tensor,
    preds: torch.Tensor,
    masks: torch.Tensor,
    cam2ws: torch.Tensor, 
):
    assert  gts.shape[0] == 2 and preds.shape[0] == 2 and masks.shape[0] and cam2ws.shape[0] == 2

    # Transform to global coordinates
    preds_flat = rearrange(preds, "b c h w -> b (h w) c")
    gts_flat = rearrange(gts, "b c h w -> b (h w) c")
    masks_flat = rearrange(masks, "b c h w -> b (h w) c").squeeze(-1)
    B, N, _ = preds_flat.shape
    ones = torch.ones(B, N, 1, device=preds_flat.device)
    gts_flat = torch.cat([gts_flat, ones], dim=-1)
    gts_flat = torch.bmm(cam2ws, gts_flat.transpose(-2, -1))
    gts_flat = gts_flat.permute(0, 2, 1)[:, :, :3]
    preds_flat = torch.cat([preds_flat, ones], dim=-1)
    preds_flat = torch.bmm(cam2ws, preds_flat.transpose(-2, -1))
    preds_flat = preds_flat.permute(0, 2, 1)[:, :, :3]
     
    # Calculate metrics
    summary_metrics = defaultdict(list)

    for name, fn in DICT_METRICS_SEQUENCE.items():
        value = fn(gts_flat, preds_flat, masks_flat)
        if value: summary_metrics[name].append(value.mean())

    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}