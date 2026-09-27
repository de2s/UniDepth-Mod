import json5
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler
import torchvision
from PIL import Image
from functools import partial
import gc
import gzip

import unidepth.datasets as datasets
from unidepth.datasets import collate_fn
from unidepth.models import UniDepthV1
from unidepth.utils import colorize, image_grid
from unidepth.utils.validation import original_image

import matplotlib.pyplot as plt
import matplotlib


"""
METRICS:
Dataset: DDAD, min_depth: 0.0, max_depth: 230.7109375, max_error: 10.653860092163086
Dataset: DiodeIndoor, min_depth: 0.0, max_depth: 3.59765625, max_error: 0.3686502277851105
Dataset: HAMMER, min_depth: 0.0, max_depth: 1.0700000524520874, max_error: 1.633978247642517
Dataset: IBims, min_depth: 0.0, max_depth: 50.0, max_error: 206.04718017578125
Dataset: KITTI, min_depth: 0.0, max_depth: 82.99609375, max_error: 1.3681507110595703
Dataset: Nuscenes, min_depth: 0.0, max_depth: 103.60546875, max_error: 5.13720703125
Dataset: NYUv2Depth, min_depth: 0.0, max_depth: 10.0, max_error: 2.0212600231170654
Dataset: SUNRGBD, min_depth: 0.0, max_depth: 10.0, max_error: 3.0381264686584473
Dataset: VOID, min_depth: 0.0, max_depth: 5.15625, max_error: 1.7412974834442139
Dataset: ETH3D, min_depth: 0.0, max_depth: 6.747000217437744, max_error: 2.3731789588928223
Dataset: Argoverse2, min_depth: 0.0, max_depth: 213.2265625, max_error: 15.624911308288574
Dataset: ScanNet, min_depth: 0.0, max_depth: 3.065999984741211, max_error: 0.5790858864784241
Dataset: Waymo, min_depth: 0.0, max_depth: 74.47265625, max_error: 2.5550014972686768
Dataset: KITTIMulti, min_depth: 0.0, max_depth: 167.8359375, max_error: 0.7049074769020081
Dataset: BlendedMVG, min_depth: 0.0, max_depth: 125.87000274658203, max_error: 50.594322204589844
Dataset: DTURMVD, min_depth: 0.0, max_depth: 0.890999972820282, max_error: 4.10372257232666
"""

max_depths = {
    "DDAD": 80.0, # 230.7109375, (80)
    "DiodeIndoor": 4.0, # 3.59765625, (20)
    "HAMMER": 3.0, # 5.0, # 1.0700000524520874 (5)
    "IBims": 12.0, # 14.0, # 50.0, (14)
    "KITTI": 80.0, # 82.99609375 (80)
    "Nuscenes": 80.0, # 103.60546875, (80)
    "NYUv2Depth": 5.0, #10.0, # 10.0, (10)
    "SUNRGBD": 10.0, # 10.0, (10)
    "VOID": 8.0, # 10.0, # 5.15625, (10)
    "ETH3D": 20.0, # 30.0, # 6.747000217437744, (30)
    "Argoverse2": 80.0, # 213.2265625,
    "ScanNet": 10.0, # 20.0, # 3.065999984741211,
    "Waymo": 80.0, # 74.47265625,
    "KITTIMulti": 80.0, # 167.8359375,
    "BlendedMVG": 100.0, # 125.87000274658203,
    "DTURMVD": 5.0, # 0.890999972820282,
    "ARKit": 10.0,
}

max_errors = {
    "DDAD": 0.5, # 10.653860092163086, (0.20)
    "DiodeIndoor": 0.4, # 0.3686502277851105, (0.30)
    "HAMMER": 1.0, # 1.5, # 1.633978247642517, (1.5)
    "IBims": 0.5, # 206.04718017578125, (0.5)
    "KITTI": 0.4, # 0.05, # 1.3681507110595703, (0.05)
    "Nuscenes": 0.20, # 5.13720703125, (0.20)
    "NYUv2Depth": 0.12, # 0.15, # 0.20, # 2.0212600231170654, (0.20)
    "SUNRGBD": 0.30, # 3.0381264686584473, (0.30)
    "VOID": 0.30, # 1.7412974834442139, (0.30)
    "ETH3D": 0.30, # 2.3731789588928223, (0.30)
    "Argoverse2": 0.20, # 15.624911308288574,
    "ScanNet": 0.10, # 0.20, # 0.5790858864784241,
    "Waymo": 0.05, # 2.5550014972686768,
    "KITTIMulti": 0.4, # 0.05, # 0.7049074769020081,
    "BlendedMVG": 1.0, # 0.5, # 50.594322204589844,
    "DTURMVD": 0.5, # 0.30, # 4.10372257232666,
    "ARKit": 0.2
}

experiment_list = [
    # # SELECTED RUNS FOR BASELINE:
    "21-Aug-2026 21-00 - Unidepth",
    # "19-Nov-2025 20-24 - Global - Baseline",
    # "19-Dec-2025 20-12 - Global - Invariance in SE3 - cos_sim",
    # "02-Jan-2026 09-09 - Global - Ablation - Disable Invariance and Equivariance",

    # # SELECTED RUNS FOR GLOBAL EQUIVARIANCE:
    # "19-Nov-2025 20-24 - Global - Baseline",
    # "01-Aug-2026 20-45 - Global - SE(3) to SE(n) - l2_dist",
    # "26-Nov-2025 12-37 - Global - SO(3) to SO(n) - (with change of base)",
    # "31-Aug-2026 23-22 - Global - SE(3) to PGL(n) - l2_dist",
    # "08-Aug-2026 18-43 - Global - SE(3) to GL(n) - l2_dist",

    # # SELECTED RUNS FOR LOCAL EQUIVARIANCE:
    # "14-Mar-2026 07-56 - Local - Baseline",
    # "18-Apr-2026 10-40 - Local - SE(3) to SE(n) - l2_dist (fixed max depth)",
    # "19-Mar-2026 04-51 - Local - SO(3) to SO(n)",
    # # "17-Aug-2026 20-36 - Local - SE(3) to PGL(n) - l2_dist",
    # "11-Sep-2026_20-28 - Local - SE(3) to PGL(n) - l2_dist (2)",
    # "26-Apr-2026 22-22 - Local - SE(3) to GL(n) - l2_dist (fixed max depth)",

    # # Fine Tuning - ScanNet
    # "17-May-2026 08-03 - Fine-tuning ScanNet - SO(3) to SO(n)",
    # "19-May-2026 13-03 - Fine-tuning ScanNet - Baseline",
    # "20-May-2026 07-11 - Fine-tuning ScanNet - SE(3) to SE(n)",

    # # Fine Tuning - Argoverse 2
    # "22-May-2026 21-57 - Fine-tuning Argoverse 2 - Baseline",
    # "23-May-2026 19-58 - Fine-tuning Argoverse 2 - SO(2) to SO(n)",
    # "26-May-2026 10-09 - Fine-tuning Argoverse 2 - SE(3) to SE(n)",
    
    # # Fine Tuning - KITTI MOTS
    # "28-May-2026 09-53 - Fine-tuning KITTI MOTS - Baseline",
    # "31-May-2026 15-23 - Fine-tuning KITTI MOTS - SO(3) to SO(n)",
    # "03-Jun-2026 10-57 - Fine-tuning KITTI-MOTS - SE(3) to SE(n)",
    
    # Fine Tuning - ARKitScenes
    "09-Jun-2026 12-24 - Fine-tuning ARKitScenes - Baseline",
    "10-Jun-2026 21-31 - Fine-tuning ARKitScenes - SO(3) to SO(n)",
    "13-Jun-2026 10-28 - Fine-tuning ARKitScenes - SE(3) to SE(n)",
]

checkpoint_dict = {
    # SELECTED RUNS FOR BASELINE:
    "21-Aug-2026 21-00 - Unidepth": "lpiccinelli/unidepth-v1-vitl14",
    "19-Nov-2025 20-24 - Global - Baseline": "Unidepth - 19-Nov-2025 20-24 - Global - Baseline.pth",
    "19-Dec-2025 20-12 - Global - Invariance in SE3 - cos_sim": "Unidepth - 19-Dec-2025 20-12 - Global - Invariance in SE3 - cos_sim.pth",
    "02-Jan-2026 09-09 - Global - Ablation - Disable Invariance and Equivariance": "Unidepth - 02-Jan-2026 09-09 - Global - Ablation - Disable Invariance and Equivariance.pth",

    # SELECTED RUNS FOR GLOBAL EQUIVARIANCE:
    "19-Nov-2025 20-24 - Global - Baseline": "Unidepth - 19-Nov-2025 20-24 - Global - Baseline.pth",
    "01-Aug-2026 20-45 - Global - SE(3) to SE(n) - l2_dist": "Unidepth - 01-Aug-2026 20-45 - Global - SE(3) to SE(n) - l2_dist.pth",
    "26-Nov-2025 12-37 - Global - SO(3) to SO(n) - (with change of base)": "Unidepth - 26-Nov-2025 12-37 - Global - SO(3) to SO(n) - (with change of base).pth",
    "31-Aug-2026 23-22 - Global - SE(3) to PGL(n) - l2_dist": "Unidepth - 31-Aug-2026 23-22 - Global - SE(3) to PGL(n) - l2_dist.pth",
    "08-Aug-2026 18-43 - Global - SE(3) to GL(n) - l2_dist": "Unidepth - 08-Aug-2026 18-43 - Global - SE(3) to GL(n) - l2_dist.pth",

    # SELECTED RUNS FOR LOCAL EQUIVARIANCE:
    "14-Mar-2026 07-56 - Local - Baseline": "Unidepth - 14-Mar-2026 07-56 - Local - Baseline.pth",
    "18-Apr-2026 10-40 - Local - SE(3) to SE(n) - l2_dist (fixed max depth)": "Unidepth - 18-Apr-2026 10-40 - Local - SE(3) to SE(n) - l2_dist (fixed max depth).pth",
    "19-Mar-2026 04-51 - Local - SO(3) to SO(n)": "Unidepth - 19-Mar-2026 04-51 - Local - SO(3) to SO(n).pth",
    "17-Aug-2026 20-36 - Local - SE(3) to PGL(n) - l2_dist": "Unidepth - 17-Aug-2026 20-36 - Local - SE(3) to PGL(n) - l2_dist.pth",
    "11-Sep-2026_20-28 - Local - SE(3) to PGL(n) - l2_dist (2)": "Unidepth - 11-Sep-2026 20-28 -  - Local - SE(3) to PGL(n) - l2_dist (2).pth",
    "26-Apr-2026 22-22 - Local - SE(3) to GL(n) - l2_dist (fixed max depth)": "Unidepth - 26-Apr-2026 22-22 - Local - SE(3) to GL(n) - l2_dist (fixed max depth).pth",

    # Fine Tuning - ScanNet
    "17-May-2026 08-03 - Fine-tuning ScanNet - SO(3) to SO(n)": 'Unidepth - 17-May-2026 08-03 - Fine-tuning ScanNet - SO(3) to SO(n).pth',
    "19-May-2026 13-03 - Fine-tuning ScanNet - Baseline": 'Unidepth - 19-May-2026 13-03 - Fine-tuning ScanNet - Baseline.pth',
    "20-May-2026 07-11 - Fine-tuning ScanNet - SE(3) to SE(n)": 'Unidepth - 20-May-2026 07-11 - Fine-tuning ScanNet - SE(3) to SE(n).pth',

    # Fine Tuning - Argoverse 2
    "22-May-2026 21-57 - Fine-tuning Argoverse 2 - Baseline": 'Unidepth - 22-May-2026 21-57 - Fine-tuning Argoverse 2 - Baseline.pth',
    "23-May-2026 19-58 - Fine-tuning Argoverse 2 - SO(2) to SO(n)": 'Unidepth - 23-May-2026 19-58 - Fine-tuning Argoverse 2 - SO(2) to SO(n).pth',
    "26-May-2026 10-09 - Fine-tuning Argoverse 2 - SE(3) to SE(n)": 'Unidepth - 26-May-2026 10-09 - Fine-tuning Argoverse 2 - SE(3) to SE(n).pth',
    
    # Fine Tuning - KITTI MOTS
    "28-May-2026 09-53 - Fine-tuning KITTI MOTS - Baseline": 'Unidepth - 28-May-2026 09-53 - Fine-tuning KITTI MOTS - Baseline.pth',
    "31-May-2026 15-23 - Fine-tuning KITTI MOTS - SO(3) to SO(n)": 'Unidepth - 31-May-2026 15-23 - Fine-tuning KITTI MOTS - SO(3) to SO(n).pth',
    "03-Jun-2026 10-57 - Fine-tuning KITTI-MOTS - SE(3) to SE(n)": 'Unidepth - 03-Jun-2026 10-57 - Fine-tuning KITTI-MOTS - SE(3) to SE(n).pth',
    
    # Fine Tuning - ARKitScenes
    "09-Jun-2026 12-24 - Fine-tuning ARKitScenes - Baseline": 'Unidepth - 09-Jun-2026 12-24 - Fine-tuning ARKitScenes - Baseline.pth',
    "10-Jun-2026 21-31 - Fine-tuning ARKitScenes - SO(3) to SO(n)": 'Unidepth - 10-Jun-2026 21-31 - Fine-tuning ARKitScenes - SO(3) to SO(n).pth',
    "13-Jun-2026 10-28 - Fine-tuning ARKitScenes - SE(3) to SE(n)": 'Unidepth - 13-Jun-2026 10-28 - Fine-tuning ARKitScenes - SE(3) to SE(n).pth',
}

experiment_shortname_dict = {

    # SELECTED RUNS FOR BASELINE:
    "21-Aug-2026 21-00 - Unidepth":  "unidepth",
    "19-Nov-2025 20-24 - Global - Baseline": "baseline_1",
    "19-Dec-2025 20-12 - Global - Invariance in SE3 - cos_sim": "se3_invariance",
    "02-Jan-2026 09-09 - Global - Ablation - Disable Invariance and Equivariance": "no_invariance",

    # SELECTED RUNS FOR GLOBAL EQUIVARIANCE:
    "01-Aug-2026 20-45 - Global - SE(3) to SE(n) - l2_dist": "global_se3_sen_l2",
    "26-Nov-2025 12-37 - Global - SO(3) to SO(n) - (with change of base)": "global_so3_son_sc",
    "31-Aug-2026 23-22 - Global - SE(3) to PGL(n) - l2_dist": "global_se3_pgln_l2",
    "08-Aug-2026 18-43 - Global - SE(3) to GL(n) - l2_dist": "global_se3_gln_l2",

    # SELECTED RUNS FOR LOCAL EQUIVARIANCE:
    "14-Mar-2026 07-56 - Local - Baseline": "baseline_2",
    "18-Apr-2026 10-40 - Local - SE(3) to SE(n) - l2_dist (fixed max depth)": "local_se3_sen_l2_fix",
    "19-Mar-2026 04-51 - Local - SO(3) to SO(n)": "local_so3_son_sc",
    "17-Aug-2026 20-36 - Local - SE(3) to PGL(n) - l2_dist":  "local_se3_pgln_l2",
    "11-Sep-2026_20-28 - Local - SE(3) to PGL(n) - l2_dist (2)": "local_se3_pgln_l2_2",
    "26-Apr-2026 22-22 - Local - SE(3) to GL(n) - l2_dist (fixed max depth)": "local_se3_gln_l2_fix",

    # FINE TUNING - SCANNET
    "19-May-2026 13-03 - Fine-tuning ScanNet - Baseline": "baseline_3",
    "20-May-2026 07-11 - Fine-tuning ScanNet - SE(3) to SE(n)": "finetuning_se3_sen_l2",
    "17-May-2026 08-03 - Fine-tuning ScanNet - SO(3) to SO(n)": "finetuning_so3_son_sc",

    # FINE TUNING - ARGOVERSE 2
    "22-May-2026 21-57 - Fine-tuning Argoverse 2 - Baseline": "baseline_3",
    "26-May-2026 10-09 - Fine-tuning Argoverse 2 - SE(3) to SE(n)": "finetuning_se3_sen_l2",
    "23-May-2026 19-58 - Fine-tuning Argoverse 2 - SO(2) to SO(n)": "finetuning_so3_son_sc",
    
    # FINE TUNING - KITTI MOTS
    "28-May-2026 09-53 - Fine-tuning KITTI MOTS - Baseline": "baseline_3",
    "03-Jun-2026 10-57 - Fine-tuning KITTI-MOTS - SE(3) to SE(n)":  "finetuning_se3_sen_l2",
    "31-May-2026 15-23 - Fine-tuning KITTI MOTS - SO(3) to SO(n)": "finetuning_so3_son_sc",
    
    # FINE TUNING - ARKITSCENES
    "09-Jun-2026 12-24 - Fine-tuning ARKitScenes - Baseline": "baseline_3",
    "13-Jun-2026 10-28 - Fine-tuning ARKitScenes - SE(3) to SE(n)": "finetuning_se3_sen_l2",
    "10-Jun-2026 21-31 - Fine-tuning ARKitScenes - SO(3) to SO(n)": "finetuning_so3_son_sc",

    # # OTHERS
    # "Unidepth - 26-Jan-2026 15-00 - Global - SO(3) to SO(n) (with identity penalty).pth":                   "global_so3_son_sc_lreg1",
    # "Unidepth - 11-Feb-2026 22-53 - Global - SO(3) to SO(n) (with Identity penalty 2).pth":                 "global_so3_son_sc_lreg2",
    # "Unidepth - 27-Mar-2026 23-24 - Local - SE(3) to GL(n) - cos_sim.pth":                                  "local_se3_gln_sc",
    # "Unidepth - 01-Apr-2026 18-25 - Local - SE(3) to GL(n) - l2_dist.pth":                                  "local_se3_gln_l2",
    # "Unidepth - 05-Apr-2026 09-49 - Local - SE(3) to SE(n) - l2_dist.pth":                                  "local_se3_sen_l2",
    # "Unidepth - 11-Apr-2026 16-31 - Local - SE(3) to SE(n) - cos_sim.pth":                                  "local_se3_sen_sc",
    # "Unidepth - 09-May-2026 23-06 - Local - SE(3) to SE(n) - l2_dist (relative traslation).pth":            "local_se3_sen_l2_fix2",
}

for experiment in experiment_list:
    if experiment not in checkpoint_dict and experiment not in experiment_shortname_dict:
        raise Exception(f"Experiment doesn't have a checkpoint or short name: '{experiment}'")

dataset_list = [
    # # Image datasets from UniDepth
    # "DDAD",
    # "DiodeIndoor",
    # "HAMMER",
    # "IBims",
    # "KITTI",
    # "Nuscenes",
    # "NYUv2Depth",
    # "SUNRGBD",
    # "VOID",
    # "ETH3D",
    # # New sequence datasets (test split)
    # "Argoverse2",
    # "ScanNet",
    # "Waymo",
    # # New sequence datasets (zero shot)
    # "KITTIMulti",
    # "BlendedMVG",
    # "DTURMVD",
    # #
    "ARKit",
]

rename_datasets_dict = {
    # In Domain (Sequence)
    "Argoverse2":  "argoverse2",
    "ScanNet":     "scannet",
    "Waymo":       "waymo",
    # Out of Domain (Sequence)
    "ARKit":       "arkitscenes",
    "BlendedMVG":  "blendedmvs",
    "DTURMVD":     "dturmvs",
    "KITTIMulti":  "kittimots",
    # Out of Domain (Image)
    "DDAD":        "ddad",
    "DiodeIndoor": "diode",
    "ETH3D":       "eth3d",
    "HAMMER":      "hammer",
    "IBims":       "ibims",
    "KITTI":       "kitti",
    "Nuscenes":    "nuscenes",
    "NYUv2Depth":  "nyu",
    "SUNRGBD":     "sunrgbd",
    "VOID":        "void",   
}

# selected_samples = {
#         "DDAD": list(range(10)),
#         "DiodeIndoor": list(range(10)),
#         "HAMMER": list(range(10)),
#         "IBims": list(range(10)),
#         "KITTI": list(range(10)),
#         "Nuscenes": list(range(10)),
#         "NYUv2Depth": list(range(10)),
#         "SUNRGBD": list(range(10)),
#         "VOID": list(range(10)),
#         "ETH3D": list(range(10)),
#         # New sequence datasets (test split)
#         "Argoverse2": list(range(10)),
#         "ScanNet": list(range(10)),
#         "Waymo": list(range(10)),
#         # New sequence datasets (zero shot)
#         "KITTIMulti": list(range(10)),
#         "BlendedMVG": list(range(10)),
#         "DTURMVD": list(range(10)),
# }

selected_samples = {
        "DDAD": [6],
        "DiodeIndoor": [4],
        "HAMMER": [0],
        "IBims": [4],
        "KITTI": [4],
        "Nuscenes": [4],
        "NYUv2Depth": [0],
        "SUNRGBD": [5],
        "VOID": [4],
        "ETH3D": [5],
        # New sequence datasets (test split)
        "Argoverse2": [5],
        "ScanNet": [7],
        "Waymo": [5],
        # New sequence datasets (zero shot)
        "KITTIMulti": [6],
        "BlendedMVG": [9],
        "DTURMVD": [3],
        #
        "ARKit": [22],
}

def generate_color_map(min, max, color_scale, image_name):

    # Configure size according to the dataset
    if "argoverse2" in image_name:
        fig, ax = plt.subplots(figsize=(0.8, 3.9287))
    elif "scannet" in image_name or "blendedmvs" in image_name or "dturmvs" in image_name or "diode" in image_name or "ibims" in image_name or "nyu" in image_name or "void" in image_name or "arkit" in image_name:
        fig, ax = plt.subplots(figsize=(0.8, 5.0792))
    elif "waymo" in image_name or "eth3d" in image_name:
        fig, ax = plt.subplots(figsize=(0.8, 4.5253)) 
    elif "kitti" in image_name:
        fig, ax = plt.subplots(figsize=(0.8, 2.0243)) 
    elif "ddad" in image_name or "nuscenes" in image_name:
        fig, ax = plt.subplots(figsize=(0.8, 3.8373))
    elif "hammer" in image_name or "sunrgbd" in image_name:
        fig, ax = plt.subplots(figsize=(0.8, 5.1655))
    else:
        raise Exception("Dimensions for dataset not found")

    fig.subplots_adjust(right=0.35) # Give room on the right for the scale labels


    # 2. Choose the colormap and define your data range (min and max)
    cmap = matplotlib.colormaps[color_scale] 
    norm = matplotlib.colors.Normalize(vmin=min, vmax=max)

    # 3. Create the standalone colorbar
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, 
        orientation='vertical', # Set orientation to vertical
        # label='Depth (m)'
    )

    plt.savefig(image_name)


def remove_padding(out, paddings):
    H, W = out.shape[-2:]
    outs = [
        o[..., padding[2] : H - padding[3], padding[0] : W - padding[1]]
        for padding, o in zip(paddings, out)
    ]
    return torch.stack(outs)


def resize_images(batch, preds):

    # Resize original image
    paddings = [
        torch.tensor(pads)
        for img_meta in batch["img_metas"]
        for pads in img_meta.get("paddings", [[0] * 4])
    ]
    paddings = torch.stack(paddings).to(batch["data"]["image_original"].device)[
        ..., [0, 2, 1, 3]
    ]  # lrtb

    T, _, H, W = batch["data"]["depth"].shape
    resized_images = []
    for i in range(T):
        p = paddings[i]
        if torch.uint8 != batch["data"]["image_original"][i].dtype:
            raise Exception("NEW TYPE NOT HANDLED")
        resized_images.append(
            F.interpolate(
                batch["data"]["image_original"][i].unsqueeze(0).float(),
                (H + p[2] + p[3], W + p[1] + p[2]), # TODO: IS THIS AN ERROR?
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).to(batch["data"]["image_original"].dtype)
        )
    batch["data"]["image_original"] = torch.cat(resized_images, dim=0)

    batch["data"]["image_original"] = remove_padding(
        batch["data"]["image_original"], paddings.repeat(T, 1)
    )
    
    # Resize predicted depth
    paddings = [
        torch.tensor(pads)
        for img_meta in batch["img_metas"]
        for pads in img_meta.get("depth_paddings", [[0] * 4])
    ]
    paddings = torch.stack(paddings).to(batch["data"]["depth"].device)[
        ..., [0, 2, 1, 3]
    ]  
    
    if preds is not None:
        for key in ["depth"]:
            if key in preds:
                # T, _, H, W = preds[key].shape
                T, _, H, W = batch["data"]["depth"].shape
                resized_preds = []
                for i in range(T):
                    p = paddings[i]
                    resized_preds.append(
                        F.interpolate(
                            preds[key],
                            (H + p[2] + p[3], W + p[1] + p[2]),
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )
                    )
                preds[key] = torch.cat(resized_preds, dim=0)

                preds[key] = remove_padding(preds[key], paddings.repeat(T, 1))

    return batch, preds


# def create_and_save_point_cloud(rgb, depth, intrinsics, filename):
#     """
#     Generates and saves a binary .ply point cloud using only NumPy.
#     """
#     h, w = depth.shape
#     u, v = np.meshgrid(np.arange(w), np.arange(h))
    
#     # Filter out invalid depth pixels
#     valid = depth > 0.0
#     z = depth[valid]
#     u = u[valid]
#     v = v[valid]
    
#     # Extract focal lengths and principal points from the K matrix
#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
#     # Unproject 2D pixels to 3D points
#     x = (u - cx) * z / fx
#     y = (v - cy) * z / fy
#     points = np.stack((x, y, z), axis=-1).astype(np.float32)
    
#     # Extract colors (already 0-255 in original image)
#     colors = rgb[valid][:, :3].astype(np.uint8)
    
#     # Create structured array for binary PLY format
#     vertices = np.empty(points.shape[0], dtype=[
#         ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
#     ])
    
#     vertices['x'] = points[:, 0]
#     vertices['y'] = points[:, 1]
#     vertices['z'] = points[:, 2]
#     vertices['red'] = colors[:, 0]
#     vertices['green'] = colors[:, 1]
#     vertices['blue'] = colors[:, 2]
    
#     # Write to binary PLY
#     with open(filename, 'wb') as f:
#         f.write(b"ply\n")
#         f.write(b"format binary_little_endian 1.0\n")
#         f.write(f"element vertex {len(points)}\n".encode('utf-8'))
#         f.write(b"property float x\n")
#         f.write(b"property float y\n")
#         f.write(b"property float z\n")
#         f.write(b"property uchar red\n")
#         f.write(b"property uchar green\n")
#         f.write(b"property uchar blue\n")
#         f.write(b"end_header\n")
#         f.write(vertices.tobytes())

def create_and_save_point_cloud(rgb, depth, intrinsics, filename):
    """
    Generates and saves a binary .ply point cloud using only NumPy.
    """
    h, w = depth.shape
    
    # NEW: Ensure RGB matches Depth dimensions to avoid IndexError
    if rgb.shape[:2] != (h, w):
        rgb_pil = Image.fromarray(rgb)
        # Resize RGB to match the (width, height) of the depth map
        rgb = np.array(rgb_pil.resize((w, h), Image.Resampling.BILINEAR))
        
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Filter out invalid depth pixels
    valid = depth > 0.0
    z = depth[valid]
    u = u[valid]
    v = v[valid]
    
    # Extract focal lengths and principal points from the K matrix
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Unproject 2D pixels to 3D points
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).astype(np.float32)
    
    # Extract colors
    colors = rgb[valid][:, :3].astype(np.uint8)
    
    # Create structured array for binary PLY format
    vertices = np.empty(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertices['x'] = points[:, 0]
    vertices['y'] = points[:, 2] # vertices['y'] = points[:, 1]
    vertices['z'] = - points[:, 1] # vertices['z'] = points[:, 2]
    vertices['red'] = colors[:, 0]
    vertices['green'] = colors[:, 1]
    vertices['blue'] = colors[:, 2]
    
    # Write to binary PLY
    with open(filename, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(points)}\n".encode('utf-8'))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        f.write(vertices.tobytes())

    # # Write to a GZIPPED binary PLY (add .gz to filename when calling the function)
    # with gzip.open(filename, 'wb') as f:
    #     f.write(b"ply\n")
    #     f.write(b"format binary_little_endian 1.0\n")
    #     f.write(f"element vertex {len(points)}\n".encode('utf-8'))
    #     f.write(b"property float x\n")
    #     f.write(b"property float y\n")
    #     f.write(b"property float z\n")
    #     f.write(b"property uchar red\n")
    #     f.write(b"property uchar green\n")
    #     f.write(b"property uchar blue\n")
    #     f.write(b"end_header\n")
    #     f.write(vertices.tobytes())


if __name__ == "__main__":

    from pathlib import Path
    current_dir = Path.cwd()
    print(current_dir)

    # Open configuration file
    with open("configs/eval_v1_vitl14_mod.json", "r", encoding="utf-8") as file:
        config = json5.load(file)

    # Get cuda device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Datasets
    val_datasets = {}
    for dataset_name in dataset_list:
        print(dataset_name)
        val_dataset: datasets.BaseDataset = getattr(datasets, dataset_name)
        val_datasets[dataset_name]=val_dataset(
            image_shape=config["data"]["image_shape"],
            split_file=val_dataset.test_split,
            test_mode=True,
            crop=config["data"].get("crop", "garg"),
            augmentations_db=config["data"].get("augmentations", {}),
            normalize=config["data"].get("normalization", "imagenet"),
            resize_method=config["data"].get("resize_method", "hard"),
            num_frames=config["data"].get("num_frames", 1),
            mini=1.0,
            sampling_mode=config["data"]["sampling_mode"],
            sampling_window_size=config["data"]["sampling_window_size"][dataset_name],
        )
        # # TODO: TEMP
        # break

    # Create Data Loaders
    val_batch_size = 1
    valid_samplers = {k: SequentialSampler(v) for k, v in val_datasets.items()}
    val_loaders = {
        name_dataset: DataLoader(
            dataset,
            batch_size = val_batch_size,
            shuffle = False,
            num_workers = 0,
            sampler = valid_samplers[name_dataset],
            pin_memory = False,
            drop_last = False,
            collate_fn = partial(collate_fn, is_batched=False),
        )
        for name_dataset, dataset in val_datasets.items()
    }

    # Iterate Models
    for experiment in experiment_list:
        print(f"Experiment: {experiment}")

        # Load model
        checkpoint_name = checkpoint_dict[experiment]
        if "lpiccinelli" in checkpoint_name:
            model = UniDepthV1.from_pretrained(checkpoint_name).to(device)
        else:
            model = UniDepthV1(config).to(device)
            checkpoint = torch.load(os.path.join("/data/checkpoints", checkpoint_name))
            model.load_state_dict(checkpoint['model_state'], strict=False)

        # Init metrics
        min_depth, max_depth = None, None
        max_error = None
        
        # Iterate datasets
        context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False)
        for name_dataset, loader in val_loaders.items():
            print(f"Dataset: {name_dataset}")

            # Get the maximum number of samples:
             
            if selected_samples[name_dataset]:
                max_iterations = max(selected_samples[name_dataset])
            else:
                continue

            color_map_generated = False

            # Iterate samples
            for i, batch in enumerate(loader):
                print(f"Sample: {i}")

                # Iterate first 5 samples
                if i > max_iterations:
                    break

                # Skip not included samples
                if i not in selected_samples[name_dataset]:
                    continue

                # Infer depth
                with context:
                    # Move data to gpu
                    batch["data"] = {
                        k: v.to(model.device) for k, v in batch["data"].items()
                    }
                    # Remove temporal dimension of the dataloder, here is always 1!
                    batch["data"] = {
                        k: torch.flatten(v, 0, 1) for k, v in batch["data"].items()
                    }
                    # Call model
                    # (batch["data"]["image"].shape, batch["data"]["image"].dtype, batch["data"]["image"].min(), batch["data"]["image"].max())
                    # (torch.Size([1, 3, 378, 672]), torch.float32, tensor(-2.1179, device='cuda:0'), tensor(2.6400, device='cuda:0'))
                    # (batch["data"]["image_original"].shape, batch["data"]["image_original"].dtype, batch["data"]["image_original"].min(), batch["data"]["image_original"].max())
                    # (torch.Size([1, 3, 378, 672]), torch.uint8, tensor(0, device='cuda:0', dtype=torch.uint8), tensor(255, device='cuda:0', dtype=torch.uint8))
                    preds, losses, metrics = model(batch["data"], batch["img_metas"], False, i)

                # Resize images
                batch, _ = original_image(batch, preds=None)

                # # Calculate evaluation metrics
                # loader.dataset.accumulate_metrics(
                #     inputs=batch["data"],
                #     preds=preds,
                #     keyframe_idx=batch["img_metas"][0].get("keyframe_idx"),
                # )
                # evaluation = loader.dataset.get_evaluation()
                

                # Resize remaining images
                resize_images(batch, preds)

                # Save image
                torchvision.io.write_png(batch["data"]["image_original"][0].cpu(), f"/unidepth/images/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_image.png")   # (torch.Size([1, 3, 378, 672]), torch.uint8)

                # Save gt depth
                depth_gt_col = colorize(batch["data"]["depth"][0].squeeze(0).cpu(), vmin=0.01, vmax=max_depths[name_dataset], cmap="magma_r")
                Image.fromarray(depth_gt_col).save(f"/unidepth/images/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_depth_gt.png")

                # Save predicted depth
                depth_col = colorize(preds["depth"][0].squeeze(0).detach().cpu(), vmin=0.01, vmax=max_depths[name_dataset], cmap="magma_r")
                Image.fromarray(depth_col).save(f"/unidepth/images/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_depth.png")

                # Generate color scale for depth
                if color_map_generated == False:
                    generate_color_map(0.0, max_depths[name_dataset], "magma_r", f"/unidepth/images/{rename_datasets_dict[name_dataset]}_depth_color_map.pdf")

                # # compute error, you have zero divison where depth_gt == 0.0
                # depth_arel = np.abs(depth_gt - depth_pred) / depth_gt
                # depth_arel[depth_gt == 0.0] = 0.0

                # Calculate Error
                depth_error = torch.abs(batch["data"]["depth"][0] - preds["depth"][0]) / batch["data"]["depth"][0]
                depth_error[batch["data"]["depth_mask"][0] == 0] = 0.0

                # Calculate Max/Min depth, Max/Min Error.
                if min_depth is None and max_depth is None and max_error is None:
                    min_depth, max_depth = batch["data"]["depth"][0].min().detach().cpu() , batch["data"]["depth"][0].max().detach().cpu()
                    max_error = depth_error.max().detach().cpu()
                else:
                    min_depth, max_depth = torch.min(batch["data"]["depth"][0].min().detach().cpu(), min_depth), torch.max(batch["data"]["depth"][0].max().detach().cpu(), max_depth)
                    max_error = torch.max(depth_error.max().detach().cpu(), max_error)

                # Save depth error
                depth_error_col = colorize(depth_error.squeeze(0).detach().cpu(), vmin=0.0, vmax=max_errors[name_dataset], cmap="coolwarm")
                Image.fromarray(depth_error_col).save(f"/unidepth/images/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_depth_error.png")

                # Generate color scale for error
                if color_map_generated == False:
                    generate_color_map(0.0, max_errors[name_dataset], "coolwarm", f"/unidepth/images/{rename_datasets_dict[name_dataset]}_depth_error_color_map.pdf")
                    color_map_generated = True

                # --- NEW: Extracting Arrays for Point Clouds ---

                # # 1. Image (H, W, 3 format for Open3D)
                # img_np = batch["data"]["image_original"][0].cpu().numpy()
                # img_np = np.transpose(img_np, (1, 2, 0)) # from (3, H, W) to (H, W, 3)

                # # 2. GT and Predicted Depth (H, W)
                # depth_gt_np = batch["data"]["depth"][0].squeeze(0).cpu().numpy()
                # depth_pred_np = preds["depth"][0].squeeze(0).detach().cpu().numpy()

                # # 3. Fetch Camera Intrinsics
                # # Try getting the original restored intrinsics from img_metas first, 
                # # fallback to batch['data']['intrinsics']
                # if isinstance(batch["img_metas"][0], dict) and "intrinsics" in batch["img_metas"][0]:
                #     intrinsics_np = np.array(batch["img_metas"][0]["intrinsics"])
                # else:
                #     intrinsics_np = batch["data"]["intrinsics"][0].cpu().numpy()

                # # 4. CREATE GROUND TRUTH POINT CLOUD
                # pcd_gt = create_point_cloud(img_np, depth_gt_np, intrinsics_np)
                # o3d.io.write_point_cloud(f"/unidepth/points/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_pcd_gt.ply", pcd_gt)

                # # 5. CREATE ESTIMATED POINT CLOUD
                # pcd_pred = create_point_cloud(img_np, depth_pred_np, intrinsics_np)
                # o3d.io.write_point_cloud(f"/unidepth/points/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_pcd_pred.ply", pcd_pred)
                
                # # Clean up local refs to avoid memory leaks
                # del img_np, depth_gt_np, depth_pred_np, intrinsics_np, pcd_gt, pcd_pred

                # # 4. CREATE GROUND TRUTH POINT CLOUD
                # gt_filename = f"/unidepth/points/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_pcd_gt.ply"
                # create_and_save_point_cloud(img_np, depth_gt_np, intrinsics_np, gt_filename)

                # # 5. CREATE ESTIMATED POINT CLOUD
                # pred_filename = f"/unidepth/points/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_pcd_pred.ply"
                # create_and_save_point_cloud(img_np, depth_pred_np, intrinsics_np, pred_filename)
                
                # # Clean up local refs to avoid memory leaks
                # del img_np, depth_gt_np, depth_pred_np, intrinsics_np

                # --- NEW: Extracting Arrays for Point Clouds ---
                
                # 1. Image (H, W, 3 format for our PLY writer)
                img_np = batch["data"]["image_original"][0].cpu().numpy()
                img_np = np.transpose(img_np, (1, 2, 0)) # from (3, H, W) to (H, W, 3)

                # 2. GT and Predicted Depth (H, W)
                depth_gt_np = batch["data"]["depth"][0].squeeze(0).cpu().numpy()
                depth_pred_np = preds["depth"][0].squeeze(0).detach().cpu().numpy()

                # 3. Fetch Camera Intrinsics directly from the camera object
                # batch["data"]["camera"].K is shape (1, 3, 3), so we take [0] to get (3, 3)
                # intrinsics_np = batch["data"]["camera"].K[0].cpu().numpy()
                intrinsics_np = batch["data"]["camera_original"].K[0].cpu().numpy()

                # 4. CREATE GROUND TRUTH POINT CLOUD
                gt_filename = f"/unidepth/points/{rename_datasets_dict[name_dataset]}[{i}]_gt.ply"
                if os.path.exists(gt_filename) == False:
                    create_and_save_point_cloud(img_np, depth_gt_np, intrinsics_np, gt_filename)

                # 5. CREATE ESTIMATED POINT CLOUD
                pred_filename = f"/unidepth/points/{rename_datasets_dict[name_dataset]}[{i}]_{experiment_shortname_dict[experiment]}_pred.ply"
                create_and_save_point_cloud(img_np, depth_pred_np, intrinsics_np, pred_filename)
                
                # Clean up local refs to avoid memory leaks
                del img_np, depth_gt_np, depth_pred_np, intrinsics_np
                # -----------------------------------------------
                
                # -----------------------------------------------

                del preds, losses, metrics, depth_error
                gc.collect()
                torch.cuda.empty_cache()

            # Print metrics ped dataset
            print(f"Dataset: {name_dataset}, min_depth: {min_depth}, max_depth: {max_depth}, max_error: {max_error}")


        # Remove model and collect free memory
        del model 
        gc.collect()
        torch.cuda.empty_cache()

    print("DONE!")

            
