
from datetime import datetime

import torch
import matplotlib.cm as cm
from torch.utils.tensorboard import SummaryWriter

# Singleton factory for Logger
_summary_writer_instance = None

def getSummaryWriter():
    global _summary_writer_instance
    if _summary_writer_instance is None:
        log_dir = f"/data/runs/unidepth_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        _summary_writer_instance = SummaryWriter(log_dir)
    return _summary_writer_instance


# TODO: MOVE THIS FUNCTION TO A SUITABLE FILE FOR VISUALIZATION PURPOSES

# Cache of LUTs so we don't recompute
_lut_cache = {}

def get_colormap_lut(map_name, num_colors = 256, device = "cuda"):

    key = (map_name, num_colors, device)
    if key not in _lut_cache:
        cmap = cm.get_cmap(map_name, num_colors)
        lut = torch.tensor(cmap.colors[:, :3], dtype=torch.float32, device=device)
        _lut_cache[key] = lut
    return _lut_cache[key]

def colorize(images, map_name = "magma_r", num_colors = 256):

    # Standardize input to (N,1,H,W)
    if images.dim() == 2:
        images = images.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif images.dim() == 3 and images.shape[0] == 1:
        images = images.unsqueeze(0)  # (1,1,H,W)
    elif images.dim() == 4 and images.shape[1] == 1:
        pass  # already batch
    else:
        raise ValueError("Input must be (H,W), (1,H,W), or (N,1,H,W) single-channel images")

    device = images.device
    lut = get_colormap_lut(map_name, num_colors, device)

    # normalize each image in the batch individually to [0,1]
    #min_vals = images.view(images.size(0), -1).min(dim=1)[0].view(-1,1,1,1)
    #max_vals = images.view(images.size(0), -1).max(dim=1)[0].view(-1,1,1,1)
    #img_norm = (images - min_vals) / (max_vals - min_vals + 1e-8)

    min = images.min()
    max = images.max()
    img_norm = (images - min) / (max - min + 1e-8)

    # Map to LUT indices
    idx = (img_norm * (num_colors - 1)).long()  # shape (N,1,H,W)
    idx = idx.squeeze(1)  # (N,H,W)

    # Apply LUT (broadcast over batch)
    colored = lut[idx]  # (N,H,W,3)
    return colored.permute(0, 3, 1, 2)  # (N,3,H,W)
