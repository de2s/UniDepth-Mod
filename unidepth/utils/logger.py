import io
import os

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from aim import Run, Image

# Use non-interactive backend
matplotlib.use('Agg') 

# Logger singleton instance
_aim_run = None

def getAimRun(run_hash=None):
    global _aim_run
    if _aim_run is None:
        _aim_run = Run(experiment="unidepth", repo="/data/aim", run_hash=run_hash)
        _aim_run.set_artifacts_uri("file:///data/aim/artifacts")
    return _aim_run

def log_equivariant_loss_artifacts(
    image0, image1,
    mask0, mask1, 
    feature0, feature1, 
    points2d0, points2d1,
    matches, 
    step
):
    # Get logger
    run = getAimRun()

    # Get path on temp folder
    temp_folder = os.path.join(os.environ["CACHE"], "temp")
    file_name = "equivariant_loss_tensors.pth"
    path = os.path.join(temp_folder, file_name)

    # Log tensors to a file and save artifact
    if 'equivariant_loss_tensors' not in run.artifacts:
        torch.save(
            {
                "image0": image0, "image1": image1, 
                "mask0": mask0, "mask1": mask1,
                "feature0": feature0, "feature1": feature1, 
                "points2d0": points2d0, "points2d1": points2d1,
                "matches": matches,
            }, 
            path
        )
        run.log_artifact(path, name="artifacts/train/equivariant_loss_tensors")

    ### PLOT ORIGINAL IMAGES ###
    
    # TODO: ORIGINAL IMAGES ARE DRAWN AT PROPPER SCALE

    # Permute channels
    image0 = image0.permute(1, 2, 0)
    image1 = image1.permute(1, 2, 0)

    # Convert dtype to unit8
    if image0.dtype == torch.uint8 and image1.dtype == torch.uint8:
        pass
    elif image0.dtype == torch.float32 and image1.dtype == torch.float32:
        image0 = ((image0 - image0.min()) / (image0.max() - image0.min()) * 255).to(torch.uint8)
        image1 = ((image1 - image1.min()) / (image1.max() - image1.min()) * 255).to(torch.uint8)
    else:
        raise RuntimeError(f"Image type not supported: {image0.dtype}")

    # Combine images
    combined = torch.cat([image0, image1], dim=1).cpu().numpy()
    h, w, _ = combined.shape
    
    # Save figure
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    axe = plt.Axes(fig, [0., 0., 1., 1.])
    axe.set_axis_off()
    fig.add_axes(axe)
    axe.imshow(combined)
    # Draw patch centers
    for patch0_center, patch1_center in zip(points2d0[matches[:, 0]].cpu().numpy(), points2d1[matches[:, 1]].cpu().numpy()):
        axe.plot(patch0_center[1], patch0_center[0], marker="o", color="limegreen", markersize=0.5)
        axe.plot(patch1_center[1] + mask0.shape[1], patch1_center[0], marker="o", color="limegreen", markersize=0.5)
    run.track(Image(fig), name="artifacts/train/matched_images", context={"subset": "train"}, step=step)
    plt.close(fig)

    ### PLOT MASKS ###

    # TODO: MASKS ARE NOT BEING DRAWN AT PROPPER SCALE

    # Combine masks
    combined = torch.cat([mask0, mask1], dim=1).cpu().numpy()
    combined = np.where(combined, 0.5, 1.0)
    h, w = combined.shape

    # Save figure
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    axe = plt.Axes(fig, [0., 0., 1., 1.])
    axe.set_axis_off()
    fig.add_axes(axe)
    axe.imshow(combined, cmap='gray', vmin=0, vmax=1)
    plt.axis("off")
    # Draw patch centers
    for patch0_center, patch1_center in zip(points2d0[matches[:, 0]].cpu().numpy(), points2d1[matches[:, 1]].cpu().numpy()):
        axe.plot(patch0_center[1], patch0_center[0], marker="o", color="limegreen", markersize=0.5)
        axe.plot(patch1_center[1] + mask0.shape[1], patch1_center[0], marker="o", color="limegreen", markersize=0.5)
    run.track(Image(fig), name="artifacts/train/matched_masks", context={"subset": "train"}, step=step)
    plt.close(fig)


def save_point_cloud(tag, points, colors):

    # Ensure colors are int and in 0-255
    colors = colors.clamp(0, 255).to(torch.uint8)

    # Build body efficiently in memory
    buf = io.StringIO()

    # Header
    buf.write("ply\n")
    buf.write("format ascii 1.0\n")
    buf.write(f"element vertex {points.shape[0]}\n")
    buf.write("property float x\n")
    buf.write("property float y\n")
    buf.write("property float z\n")
    buf.write("property uchar red\n")
    buf.write("property uchar green\n")
    buf.write("property uchar blue\n")
    buf.write("end_header\n")

    # Convert to NumPy for efficient vectorized writing
    pts_np = points.cpu().numpy()
    cols_np = colors.cpu().numpy()

    # Join all rows into one large string (very fast)
    data_str = "\n".join(
        f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}"
        for (x, y, z), (r, g, b) in zip(pts_np, cols_np)
    )
    buf.write(data_str)
    buf.write("\n")

    # Get path on temporary folder, create it if needed.
    temp_folder = os.path.join(os.environ["CACHE"], "temp")
    os.makedirs(temp_folder, exist_ok=True)
    filename = f"{tag.split('/')[-1]}.ply"
    path = os.path.join(temp_folder, filename)  

    # Write once to disk
    with open(path, "w") as f:
        f.write(buf.getvalue())

    # Log artifact
    run = getAimRun()
    run.log_artifact(path, name=tag)

    # Delete file
    os.remove(path)