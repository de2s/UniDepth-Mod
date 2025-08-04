import matplotlib.pyplot as plt
import torch

def save_tensors(image0, image1, mask0, mask1, feature0, feature1, centers0, centers1, points0, points1, distances, plan, matches_a, matches_b):

    torch.save(image0, "image0.pth")
    torch.save(image1, "image1.pth")
    torch.save(mask0, "mask0.pth")
    torch.save(mask1, "mask1.pth")
    torch.save(feature0, "feature0.pth")
    torch.save(feature1, "feature1.pth")
    torch.save(centers0, "centers0.pth")
    torch.save(centers1, "centers1.pth")
    torch.save(points0, "points0.pth")
    torch.save(points1, "points1.pth")
    torch.save(distances, "distances.pth")
    torch.save(plan, "plan.pth")
    torch.save(matches_a, "matches_a.pth")
    torch.save(matches_b, "matches_b.pth")
    
def plot_images(image_a: torch.Tensor, image_b: torch.Tensor):

    plt.figure(figsize=(8, 8))
    return