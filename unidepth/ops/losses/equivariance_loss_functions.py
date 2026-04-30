import torch


LOSS_FNS = {
    "cos_sim": lambda features0, features1: cos_sim(features0, features1),
    "l2_dist": lambda features0, features1: l2_dist(features0, features1),
    "l1_dist": lambda features0, features1: l1_dist(features0, features1),
}


def cos_sim(features0, features1, eps: float = 1e-5):

    # Compute cosine similarity for each pair
    dots = torch.sum(features0 * features1, dim=1) 
    norms0 = torch.norm(features0, dim=1)
    norms1 = torch.norm(features1, dim=1)
    cosine_similarities = dots / (norms0 * norms1 + eps) 

    # Return cosine distances
    return 1 - torch.abs(cosine_similarities)


def l2_dist(features0, features1):

    # Return L2 distance between vectors
    distances = torch.linalg.norm(features0 - features1, dim=1)
    return distances

def l1_dist(features0, features1):

    # Return L2 distance between vectors
    distances = torch.linalg.norm(features0 - features1, ord=1, dim=1)
    return distances