import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import split_even_odd, FNS, masked_mean


from unidepth.layers.mlp import MLP
from unidepth.utils.geometric import generate_rays, invert_matrix
from unidepth.utils.log_samples import save_tensors

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


# TODO: HOW DO PREVENT COLAPSE TO A TRIVIAL SOLUTION

class EqSelfDistill(nn.Module):

    def __init__(self, weight: float, output_fn: str = "sqrt", eps: float = 1e-5, use_gt: bool = True):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.eps: float = eps
        self.use_gt = use_gt
        
        
    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        image: torch.Tensor, # Used only for debugging # torch.Size([8, 3, 378, 644])
        feature: torch.Tensor,                         # torch.Size([8, 512, 27, 46])
        depth: torch.Tensor,                           # torch.Size([8, 1, 378, 644])
        intrinsics: torch.Tensor,                      # torch.Size([8, 3, 3])
        estimated_depth: torch.Tensor,                 # torch.Size([8, 1, 378, 644])
        estimated_intrinsics: torch.Tensor,            # torch.Size([8, 3, 3])
        extrinsics: torch.Tensor,                      # torch.Size([8, 4, 4])
        mask: torch.Tensor,                            # torch.Size([8, 1, 378, 644])
        flips: torch.Tensor,                           # torch.Size([8, 1])
        patch_size=14,
    ) -> torch.Tensor:
        
        # Choose between ground truth or estimated data
        depth = depth if self.use_gt else estimated_depth
        K = intrinsics if self.use_gt else estimated_intrinsics
        
        # Flip back flipped tensors
        *_, H, W = depth.shape
        flip_indices = torch.nonzero(flips.view(-1)).squeeze(1)
        image[flip_indices] = image[flip_indices].flip(dims=(3,))
        feature[flip_indices] = feature[flip_indices].flip(dims=(3,))
        depth[flip_indices] = depth[flip_indices].flip(dims=(3,))
        mask[flip_indices] = mask[flip_indices].flip(dims=(3,))
        K[flip_indices, 0, 2] = W - 1 - K[flip_indices, 0, 2]
  
        # Generate rays  
        *_, H, W = feature.shape    
        K[:, 0, 0] /= patch_size  # fx
        K[:, 1, 1] /= patch_size  # fy
        K[:, 0, 2] /= patch_size  # cx
        K[:, 1, 2] /= patch_size  # cy

        rays, _ = generate_rays(K, (H, W)) # torch.Size([8, 1242, 3])

        # Calculate number of patches
        B, _, H, W = depth.shape
        device = depth.device
        assert W % patch_size == 0 and H % patch_size == 0
        num_patches_x, num_patches_y = W // patch_size, H // patch_size

        # Create grid with the pixel coordinates of the patch centers
        centers_x = (torch.arange(num_patches_x, dtype=torch.float32, device=device) + 0.5) * patch_size                   # torch.Size([46])
        centers_y = (torch.arange(num_patches_y, dtype=torch.float32, device=device) + 0.5) * patch_size                   # torch.Size([27])
        centers = torch.stack(torch.meshgrid(centers_y, centers_x, indexing="ij"), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # torch.Size([8, 27, 46, 2])

        # Create grid with shape (1, num_patches_y, num_patches_x, 2) with values normalized to [-1, 1]
        norm_centers_x = 2.0 * centers_x / (W - 1) - 1.0
        norm_centers_y = 2.0 * centers_y / (H - 1) - 1.0
        grid = torch.stack(torch.meshgrid(norm_centers_x, norm_centers_y, indexing="xy"), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # torch.Size([8, 27, 46, 2])

        # Resample depth and mask
        original_depth = depth
        original_mask = mask
        depth = F.grid_sample(depth, grid, mode="nearest", align_corners=True)              # torch.Size([8, 1, 27, 46])
        mask = F.grid_sample(mask.float(), grid, mode="nearest", align_corners=True).bool() # torch.Size([8, 1, 27, 46])
        
        # Flatten tensors                                           
        depth = rearrange(depth, "b c h w -> b (h w) c")                                     # torch.Size([8, 1242, 1])
        mask = rearrange(mask, "b c h w -> b (h w) c")                                       # torch.Size([8, 1242, 1])
        feature = rearrange(feature, "b c h w -> b (h w) c")                                # torch.Size([8, 1242, 512])
        centers = rearrange(centers, "b h w c -> b (h w) c")                                # torch.Size([8, 1242, 2])

        # Retroproject points
        points = torch.stack([
            rays[..., 0] * depth[..., 0] / rays[..., 2],
            rays[..., 1] * depth[..., 0] / rays[..., 2],
            depth[..., 0]
        ], dim=-1) # torch.Size([8, 1242, 3])
        
        # Transform to global coordinates
        B, N, _ = points.shape
        ones = torch.ones(B, N, 1, device=points.device)
        points = torch.cat([points, ones], dim=-1)
        points = torch.bmm(extrinsics, points.transpose(-2, -1))
        points = points.permute(0, 2, 1)[:, :, :3]   # Drop the homogeneous coordinate

        # Iterate samples by pairs
        #chunks = feature.shape[0] // 2
        #iters = zip(
        #    image.chunk(chunks),
        #    points.chunk(chunks),
        #    feature.chunk(chunks),
        #    centers.chunk(chunks),
        #    mask.chunk(chunks),
        #)
        #for i, (pair_image, pair_points, pair_features, pair_centers, pair_mask) in enumerate(iters):
        for i in range(B // 2):
            image0, image1 = image[2 * i], image[2 * i + 1]
            points0, points1 = points[2 * i], points[2 * i + 1]
            feature0, feature1 = feature[2 * i], feature[2 * i + 1]
            centers0, centers1 = centers[2 * i], centers[2 * i + 1]
            mask0, mask1 = mask[2 * i], mask[2 * i + 1]
            original_mask0, original_mask1 = original_mask[2 * i], original_mask[2 * i + 1]

            #image0, image1 = pair_image         # torch.Size([3, 378, 644])
            #points0, points1 = pair_points      # torch.Size([1242, 3])
            #feature0, feature1 = pair_features  # torch.Size([1242, 512])
            #centers0, centers1 = pair_centers   # torch.Size([1242, 2])
            #mask0, mask1 = pair_mask            # torch.Size([1242, 1])

            if self.use_gt:
                mask0, mask1 = mask0.squeeze(), mask1.squeeze()
                points0 = points0[mask0]
                points1 = points1[mask1]
                feature0 = feature0[mask0]
                feature1 = feature1[mask1]
                centers0 = centers0[mask0]
                centers1 = centers1[mask1]
                
            # Calculate matrix of distances between 3D points
            distances = torch.cdist(points0, points1, p=2)

            # Compute matching plan 
            reg = 0.05       # entropy regularization
            reg_m = 1.0      # mass relaxation for unbalanced OT
            A = torch.ones(points0.shape[0], device=points0.device)
            B = torch.ones(points1.shape[0], device=points1.device)
            A /= A.sum()
            B /= B.sum()            
            # plan = ot.sinkhorn_unbalanced(a, b, C, reg, reg_m, method='sinkhorn', reg_type='kl', numItermax=1000, stopThr=1e-6, verbose=False)!
            plan = ot.unbalanced.sinkhorn_unbalanced(A, B, distances, reg, reg_m)
            # TODO: CHECK DISTANCE SQUARED

            # Calculate costs using unbalanced optimal transport
            matches_a = torch.argmax(plan, dim=1)  # For each A[i], best B[j]
            matches_b = torch.argmax(plan, dim=0)  # For each B[i], best A[j]

            # Select only mutual matches

            # Optional: get the max values (mass) to filter weak matches
            #max_vals, _ = torch.max(distances, dim=1)
            #threshold = 1e-3  # tune this threshold based on your application

            # Filtered matches, set unmatched points to -1
            #matches_filtered = torch.where(max_vals > threshold, matches, torch.tensor(-1, device=plan.device))

            # PLOT IMAGES
            PLOT_IMAGES = False # Set a breakpoint and change value in the debugger to make sure this method is called on demand
            if(PLOT_IMAGES):
                save_tensors(image0, image1, original_mask0, original_mask1, feature0, feature1, centers0, centers1, points0, points1, distances, plan, matches_a, matches_b)
                #plot_matched_points(image0, image1, )
                ...

        raise NotImplementedError("Method not implemented")
        

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
