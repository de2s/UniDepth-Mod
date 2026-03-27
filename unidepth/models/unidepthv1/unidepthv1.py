"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import importlib
import json
from copy import deepcopy
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from unidepth.models.unidepthv1.decoder import Decoder
from unidepth.utils.constants import (IMAGENET_DATASET_MEAN,
                                      IMAGENET_DATASET_STD)
from unidepth.utils.distributed import is_main_process
from unidepth.utils.geometric import (generate_rays,
                                      spherical_zbuffer_to_euclidean)
from unidepth.utils.misc import get_params, match_gt
from unidepth.utils import getAimRun
from unidepth.utils.visualization import rescale, colorize
from unidepth.ops.losses import match_features

from aim import Image, Text, Distribution

MAP_BACKBONES = {"ViTL14": "vitl14", "ConvNextL": "cnvnxtl"}


# inference helpers
def _paddings(image_shape, network_shape):
    cur_h, cur_w = image_shape
    h, w = network_shape
    pad_top, pad_bottom = (h - cur_h) // 2, h - cur_h - (h - cur_h) // 2
    pad_left, pad_right = (w - cur_w) // 2, w - cur_w - (w - cur_w) // 2
    return pad_left, pad_right, pad_top, pad_bottom


def _shapes(image_shape, network_shape):
    h, w = image_shape
    input_ratio = w / h
    output_ratio = network_shape[1] / network_shape[0]
    if output_ratio > input_ratio:
        ratio = network_shape[0] / h
    elif output_ratio <= input_ratio:
        ratio = network_shape[1] / w
    return (ceil(h * ratio - 0.5), ceil(w * ratio - 0.5)), ratio


def _preprocess(rgbs, intrinsics, shapes, pads, ratio, output_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    rgbs = F.interpolate(
        rgbs, size=shapes, mode="bilinear", align_corners=False, antialias=True
    )
    rgbs = F.pad(rgbs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant")
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio + pad_left
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio + pad_top
        return rgbs, intrinsics
    return rgbs, None


def _postprocess(predictions, intrinsics, shapes, pads, ratio, original_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    # pred mean, trim paddings, and upsample to input dim
    predictions = sum(
        [
            F.interpolate(
                x.clone(),
                size=shapes,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            for x in predictions
        ]
    ) / len(predictions)
    predictions = predictions[
        ..., pad_top : shapes[0] - pad_bottom, pad_left : shapes[1] - pad_right
    ]
    predictions = F.interpolate(
        predictions,
        size=original_shapes,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    intrinsics[:, 0, 0] = intrinsics[:, 0, 0] / ratio
    intrinsics[:, 1, 1] = intrinsics[:, 1, 1] / ratio
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - pad_left) / ratio
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - pad_top) / ratio
    return predictions, intrinsics


class UniDepthV1(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="UniDepth",
    repo_url="https://github.com/lpiccinelli-eth/UniDepth",
    tags=["monocular-metric-depth-estimation"],
):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.build(config)
        self.build_losses(config)
        self.eps = eps

    def forward(self, inputs, image_metas, log=False, step=None):
        rgbs = inputs["image"]

        # Check if it should log artifacts
        if log: assert step is not None
        # Log input artifacts with Aim
        if log:

            # Log with Aim
            run = getAimRun()
            
            # Log text with image metadata
            selected = ["dataset_name", "filename", "si", "dense", "synthetic", "valid_camera", "valid_pose", "flip"]
            metas = [{prop: sequence[prop][i] for prop in selected} for sequence in image_metas for i in range(len(next(iter(sequence.values()))))]
            run.track([Text(json.dumps(meta)) for meta in metas], name="artifacts/train/image_metas", context={"subset": "train"}, step=step)
            
            # Log original input images
            datasets = [dataset for meta in image_metas for dataset in meta["dataset_name"]]
            filenames = [filename for meta in image_metas for filename in meta["filename"]]
            images = [image for image in inputs["image_original"]]
            run.track([Image(image, caption=f"{dataset}: {filename}") for image, dataset, filename in zip(images, datasets, filenames)], name="artifacts/train/original_image", context={"subset": "train"}, step=step)
            
            # Log preprocessed images
            preprocessed_images = [rescale(image) for image in inputs["image"]]
            run.track([Image(preprocessed_image, caption=f"{dataset}: {filename}") for preprocessed_image, dataset, filename in zip(preprocessed_images, datasets, filenames)], name="artifacts/train/preprocessed_image", context={"subset": "train"}, step=step)
            
            # Log depths
            depths = [colorize(depth.squeeze().detach().cpu().numpy()) for depth in inputs["depth"]]
            run.track([Image(depth, caption=f"{dataset}: {filename}") for depth, dataset, filename in zip(depths, datasets, filenames)], name="artifacts/train/depth", context={"subset": "train"}, step=step)
            for i in range(inputs["depth"].shape[0]):
                run.track(Distribution(inputs["depth"][i].detach().cpu().numpy()), name=f"artifacts/train/depth_distribution", context={"subset": "train", "index": i}, step=step)
                run.track(Distribution(inputs["depth"][i].detach().cpu().numpy()), name=f"artifacts/train/depth_distribution_by_dataset", context={"subset": "train", "dataset": datasets[i]}, step=step)
            
            # Log depth mask
            masks = [mask.float() for mask in inputs["depth_mask"]]
            run.track([Image(mask, caption=f"{dataset}: {filename}") for mask, dataset, filename in zip(masks, datasets, filenames)], name="artifacts/train/depth_mask", context={"subset": "train"}, step=step)
            
            # Log distance between frames # TODO: IT DOES NOT LOG ARRAYS
            run.track(image_metas[0]["frames_offset"], name="train/frames_offset")

        B, _, H, W = rgbs.shape
        cameras = inputs["camera"]

        # Encode
        encoder_outputs, cls_tokens = self.pixel_encoder(rgbs)
        if "dino" in self.pixel_encoder.__class__.__name__.lower():
            encoder_outputs = [
                (x + y.unsqueeze(1)).contiguous()
                for x, y in zip(encoder_outputs, cls_tokens)
            ]
        inputs["encoder_outputs"] = encoder_outputs
        inputs["cls_tokens"] = cls_tokens

        # Get camera rays for supervision, all in unit sphere
        inputs["rays"] = rearrange(
            cameras.get_rays(shapes=(B, H, W)), "b c h w -> b (h w) c"
        )

        # Decode
        pred_intrinsics, predictions, depth_features = self.pixel_decoder(inputs, {})
        predictions = sum(
            [
                F.interpolate(
                    x.clone(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                for x in predictions
            ]
        ) / len(predictions)

        # Final 3D points backprojection
        pred_rays, pred_angles = generate_rays(pred_intrinsics, (H, W), noisy=False)

        # You may want to use inputs["angles"] if available?
        pred_angles = rearrange(pred_angles, "b (h w) c -> b c h w", h=H, w=W)

        # reshape to match GT as paddings and shape
        if not self.training:
            depth_gt = inputs["depth"]
            image_paddings = image_metas[0]["paddings"]
            depth_paddings = image_metas[0]["depth_paddings"]
            predictions = match_gt(
                predictions, depth_gt, padding1=image_paddings, padding2=depth_paddings
            )
            pred_angles = match_gt(
                pred_angles, depth_gt, padding1=image_paddings, padding2=depth_paddings
            )

        points_3d = torch.cat((pred_angles, predictions), dim=1)
        points_3d = spherical_zbuffer_to_euclidean(
            points_3d.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        # Output data, use for loss computation
        outputs = {
            "angles": pred_angles,
            "rays": pred_rays,
            "intrinsics": pred_intrinsics,
            "points": points_3d,
            "depth": predictions[:, -1:],
            "cond_features": depth_features,
            "encoder_outputs": inputs["encoder_outputs"],
        }
        self.pixel_decoder.test_fixed_camera = False

        # Log output artifacts with Aim
        if log:
            
            # Log with Aim
            run = getAimRun()

            # Log estimated depths
            datasets = [dataset for meta in image_metas for dataset in meta["dataset_name"]]
            filenames = [filename for meta in image_metas for filename in meta["filename"]]
            estimated_depths = [colorize(estimated_depth.squeeze().detach().cpu().numpy()) for estimated_depth in outputs["depth"]]
            run.track([Image(estimated_depth, caption=f"{dataset}: {filename}") for estimated_depth, dataset, filename in zip(estimated_depths, datasets, filenames)], name="artifacts/train/estimated_depth", context={"subset": "train"}, step=step)
            for i in range(inputs["depth"].shape[0]):
                run.track(Distribution(outputs["depth"][i].detach().cpu().numpy()), name=f"artifacts/train/estimated_depth_distribution", context={"subset": "train", "index": i}, step=step)
            
            # Log depth error
            estimated_depth_errors = [colorize((estimated_depth - depth).squeeze().detach().cpu().numpy()) for estimated_depth, depth in zip(outputs["depth"], inputs["depth"])]
            run.track([Image(estimated_depth_error, caption=f"{dataset}: {filename}") for estimated_depth_error, dataset, filename in zip(estimated_depth_errors, datasets, filenames)], name="artifacts/train/estimated_depth_error", context={"subset": "train"}, step=step)
            for i in range(inputs["depth"].shape[0]):
                run.track(Distribution((outputs["depth"][i] - inputs["depth"][i]).detach().cpu().numpy()), name=f"artifacts/train/estimated_depth_error_distribution", context={"subset": "train", "index": i}, step=step)

        losses, metrics = self.compute_losses(outputs, inputs, image_metas, log, step)
    
        return outputs, losses, metrics

    def compute_losses(self, outputs, inputs, image_metas, log=False, step=None):
        losses = {"opt": {}, "stat": {}}
        metrics = {}
        if (
            not self.training
        ):  # only compute losses during training, avoid issues for mismatch size of pred and GT
            return losses, metrics
        losses_to_be_computed = list(self.losses.keys())

        # depth loss
        si = torch.tensor([meta.get("si", False) for meta in image_metas], device=self.device).reshape(-1, 1)
        loss = self.losses["depth"]
        depth_losses = loss(
            outputs["depth"],
            target=inputs["depth"],
            mask=inputs["depth_mask"].clone(),
            si=si,
        )
        losses["opt"][loss.name] = loss.weight * depth_losses.mean()
        losses_to_be_computed.remove("depth")

        # camera loss, here we apply to rays for simplicity
        # in the original training was on angles
        # however, we saw no difference (see supplementary)
        loss = self.losses["camera"]
        camera_losses = loss(outputs["rays"], target=inputs["rays"])
        losses["opt"][loss.name] = loss.weight * camera_losses.mean()
        losses_to_be_computed.remove("camera")

        # Invariance loss
        if "invariance" in losses_to_be_computed:
            flips = torch.tensor([meta.get("flip", False) for meta in image_metas], device=self.device).reshape(-1, 1)
            idxs = [
                m * meta["num_copies"] * meta["num_frames"] +
                c * meta["num_frames"] +
                f
                for m, meta in enumerate(image_metas)
                for f in range(meta["num_frames"])
                for c in range(meta["num_copies"])
            ]
            loss = self.losses["invariance"]
            invariance_losses = loss(
                outputs["cond_features"][idxs],
                intrinsics=inputs["camera"].K[idxs],
                mask=inputs["depth_mask"][idxs],
                flips=flips[idxs],
            )
            losses["opt"][loss.name] = loss.weight * invariance_losses.mean()
            losses_to_be_computed.remove("invariance")

        # Match features for equivariant related losses
        if "equivariance" in losses_to_be_computed:
            flips = torch.tensor([meta.get("flip", False) for meta in image_metas], device=self.device).reshape(-1, 1)
            idxs = [
                m * meta["num_copies"] * meta["num_frames"]
                + c * meta["num_frames"]
                + f
                for m, meta in enumerate(image_metas)
                for c in range(meta["num_copies"])
                for f in range(meta["num_frames"])
            ]
            (
                list_features0, list_features1, 
                list_points3d0, list_points3d1, 
                list_poses0, list_poses1, 
                list_flips0, list_flips1, 
                matcher_metrics
            ) = match_features(
                image = inputs["image_original"][idxs],
                feature = outputs["cond_features"][idxs],
                depth = inputs["depth"][idxs],
                intrinsics = inputs["camera"].K[idxs],
                estimated_depth = outputs["depth"][idxs],
                estimated_intrinsics = outputs["intrinsics"][idxs],
                poses = inputs["cam2w"][idxs],
                mask = inputs["depth_mask"][idxs],
                flips = flips[idxs],
                use_gt = True, # TODO: PARAMETRIZE THIS
                log=log,
                step=step
            )
        # Equivariance loss
        if "equivariance" in losses_to_be_computed:
            loss = self.losses["equivariance"]
            equivariance_losses= loss(
                list_features0, list_features1, 
                list_points3d0, list_points3d1,
                list_poses0, list_poses1, 
                list_flips0, list_flips1,
                log=log,
                step=step
            )
            losses["opt"][loss.name] = loss.weight * equivariance_losses.mean()
            losses_to_be_computed.remove("equivariance")
            metrics.update(matcher_metrics)

        # remaining losses, we expect no more losses to be computed
        assert (
            not losses_to_be_computed
        ), f"Losses {losses_to_be_computed} not computed, revise `compute_loss` method"

        return losses, metrics

    @torch.no_grad()
    def infer(self, rgbs: torch.Tensor, intrinsics=None, skip_camera=False):
        if rgbs.ndim == 3:
            rgbs = rgbs.unsqueeze(0)
        if intrinsics is not None and intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        B, _, H, W = rgbs.shape

        rgbs = rgbs.to(self.device)
        if intrinsics is not None:
            intrinsics = intrinsics.to(self.device)

        # process image and intrinsiscs (if any) to match network input (slow?)
        if rgbs.max() > 5 or rgbs.dtype == torch.uint8:
            rgbs = rgbs.to(torch.float32).div(255)
        if rgbs.min() >= 0.0 and rgbs.max() <= 1.0:
            rgbs = TF.normalize(
                rgbs,
                mean=IMAGENET_DATASET_MEAN,
                std=IMAGENET_DATASET_STD,
            )

        (h, w), ratio = _shapes((H, W), self.image_shape)
        pad_left, pad_right, pad_top, pad_bottom = _paddings((h, w), self.image_shape)
        rgbs, gt_intrinsics = _preprocess(
            rgbs,
            intrinsics,
            (h, w),
            (pad_left, pad_right, pad_top, pad_bottom),
            ratio,
            self.image_shape,
        )

        # run encoder
        encoder_outputs, cls_tokens = self.pixel_encoder(rgbs)
        if "dino" in self.pixel_encoder.__class__.__name__.lower():
            encoder_outputs = [
                (x + y.unsqueeze(1)).contiguous()
                for x, y in zip(encoder_outputs, cls_tokens)
            ]

        # get data for decoder and adapt to given camera
        inputs = {}
        inputs["encoder_outputs"] = encoder_outputs
        inputs["cls_tokens"] = cls_tokens
        inputs["image"] = rgbs
        if gt_intrinsics is not None:
            rays, angles = generate_rays(
                gt_intrinsics, self.image_shape, noisy=self.training
            )
            inputs["rays"] = rays
            inputs["angles"] = angles
            inputs["K"] = gt_intrinsics
            self.pixel_decoder.test_fixed_camera = True
            self.pixel_decoder.skip_camera = skip_camera

        # decode all
        pred_intrinsics, predictions, _ = self.pixel_decoder(inputs, {})

        # undo the reshaping and get original image size (slow)
        predictions, pred_intrinsics = _postprocess(
            predictions,
            pred_intrinsics,
            self.image_shape,
            (pad_left, pad_right, pad_top, pad_bottom),
            ratio,
            (H, W),
        )

        # final 3D points backprojection
        intrinsics = gt_intrinsics if gt_intrinsics is not None else pred_intrinsics
        angles = generate_rays(intrinsics, (H, W), noisy=False)[-1]
        angles = rearrange(angles, "b (h w) c -> b c h w", h=H, w=W)
        points_3d = torch.cat((angles, predictions), dim=1)
        points_3d = spherical_zbuffer_to_euclidean(
            points_3d.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        # output data
        outputs = {
            "intrinsics": pred_intrinsics,
            "points": points_3d,
            "depth": predictions[:, -1:],
        }
        self.pixel_decoder.test_fixed_camera = False
        self.pixel_decoder.skip_camera = False
        return outputs

    def load_pretrained(self, model_file):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device)

        if "model" in dict_model:
            dict_model = dict_model["model"]
        new_state_dict = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )

        info = self.load_state_dict(new_state_dict, strict=False)
        if is_main_process():
            print(
                f"Loaded from {model_file} for {self.__class__.__name__} results in:",
                info,
            )

    def get_params(self, config):
        if hasattr(self.pixel_encoder, "get_params"):
            encoder_p, encoder_lr = self.pixel_encoder.get_params(
                config["model"]["pixel_encoder"]["lr"],
                config["training"]["wd"],
                config["training"]["ld"],
            )
        else:
            encoder_p, encoder_lr = get_params(
                self.pixel_encoder,
                config["model"]["pixel_encoder"]["lr"],
                config["training"]["wd"],
            )
        decoder_p, decoder_lr = get_params(
            self.pixel_decoder, config["training"]["lr"], config["training"]["wd"]
        )
        losses_p, losses_lr = get_params(
            self.losses, config["training"]["lr"], config["training"]["wd"]
        )
        return [*encoder_p, *decoder_p, *losses_p]

    @property
    def device(self):
        return next(self.parameters()).device

    def build(self, config):
        mod = importlib.import_module("unidepth.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder_config = {
            **config["training"],
            **config["data"],
            **config["model"]["pixel_encoder"],
            "interpolate_offset": 0.1,
        }
        pixel_encoder = pixel_encoder_factory(pixel_encoder_config)

        config["model"]["pixel_encoder"]["patch_size"] = (
            14 if "dino" in config["model"]["pixel_encoder"]["name"] else 16
        )
        pixel_encoder_embed_dims = (
            pixel_encoder.embed_dims
            if hasattr(pixel_encoder, "embed_dims")
            else [getattr(pixel_encoder, "embed_dim") * 2**i for i in range(4)]
        )
        config["model"]["pixel_encoder"]["embed_dim"] = getattr(
            pixel_encoder, "embed_dim"
        )
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims
        config["model"]["pixel_encoder"]["depths"] = pixel_encoder.depths

        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = Decoder(config)
        self.image_shape = config["data"]["image_shape"]

    def build_losses(self, config):
        losses = {}
        # for loss_name, loss_config in config["training"]["losses"].items():
        for loss_name, loss_config in config["training"].get("losses", {}).items():
            mod = importlib.import_module("unidepth.ops.losses")
            loss_factory = getattr(mod, loss_config["name"])
            losses[loss_name] = loss_factory.build(loss_config)
        self.losses = nn.ModuleDict(losses)
