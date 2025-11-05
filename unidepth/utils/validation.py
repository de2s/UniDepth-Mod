import torch
import torch.utils.data.distributed
import wandb
from torch.nn import functional as F

from unidepth.utils import barrier, is_main_process
from unidepth.utils.misc import remove_padding
from unidepth.utils import getAimRun


def original_image(batch, preds=None):
    paddings = [
        torch.tensor(pads)
        for img_meta in batch["img_metas"]
        for pads in img_meta.get("paddings", [[0] * 4])
    ]
    paddings = torch.stack(paddings).to(batch["data"]["image"].device)[
        ..., [0, 2, 1, 3]
    ]  # lrtb

    T, _, H, W = batch["data"]["depth"].shape
    resized_images = []
    for i in range(T):
        p = paddings[i]
        resized_images.append(
            F.interpolate(
                batch["data"]["image"][i].unsqueeze(0),
                (H + p[2] + p[3], W + p[1] + p[2]), # TODO: IS THIS AN ERROR?
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        )
    batch["data"]["image"] = torch.cat(resized_images, dim=0)

    batch["data"]["image"] = remove_padding(
        batch["data"]["image"], paddings.repeat(T, 1)
    )

    if preds is not None:
        for key in ["depth"]:
            if key in preds:
                T, _, H, W = preds[key].shape
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


def log_metrics(metrics_all, step):

    # Get logger for Aim
    run = getAimRun()

    for name_ds, metrics in metrics_all.items():
        for metrics_name, metrics_value in metrics.items():
            try:
                print(f"Metrics/{name_ds}/{metrics_name} {round(metrics_value, 4)}")
                wandb.log(
                    {f"Metrics/{name_ds}/{metrics_name}": metrics_value}, step=step
                )
                # Log with TensorBoard and Aim
                run.track(metrics_value, name=f"metrics/{name_ds}/{metrics_name}", context={"subset": "val"}, step=step)
            except:
                pass


def validate(model, test_loaders, context, log=False, step=None):
    metrics_all = {}
    for name_ds, test_loader in test_loaders.items():
        for i, batch in enumerate(test_loader):
            with context:
                batch["data"] = {
                    k: v.to(model.device) for k, v in batch["data"].items()
                }
                # remove temporal dimension of the dataloder, here is always 1!
                batch["data"] = {
                    #k: v.squeeze(1) for k, v in batch["data"].items() # CHANGED THIS TO PREVENT EXCEPTION WHEN NUM_FRAMES IS NOT 1
                    k: torch.flatten(v, 0, 1) for k, v in batch["data"].items()
                }
                # By enable other functions to process a batch of multiple iamges, this is not needed anymore because it creates a dict with only the entries of the first element.
                # batch["img_metas"] = [
                #    #{k: v[0] for k, v in meta.items() if isinstance(v, list)} # THE PROBLEM IS HERE BECAUSE IT REMOVES ONE PARENTHESIS...
                #    {k: v for k, v in meta.items() if isinstance(v, list)} # CHANGED THIS TO PREVENT EXCEPTION WHEN NUM_FRAMES IS NOT 1 ## TODO: IT'S NOT COPYING THE DISTANCE BETWEEN FRAMES BECAUSE IT IS NOT A LIST
                #    for meta in batch["img_metas"]
                # ]

                preds, losses, metrics = model(batch["data"], batch["img_metas"], log, step)

            batch, _ = original_image(batch, preds=None)
            test_loader.dataset.accumulate_metrics(
                inputs=batch["data"],
                preds=preds,
                keyframe_idx=batch["img_metas"][0].get("keyframe_idx"),
            )

        barrier()
        metrics_all[name_ds] = test_loader.dataset.get_evaluation()

    barrier()
    if is_main_process():
        log_metrics(metrics_all=metrics_all, step=step)
    return metrics_all
