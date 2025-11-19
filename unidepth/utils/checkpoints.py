import os
import random
import numpy as np

import torch

from unidepth.utils.distributed import is_main_process

checkpoint_file = os.path.join(os.environ["CACHE"], "checkpoints/unidepth_checkpoint.pth")

def save_checkpoint(model, optimizer, step, wandb_run_id, wandb_run_name, aim_run_hash): #, , scheduler, step, wandb_id, aim_hash): scheduler_wd, scheduler_lr, scheduler_betas

    # Create folder if it doesn't exist
    checkpoint_folder = os.path.dirname(checkpoint_file)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    # Save checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        #"scheduler_wd": scheduler_wd.state_dict(),
        #"scheduler_lr": scheduler_lr.state_dict(),
        #"scheduler_betas": scheduler_betas.state_dict(),
        "prng_python": random.getstate(),
        "prng_numpy": np.random.get_state(),
        "prng_torch_cpu": torch.get_rng_state(),
        "prng_torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "step": step,
        "wandb_run_id": wandb_run_id,
        "wandb_run_name": wandb_run_name,
        "aim_run_hash": aim_run_hash,

    }
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved to {checkpoint_file}")
    
def load_checkpoint(model, optimizer, step): #, optimizer, scheduler): scheduler_wd, scheduler_lr, scheduler_betas

    # Check if checkpoint exist.
    if not os.path.exists(checkpoint_file):
        return 0, None, None, None

    # Ask the user to confirm checkpoint loading
    if is_main_process():
        while True:
            resp = input(f"Checkpoint found at {checkpoint_file}. Load it? [y/n] ").lower()
            if resp == "y" or resp == "n":
                break
        if resp == "n":
                return 0, None, None, None

    # Load tensors
    checkpoint = torch.load(checkpoint_file)

    # Load Unidepth model
    model.load_state_dict(checkpoint['model_state'])

    # Load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # Load schedulers
    #scheduler_wd.load_state_dict(checkpoint['scheduler_wd'])
    #scheduler_lr.load_state_dict(checkpoint['scheduler_lr'])
    #scheduler_betas.load_state_dict(checkpoint['scheduler_betas'])

    # Load random number generator states
    random.setstate(checkpoint["prng_python"])
    np.random.set_state(checkpoint["prng_numpy"])
    torch.set_rng_state(checkpoint["prng_torch_cpu"])
    if "prng_torch_cuda_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["prng_torch_cuda_all"])

    # Load iteration number
    step = checkpoint.get('step', 0)

    # Load logger run ids
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    wandb_run_name = checkpoint.get('wandb_run_name', None)
    aim_run_hash = checkpoint.get('aim_run_hash', None)

    return step, wandb_run_id, wandb_run_name, aim_run_hash

