import os
import torch

# Verify that TORCH_HOME variable is set. This path must be consistent with values in `train.sh`
if not os.environ["DATA"]:
    raise Exception("Environment variable \"DATA\" not set")

# Download pretrained UniDepth weights
UNIDEPTH_URL = "https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/resolve/main/pytorch_model.bin"
state_dict = torch.hub.load_state_dict_from_url(UNIDEPTH_URL, map_location="cpu")

# Extract pixel encoder weights
encoder_state_dict = {key.replace("pixel_encoder.", ""): val for key, val in state_dict.items() if key.startswith("pixel_encoder.")}

# Save pixel encoder weights
DINOV2_FILENAME = "dinov2_vitl14_pretrain_unidepth.pth"
MODELS_PATH = os.path.join(os.environ["DATA"], "models", "pretrained")
os.makedirs(MODELS_PATH, exist_ok=True)
torch.save(encoder_state_dict, os.path.join(MODELS_PATH, DINOV2_FILENAME))

# Print confirmation
print(f"Pretrained weights saved to: {os.path.join(MODELS_PATH, DINOV2_FILENAME)}")