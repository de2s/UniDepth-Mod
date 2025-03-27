# TODO: FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel AS build
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel AS build

# Install OS requirements
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install python3-opencv git -y

# Install Python requirements
WORKDIR /workspaces/unidepth
COPY requirements.txt .
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Install Pillow-SIMD (Optional)
RUN pip uninstall pillow -y
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# Install Unidepth
COPY . .
RUN pip install -e .
