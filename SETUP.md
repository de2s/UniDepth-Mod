# Development Environment Setup for Equivariant Loss in UniDepth

This guide describes how to prepare the development environment required to train UniDepth with the equivariant loss. It covers system prerequisites, dependency installation, repository setup, and other necessary configuration steps to ensure the training pipeline runs reproducibly and without environment-related issues.

## Prerequisites:

The development environment has been validated with the following software versions:

- Linux (Ubuntu 24.04.3, kernel 6.14.0-37-generic)
- CUDA Toolkit v12.9
- Docker v29.1.3
- NVIDIA Container Toolkit v1.18.1
- Visual Studio Code
- Git

## Clone the Source Code

Clone the repository onto your Linux machine:

```bash
git clone https://github.com/de2s/UniDepth-Private.git
```

Navigate to the project directory and check out the target branch:

```bash
cd cd UniDepth-Private/
git checkout feat-equivariant-loss-2
```

## Copy the Datasets

Place the dataset files in a directory on your Linux machine and keep a note of the path, as it will be needed later.

Split | Dataset         | Filename
------|-----------------|---------------
Train | Argoverse2      | `AV2_viz.hdf5`
Train | ScanNet         | `ScanNetS.hdf5`
Train | Waymo           | `Waymo_viz.hdf5`
Train | ARKit           | `ARKitS.hdf5`
Train | HM3D            | `HM3D.hdf5`
Train | HOI4D           | `HOI4D.hdf5`
Train | MVImgNet        | `MVImgNet.hdf5`
Train | NianticMapFree  | `NianticMapFree.hdf5`
Train | WildRGBD        | `WildRGBD.hdf5`
Test  | DDAD            | `ddad/ddad_0.hdf5` to `ddad/ddad_7.hdf5`
Test  | DiodeIndoor     | `DiodeIndoor.hdf5`
Test  | ETH3D           | `ETH3D.hdf5`
Test  | HAMMER          | `hammer.hdf5`
Test  | IBims           | `ibims.hdf5`
Test  | KITTI           | `kitti.hdf5`
Test  | Nuscenes        | `nuscenes/nuscenes_0.hdf5` to `nuscenes/nuscenes_7.hdf5`
Test  | NYUv2Depth      | `nyuv2.hdf5`
Test  | SUNRGBD         | `SUNRGB.hdf5`
Test  | VOID            | `void.hdf5`
Test  | Argoverse2      | `AV2_viz.hdf5`
Test  | ScanNet         | `ScanNetS.hdf5`
Test  | Waymo           | `Waymo_viz.hdf5`
Test  | KITTIMulti      | `KITTI_sequence.hdf5`
Test  | BlendedMVG      | `BlendedMVG_.hdf5`
Test  | DTURMVD         | `dtu_rmvd.hdf5`

These files are not publicly available due to redistribution restrictions. However, the original UniDepth repository includes a few sample files, which you can use to examine their structure and create your own HDF5 files.

## Create a Folder for Storing Data

Running the UniDepth training script generates several files and directories, including:

- Model checkpoints
- A cache folder
- Artifacts from the Aim experiment tracking tool
- Artifacts from the W&B experiment tracking tool

Create a directory on your Linux machine with sufficient storage to hold these files, and note the path, as it will be required later.


## Set Up Visual Studio Code

If your Linux machine has a graphical interface, you can open the code repository directly in Visual Studio Code. Otherwise, you can run Visual Studio Code locally and connect to the Linux machine via SSH.

__SSH Remote Connection__

To enable remote access, install the [Remote SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension in Visual Studio Code.

After installing, configure a connection to your Linux machine using either a password or SSH keys. For detailed instructions, refer to the Visual Studio Code documentation on [SSH remote connections](https://code.visualstudio.com/docs/remote/ssh).

Once the remote connection is established, open the folder where the source code repository was cloned.

__DevContainers Setup__

The UniDepth training pipeline with the equivariant loss is configured to run inside a Docker container. This setup accelerates environment preparation and enhances experiment reproducibility by preventing version conflicts, with all required tools thoroughly verified and tested..

To run the UniDepth training pipeline in Docker from Visual Studio Code, install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension. For detailed guidance on using Dev Containers, see the Visual Studio Code documentation on [containers](https://code.visualstudio.com/docs/devcontainers/containers).

## Configuring your Development Environment

Before using Visual Studio Code with Dev Containers, you need to configure certain files to ensure it functions properly in your development environment.

__File `docker-compose.yaml`:__

Duplicate the docker-compose.yaml.sample file as docker-compose.yaml and update the following values below to match your environment:

```yaml
services:
  unidepth:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        TORCH_CUDA_ARCH_LIST: 8.6
        USER_UID: 1000
        USER_GID: 1000
        
    ports:
      - "43800:43800"
    volumes:
      - <data_folder>:/data
      - /mnt/data1/datasets:/data/datasets
    environment:
      - DATASETS: /data/datasets
      - DATA: /data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

- The `TORCH_CUDA_ARCH_LIST` argument should match your GPU architecture. Refer to the [NVIDIA documentation](https://developer.nvidia.com/cuda/gpus) for details.
- Volumes:
  - <DATA_FOLDER> should be replaced with the path to the folder you created for storing data.
  - <DATASETS_FOLDER> should be replaced with the path to the folder where you copied the dataset files.
- Service Name: If you plan to run multiple instances of the container, rename the service (currently `unidepth`) to a unique name for each instance.

__File `.devcontainer/docker-compose.yaml`:__

Duplicate the `.devcontainer/docker-compose.yaml.sample` file as `docker-compose.yaml` in the same folder, then update the values below to match your environment:

```yaml
services:
  # Update this to the name of the service you want to work with in your docker-compose.yml file
  unidepth:
    # Uncomment if you want to override the service's Dockerfile to one in the .devcontainer 
    # folder. Note that the path of the Dockerfile and context is relative to the *primary* 
    # docker-compose.yml file (the first in the devcontainer.json "dockerComposeFile"
    # array). The sample below assumes your primary file is in the root of your project.
    #
    # build:
    #   context: .
    #   dockerfile: .devcontainer/Dockerfile

    volumes:
      # Update this to wherever you want VS Code to mount the folder of your project
      - ./:/unidepth:cached

    # Uncomment the next four lines if you will use a ptrace-based debugger like C++, Go, and Rust.
    # cap_add:
    #   - SYS_PTRACE
    # security_opt:
    #   - seccomp:unconfined

    # Overrides default command so things don't shut down after the process ends.
    command: sleep infinity

```

- Service Name: Rename the service (currently `unidepth`) only if you changed it in the root `docker.compose.yaml` file.

These changes extend or overrides settings in the root `docker-compose.yaml` for use with Dev Containers. You can also modify other settings here as needed.

__File `.devcontainer/devcontainer.json`:__

Duplicate the `.devcontainer/devcontainer.json.sample` file as `devcontainer.json` in the same folder, then update the values below to match your environment:

```json
// For format details, see https://aka.ms/devcontainer.json. For config options, see the
// README at: https://github.com/devcontainers/templates/tree/main/src/docker-existing-docker-compose
{
	"name": "Docker Compose (Unidepth)", 

	// Update the 'dockerComposeFile' list if you have more compose files or use different names.
	// The .devcontainer/docker-compose.yml file contains any overrides you need/want to make.
	"dockerComposeFile": [
		"../docker-compose.yaml",
		"docker-compose.yaml"
	],

	// The 'service' property is the name of the service for the container that VS Code should
	// use. Update this value and .devcontainer/docker-compose.yml to the real service name.
	"service": "unidepth",

	// The optional 'workspaceFolder' property is the path VS Code should open by default when
	// connected. This is typically a file mount in .devcontainer/docker-compose.yml
	"workspaceFolder": "/unidepth",

	// Features to add to the dev container. More info: https://containers.dev/features.
	// "features": {},

	// Use 'forwardPorts' to make a list of ports inside the container available locally.
	// "forwardPorts": [],

	// Uncomment the next line if you want start specific services in your Docker Compose config.
	// "runServices": [],

	// Uncomment the next line if you want to keep your containers running after VS Code shuts down.
	// "shutdownAction": "none",

	// Uncomment the next line to run commands after the container is created.
	// "postCreateCommand": "cat /etc/os-release",

	// Configure tool-specific properties.
	"customizations": {
		"vscode": {
      		"extensions": ["ms-python.python"]
   		}
	}

	// Uncomment to connect as an existing user other than the container default. More info: https://aka.ms/dev-containers-non-root.
	// "remoteUser": "devcontainer"
}
```

- Service: Rename the service (currently `unidepth`) only if you modified it in the `docker.compose.yaml` files.

You can also adjust other Dev Containers settings in this file as needed.

## Open the Project in Dev Containers

In Visual Studio Code, open the Command Palette (View > Command Palette…), then select “Dev Containers: Rebuild and Reopen in Container” to build the container and reopen the project inside it. The process may take several minutes, depending on your internet connection speed.

Once the project is open in the container, confirm that it has access to the host GPUs by running the `nvidia-smi` command. The output should display a summary table listing all available GPUs.

## Fix ScanNet Dataset

The ScanNet dataset contains sequence frames that are not in chronological order. To train UniDepth with the equivariance loss, the frames must be properly ordered. To resolve this, run the `scripts/fix_scannet_dataset.py` script using a Visual Studio Code Terminal to ensure the command is executed within the container.

This script will modify the original file in place. While reordering the frames does not affect Unidepth original functionality, it may impact the exact reproducibility of experiments performed prior to this update.

## Download the Pre-Trained UniDepth Model.

To download the pre-trained weights for the UniDepth model, run the following command using a Visual Studio Code Terminal to ensure the command is executed within the container:

```bash
curl -L "https://huggingface.co/lpiccinelli/unidepth-v1-vitl14/resolve/main/pytorch_model.bin" -o "$DATA/models/pretrained/unidepth_v1_pytorch_model.bin"
```

Ensure that the path to the pre-trained weights is correctly set in the `training`>`pretrained` section of the training configuration file `configs/train_v1_vitl14_mod.json`.


## Run the UniDepth Training Script

Optionally, before starting the training of UniDepth model, you can initialize the experiment tracking tool Aim by running the `scripts/aim.sh` script. Use a dedicated Visual Studio Code terminal to ensure it runs continuously inside the container.

o access the Aim interface, open the following URL `http://<linux_host>:43800/` in your browser, where `<linux_host>` is the IP address or domain name of the Linux machine running the container.

To train UniDepth, run the `scripts/train.sh` script, also using a dedicated Visual Studio Code Terminal to ensure execution within the container.
