#!/usr/bin/env python3
import argparse
import atexit
import cv2
import h5py
import json
import logging
import numpy as np
import os
import re
import tables
import torch
import torch.multiprocessing as mp

from superglue.models.matching import Matching

import unidepth.datasets as datasets


##### STEPS #####

def order_sequences(dataset_name):

    # Get information from dataset class
    dataset = getattr(datasets, dataset_name)
    hdf5_paths = dataset.hdf5_paths
    sequences_file = dataset.sequences_file

    # Iterate dataset files
    for hdf5_path in hdf5_paths:
        
        # Read sequence from dataset
        DATAROOT = os.environ["DATAROOT"]
        hdf5_fullpath = os.path.join(DATAROOT, hdf5_path)
        with h5py.File(hdf5_fullpath, "r") as hdf5_file:
            sequences = json.loads(np.array(hdf5_file[sequences_file]).tobytes().decode("ascii").strip())
            
        # Iterage all sequences in alphabetical order
        ordered_sequences = {}
        for sequence_key in sorted(list(sequences.keys())):
                
            # Get the sorting array for sequence's images
            sequence = sequences[sequence_key]
            num_images = len(sequence["image"])
            sorted_indexes = sorted(range(num_images), key=lambda i: int(re.search(r"(\d+)(?=\D*$)", sequence["image"][i]).group(1)))
            
            # Apply the sorting order to all sequence arrays.
            ordered_sequence = {} 
            for key, value in sequence.items():
                ordered_sequence[key] = [value[i] for i in sorted_indexes]
            
            # Add sequence to sequences
            ordered_sequences[sequence_key] = ordered_sequence

        # Save ordered sequences in dataset
        json_string = json.dumps(ordered_sequences)
        with h5py.File(hdf5_fullpath, "a") as hdf5_file:
            del hdf5_file[sequences_file]
            hdf5_file.create_dataset(sequences_file, data=np.frombuffer(json_string.encode('ascii'), dtype=np.int8))

def create_version_file(dataset_name, version):
    
    # Get information from dataset class
    dataset = getattr(datasets, dataset_name)
    hdf5_paths = dataset.hdf5_paths
    sequences_file = dataset.sequences_file

    # Iterate dataset files
    for hdf5_path in hdf5_paths:

        # Read sequence from dataset
        DATAROOT = os.environ["DATAROOT"]
        hdf5_fullpath = os.path.join(DATAROOT, hdf5_path)
        with h5py.File(hdf5_fullpath, "r") as hdf5_file:
            sequences = json.loads(np.array(hdf5_file[sequences_file]).tobytes().decode("ascii").strip())

        # Save version
        DATAROOT = os.environ["DATAROOT"]
        hdf5_fullpath = os.path.join(DATAROOT, hdf5_path)
        with h5py.File(hdf5_fullpath, "a") as hdf5_file:
            if "VERSION" in  hdf5_file:
                del hdf5_file["VERSION"]
            hdf5_file.create_dataset("VERSION", data=np.frombuffer(version.encode('ascii'), dtype=np.int8))
        logging.debug(f"New version: {version}")

### STEPS: MATCH KEYPOINTS ###

def calculate_matches(dataset_name, num_workers):

    # Get information from dataset class
    dataset = getattr(datasets, dataset_name)
    hdf5_paths = dataset.hdf5_paths
    sequences_file = dataset.sequences_file

    # Iterate dataset files
    for hdf5_path in hdf5_paths:

        # Read sequence from dataset
        DATAROOT = os.environ["DATAROOT"]
        hdf5_fullpath = os.path.join(DATAROOT, hdf5_path)
        with h5py.File(hdf5_fullpath, "r") as hdf5_file:
            sequences = json.loads(np.array(hdf5_file[sequences_file]).tobytes().decode("ascii").strip())
        ranked_sequences = [(i, key, value) for i, (key, value) in enumerate(sequences.items())]
        
        # Process sequences in different processes
        init_worker_config = {
            "hdf5_path": hdf5_fullpath,
            "supgerglue_model": config["preprocess"]["matching"]["superglue_models"][dataset_name],
            "max_windows_size": config["preprocess"] ["matching"]["max_windows_size"],
            "num_workers": config["preprocess"]["matching"]["num_workers"]
        }
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(init_worker_config,)) as pool:
            results = pool.map(process_sequence, ranked_sequences) 

        # Concatenate results in single array
        results = [result for sequence_result in results for result in sequence_result]
    
        # Save results to JSON Lines file
        json_lines_path = os.path.splitext(hdf5_fullpath)[0] + ".jsonl"
        with open(json_lines_path, 'w') as file:
            for result in results:
                file.write(json.dumps(result) + '\n')
        
        # Log
        logging.debug(f"Created file: {json_lines_path}")

DEFAULT_SUPERGLUE_CONFIG = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048
    },
    "superglue": {
        "weights": "indoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2
    }
}

process_data = {}

def init_worker(init_worker_config):

    # Log
    logging.debug("Initializing worker")

    # Get worker configuration
    global process_data
    process_data["config"] = init_worker_config

    # Open HDF5 file
    hdf5_path = init_worker_config["hdf5_path"]
    hdf5_file = tables.open_file(hdf5_path)
    process_data["hdf5_file"] = hdf5_file
    logging.debug(f"Open HDF5 file: {hdf5_path}")

    # Create SuperGlue instance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    superglue_config = DEFAULT_SUPERGLUE_CONFIG
    superglue_config["superglue"]["weights"] = init_worker_config["supgerglue_model"]
    process_data["matcher"] = Matching(superglue_config).eval().to(device)
    process_data["matcher_device"] = device
    print(f"Matcher created on device: {device}")

    # Automatically close hdf5 file when process exits
    atexit.register(lambda: hdf5_file.close())

def process_sequence(ranked_sequence):

    # Get configuration
    global process_data
    hdf5_file = process_data["hdf5_file"]
    num_workers = process_data["config"]["num_workers"]
    max_windows_size = process_data["config"]["max_windows_size"]
    matcher = process_data["matcher"]
    device = process_data["matcher_device"]

    # Unpack arguments
    sequence_num, sequence_name, sequence = ranked_sequence

    # Get ordered list of images
    image_names = sequence["image"]
    num_images = len(image_names)

    # Log sequence
    print(f"Processing sequence {sequence_name} with {num_images} images")

    # Array to store results
    records = []

    # Iterate all images in the sequence
    for i in range(num_images - 1):

        # Read first image from dataset
        name_image_a = f"/{image_names[i]}"
        data = hdf5_file.get_node(name_image_a).read()
        image_a = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        tensor_image_a = torch.from_numpy(image_a/255.).float()[None, None].to(device)

        # Log
        if (sequence_num % num_workers == 0) and (i % 10) == 0:
           print(f"Processing image: {name_image_a} ({i} out of {num_images})")

        
        # Get camera matrix and extrinsics matrix
        k_a = np.array(sequence["K"][i])
        cam2w_a = np.array(sequence["cam2w"][i])
    
        # Iterate images forward
        max_j = i + max_windows_size if i + max_windows_size < num_images else num_images
        for j in range(i + 1, max_j):

            # Create new record
            record = {}
            record["sequence_name"] = sequence_name
                
            try:
                
                # Read second image from dataset
                name_image_b = f"/{image_names[j]}"
                data = hdf5_file.get_node(name_image_b).read()
                image_b = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                tensor_image_b = torch.from_numpy(image_b/255.).float()[None, None].to(device)
    
                # Add data to record
                record["name_image_a"] = name_image_a
                record["name_image_b"] = name_image_b
                record["distance_in_frames"] = j - i
            
                # Get camera matrix and extrinsics matrix
                k_b = np.array(sequence["K"][j])
                cam2w_b = np.array(sequence["cam2w"][j])
    
                # Match features
                result = match_features(matcher, tensor_image_a, tensor_image_b)
                keypoints_a = result["pred"]["keypoints0"]
                keypoints_b = result["pred"]["keypoints1"]
                matched_keypoints_a = result["matched_keypoints_image_0"]
                matched_keypoints_b = result["matched_keypoints_image_1"]
            
                # Add data to record
                record["num_features_image_a"] = len(keypoints_a)
                record["num_features_image_b"] = len(keypoints_b)
                record["num_matched_features"] = len(matched_keypoints_a)

            except Exception as ex:
                record["error"] = str(ex)
                break
                                
            finally:
                # Save record to JSON lines
                records.append(record)
            
    return records

def match_features (matching, image_0, image_1, result=dict()):

    # Match features using SuperGlue
    with torch.no_grad():
        pred = matching({"image0": image_0, "image1": image_1})

    # Convert prediction values to numpy arrays
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    # Save prediction in result variable
    result["pred"] = pred
    
    # Extract variables from npz file
    keypoints_image_0 = pred["keypoints0"]
    keypoints_image_1 = pred["keypoints1"]
    matches = pred["matches0"]
    
    # filter valid matches
    valid_matches = matches != -1

    # Create two arrays with the valid matches
    matched_keypoints_image_0 = keypoints_image_0[valid_matches]
    matched_keypoints_image_1 = keypoints_image_1[matches[valid_matches]]
    result["matched_keypoints_image_0"] = matched_keypoints_image_0
    result["matched_keypoints_image_1"] = matched_keypoints_image_1
    
    return result

# Define function to calculate camera pose
def estimate_camera_pose(keypoints_0, k_0, keypoints_1, k_1, result = dict()):

    # Check lengths of keypoint arrays
    if len(keypoints_0) != len(keypoints_1):
        raise Exception("Invalid keypoints")
        
    # Check if K matrix are equal
    if not np.array_equal(k_0, k_1):
        print(f"K0: {k_0}")
        print(f"K1: {k_1}")
        raise Exception("Camera intrinsic parameters are not equal")
    
    # Calculate Essential Matrix
    E, mask = cv2.findEssentialMat(keypoints_0, keypoints_1, k_0, method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
    result["E"] = E
    result["mask_1"] = mask = mask.ravel()

    # Filter outliers from keypoints
    keypoints_0 = keypoints_0[mask == 1]
    keypoints_1 = keypoints_1[mask == 1]

    # Calculate camera translation and rotation
    inliers, R, t, mask = cv2.recoverPose(E, keypoints_0, keypoints_1, k_0)
    result["R"] = R
    result["t"] = t.squeeze()
    result["inliers"] = inliers
    result["mask_2"] = mask = mask.ravel()
    
    return result

# Define function to calculate inverse matrix
def invert_matrix(T):

    if T.shape == (3, 3):
        R_inv = T.T
        return R_inv
        
    elif T.shape == (4, 4):
        R = T[:3, :3]  # Rotation part
        t = T[:3, 3]   # Translation part
        
        R_inv = R.T             # Transpose of rotation matrix
        t_inv = -R_inv @ t      # Inverted translation
        
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        
        return T_inv
    else:
        raise RuntimeError("Shape not supported")
    
##### MAIN METHOD #####

def main(config):

    # Read configuration values
    num_workers = config["preprocess"]["matching"]["num_workers"]

    # Iterate datasets
    for dataset_name in config["preprocess"]["datasets"]:
        logging.debug(f"Dataset: {dataset_name}")
    
        # Order sequences file for all datasets
        if config["preprocess"]["steps"]["order_sequences"]:
            logging.debug(f"Order sequence: {dataset_name}")
            order_sequences(dataset_name)

        # Match features between neighboor frames
        if  config["preprocess"]["steps"]["calculate_matches"]:
            logging.debug(f"Calculate matches: {dataset_name}")
            calculate_matches(dataset_name, num_workers)
    
        # Save a version file
        if config["preprocess"]["steps"]["update_version"]:
            logging.debug(f"Update version: {dataset_name}")
            create_version_file(dataset_name, config["preprocess"]["new_version"])

        
if __name__ == "__main__":

    # Configure the logging system
    logging.basicConfig(level=logging.DEBUG)

    # Configure torch multiprocessing
    mp.set_start_method('spawn')

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Preprocess UniDepth Datasets"
    )
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()
    logging.debug(f"Arguments: {args}")

    # Open configuration
    with open(args.config_file, "r") as file:
        config = json.load(file)
    logging.debug(f"Configuration: {config}")

    # Call main method
    main(config)

# DATAROOT="/data/datasets" PYTHONPATH=. python preprocess/preprocess.py --config-file preprocess/config-preprocess.json