#!/usr/bin/env python 

import json
import numpy as np
import os
import re
import h5py

# Open dataset file
datasets_folder = os.environ["DATASETS"]
hdf5_filepath = os.path.join(datasets_folder, "ScanNetS.hdf5")

with h5py.File(hdf5_filepath, "a") as hdf5_file:
    
    # Sort frames for every sequence
    sequences = json.loads(np.array(hdf5_file["sequences.json"]).tobytes().decode("ascii").strip())
    ordered_sequences = {}
    
    # Iterate sequences in alphabetical order
    for sequence_key in sorted(list(sequences.keys())):
    
        # Get corrent sequence
        sequence = sequences[sequence_key]
    
        # ict_keys(['image', 'depth', 'cam2w', 'K'])
        
        # Get the sorting array for sequence's images
        num_images = len(sequence["image"])
        index_order = sorted(range(num_images), key=lambda i: int(re.search(r'(\d+)(?=\D*$)', sequence["image"][i]).group(1)))
    
        # Create a sequence with ordered frames
        ordered_sequence = {}
        for i in index_order:
            # Order arrays of sequence
            ordered_sequence["image"] = [sequence["image"][i] for i in index_order]
            ordered_sequence["depth"] = [sequence["depth"][i] for i in index_order]
            ordered_sequence["cam2w"] = [sequence["cam2w"][i] for i in index_order]
            ordered_sequence["K"] = [sequence["K"][i] for i in index_order]
    
        # Add sequence to sequences
        ordered_sequences[sequence_key] = ordered_sequence
    
    # Replace previous sequences
    if sequences != ordered_sequences:
    
        # Delete the old dataset first
        if "sequences.json" in hdf5_file:
            del hdf5_file["sequences.json"]
    
        # Store as a uint8 byte array (exactly like before)
        hdf5_file.create_dataset(
            "sequences.json",
            data=np.frombuffer(json.dumps(ordered_sequences).encode("ascii"), dtype="uint8")
        )
        
        print("File 'ScanNetS.hdf5' has been successfully updated")

    else:

        print("File 'ScanNetS.hdf5' is already updated")