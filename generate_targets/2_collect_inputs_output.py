import os
import sys
sys.path.append('/home/abharadwaj1/soft/students/segmentation_of_micelles/segmentation/utility')
#os.environ["PYTHONPATH"] = '/home/tnw-nb4020-03/bep_lotte_micelle/notebooks/utility'

### imports
import json
import numpy as np
from find_micelles import find_number_of_membranes, extract_dummy_residues_from_pdb, get_membrane
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
import time
import matplotlib.pyplot as plt
import os
from visualization_tools import project_map, plot_projections, plot_density_contour_map, plot_density_contour_map
from tqdm import tqdm
import random
from scipy.ndimage import binary_closing, generate_binary_structure
from skimage.morphology import ball
import joblib   
import pickle
import shutil

emdb_info_full_json = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_datasets/create_test_set/6_emdb_full_info_with_micelle_closed.json"

with open(emdb_info_full_json, "r") as f:
    emdb_info_full = json.load(f)

output_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/test_set_collections"
os.makedirs(output_folder, exist_ok=True)

def copy_files_to_folder(source_file, destination_folder):
    assert os.path.isfile(source_file), f"File {source_file} does not exist"
    assert os.path.isdir(destination_folder), f"Folder {destination_folder} does not exist"
    new_file_path = os.path.join(destination_folder, os.path.basename(source_file))
    if os.path.isfile(new_file_path):
        return new_file_path
    else:
        shutil.copyfile(source_file, new_file_path)
        return new_file_path

new_collected_info = {}
for emdb_id in tqdm(emdb_info_full.keys()):
        # "X_path":
        # "Y_path":
        # "M_path":
        # "pdb_ids": 
        # "pdb_path": 
        # "aligned_pdb_path": "/home/abharadwaj1/scratch/dev/segmentation_micelle/aligned_pdb_from_opm_test_set/5lc5.pdb",
        # "rmsd_superposition": 0.0005036239330400339,
        # "difference_mask_path": "/home/abharadwaj1/scratch/dev/segmentation_micelle/unmodelled_regions_test_set/emd_4032_difference_mask.mrc",
        # "micelle_path": "/home/abharadwaj1/scratch/dev/segmentation_micelle/micelle_test_set/emd_4032_micelle.mrc",
        # "micelle_save_path": "/home/abharadwaj1/scratch/dev/segmentation_micelle/binary_closed_micelle_test_set/emd_4032_micelle_closed.mrc",
        # "volume_micelle_percentage": 0.8831125685871056

    unsharpened_path = emdb_info_full[emdb_id]["X_path"]
    mask_path = emdb_info_full[emdb_id]["M_path"]
    pdb_path = emdb_info_full[emdb_id]["aligned_pdb_path"]
    micelle_path = emdb_info_full[emdb_id]["micelle_save_path"]

    # copy the files to the output folder
    new_unsharpened_path = copy_files_to_folder(unsharpened_path, output_folder)
    new_mask_path = copy_files_to_folder(mask_path, output_folder)
    new_pdb_path = copy_files_to_folder(pdb_path, output_folder)
    new_micelle_path = copy_files_to_folder(micelle_path, output_folder)

    # make the paths relative
    relative_unsharpened_path = os.path.relpath(new_unsharpened_path, output_folder)
    relative_mask_path = os.path.relpath(new_mask_path, output_folder)
    relative_pdb_path = os.path.relpath(new_pdb_path, output_folder)
    relative_micelle_path = os.path.relpath(new_micelle_path, output_folder)
                                            
    new_collected_info[emdb_id] = {
        "X_path": relative_unsharpened_path,
        "M_path": relative_mask_path,
        "pdb_path": relative_pdb_path,
        "micelle_path": relative_micelle_path,
        "volume_micelle_percentage": emdb_info_full[emdb_id]["volume_micelle_percentage"]
    }

output_json_path = os.path.join(output_folder, "collected_paths_relative.json")

with open(output_json_path, "w") as f:
    json.dump(new_collected_info, f, indent=4)


