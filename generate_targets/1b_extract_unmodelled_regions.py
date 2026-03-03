import os 
import numpy as np 
import gemmi 
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import joblib
import sys 
sys.path.append('/home/abharadwaj1/soft/students/segmentation_of_micelles/segmentation/utility')

from locscale.include.emmer.pdb.pdb_utils import compute_rmsd_two_pdb, get_coordinates
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc, convert_pdb_to_mrc_position
from locscale.include.emmer.pdb.pdb_to_map import pdb2map
from locscale.include.emmer.ndimage.map_tools import find_unmodelled_mask_region

# Custom functions
from useful_functions import extract_dummy_residues_from_pdb
from visualization_tools import project_map, plot_projections, plot_density_contour_map


def save_unmodelled_regions(emdb_info, emdb_id, output_dir):
    try:
        output_file_path=os.path.join(output_dir, f"emd_{emdb_id}_difference_mask_lowpass.mrc")
        if os.path.isfile(output_file_path):
            return emdb_id, output_file_path
        mask_path = emdb_info["M_path_low_pass"]
        pdb_path = emdb_info["pdb_path"]
        difference_mask = find_unmodelled_mask_region(
            mask_path, pdb_path, \
            fdr_threshold=0.5, atomic_mask_threshold=0.5, \
            averaging_window_size=3\
        )
        _, apix = load_map(mask_path)
        output_file_path=os.path.join(output_dir, f"emd_{emdb_id}_difference_mask.mrc")
        save_as_mrc(difference_mask, output_file_path, apix)
        return emdb_id, output_file_path
    except Exception as e:
        return None
emdb_info_full_json = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_dataset_low_pass_filter/create_test_set/3_emdb_full_info_with_aligned_pdb.json"

difference_mask_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/unmodelled_regions_test_set_low_pass"
os.makedirs(difference_mask_folder, exist_ok=True)

with open(emdb_info_full_json, "r") as f:
    emdb_info_full = json.load(f)

n_jobs = 10 
emdb_info_with_difference_mask = {}

results = joblib.Parallel(n_jobs=n_jobs)(
    joblib.delayed(save_unmodelled_regions)\
    (emdb_info, emdb_id, difference_mask_folder) \
    for emdb_id, emdb_info in tqdm(emdb_info_full.items(), desc="Saving difference masks")
)

results_filtered = [x for x in results if x is not None]
for emdb_id, output_file_path in results_filtered:
    emdb_info_with_difference_mask[emdb_id] = emdb_info_full[emdb_id].copy()
    emdb_info_with_difference_mask[emdb_id]["difference_mask_path"] = output_file_path


output_json_path = os.path.join(os.path.dirname(emdb_info_full_json), "4_emdb_full_info_with_difference_mask.json")
with open(output_json_path, "w") as f:
    json.dump(emdb_info_with_difference_mask, f, indent=4)




