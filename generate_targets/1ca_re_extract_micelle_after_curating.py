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
from skimage.measure import label, regionprops
from find_micelles import find_number_of_membranes, extract_dummy_residues_from_pdb, get_membrane
from visualization_tools import project_map, plot_projections, plot_density_contour_map

import warnings
warnings.filterwarnings("ignore")

def remove_floating_objects(emdb_info, emdb_id, output_dir):
    try:
        output_file_path=os.path.join(output_dir, f"emd_{emdb_id}_micelle.mrc")
        if os.path.isfile(output_file_path):
            return emdb_id, output_file_path
        micelle_extracted_path = emdb_info["micelle_path"]
        

        micelle_map, apix = load_map(micelle_extracted_path)
        micelle_binarised = micelle_map > 0.5
        # Get the largest connected component using label
        labelled_micelle = label(micelle_binarised)
        regions = regionprops(labelled_micelle)
        largest_region = max(regions, key=lambda x: x.area)
        largest_region_mask = (labelled_micelle == largest_region.label).astype(int)
               
        
        output_file_path=os.path.join(output_dir, f"emd_{emdb_id}_cleaned_micelle.mrc")
        
        save_as_mrc(largest_region_mask, output_file_path, apix)
        return emdb_id, output_file_path
    except Exception as e:
        print("Error: ", e)

        return None

emdb_info_full_json = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_dataset_low_pass_filter/create_test_set/5_emdb_full_info_with_micelle.json"

micelle_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/micelle_remove_extra_floating_objects_test_set_low_pass"
os.makedirs(micelle_folder, exist_ok=True)

with open(emdb_info_full_json, "r") as f:
    emdb_info_full = json.load(f)

emdb_info_full_with_micelle = {}

n_jobs = 10
results = joblib.Parallel(n_jobs=n_jobs)(
    joblib.delayed(remove_floating_objects)\
    (emdb_info, emdb_id, micelle_folder) \
    for emdb_id, emdb_info in tqdm(emdb_info_full.items(), desc="Cleaning micelle")
)

results_filtered = [x for x in results if x is not None]
for emdb_id, output_file_path in results_filtered:
    emdb_info_full_with_micelle[emdb_id] = emdb_info_full[emdb_id].copy()
    emdb_info_full_with_micelle[emdb_id]["clean_micelle_path"] = output_file_path

output_json_path = os.path.join(os.path.dirname(emdb_info_full_json), "6_emdb_full_info_with_clean_micelle.json")
with open(output_json_path, "w") as f:
    json.dump(emdb_info_full_with_micelle, f, indent=4)

# Print the number of micelles extracted
print(f"Number of micelles extracted: {len(emdb_info_full_with_micelle)}")