import os
from tqdm import tqdm
from locscale.preprocessing.headers import run_FDR, check_axis_order
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
from locscale.utils.math_tools import round_up_to_even
# make small script to remove end if in pdb file
emdb_info_json = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_dataset_low_pass_filter/create_test_set/2_emdb_full_info_with_pdb.json"
import json 
with open(emdb_info_json, "r") as f:
    emdb_info_full = json.load(f)

emdb_info_with_lowpass_mask = emdb_info_full.copy()

for emdb_id in tqdm(emdb_info_full):
    
    emmap_path = emdb_info_with_lowpass_mask[emdb_id]["X_path"]
    confidence_map_path = emmap_path.replace(".mrc", "_confidenceMap.mrc")
    if os.path.exists(confidence_map_path):
        emdb_info_with_lowpass_mask[emdb_id].update({"M_path_low_pass": confidence_map_path})
        continue
    emmap, apix = load_map(emmap_path)

    fdr_inputs_dictionary = {
        "emmap_path": emmap_path,
        "window_size": round_up_to_even(emmap.shape[0] * 0.1),
        "fdr": 0.01,
        "filter_cutoff": 5,
        "averaging_filter_size": 3
    }
    
    
    mask_path, mask_path_raw = run_FDR(**fdr_inputs_dictionary)
    confidence_map_path = mask_path

    emdb_info_with_lowpass_mask[emdb_id].update({"M_path_low_pass": confidence_map_path})
    

output_file = os.path.join(os.path.dirname(emdb_info_json), "2a_emdb_full_info_with_lowpass_mask.json")
with open(output_file, "w") as f:
    json.dump(emdb_info_with_lowpass_mask, f, indent=4)

    




