import os
import sys
sys.path.append('/home/abharadwaj1/soft/students/segmentation_of_micelles/segmentation/utility')
#os.environ["PYTHONPATH"] = '/home/tnw-nb4020-03/bep_lotte_micelle/notebooks/utility'

### imports
import json
import numpy as np
from find_micelles import find_number_of_membranes, extract_dummy_residues_from_pdb, get_membrane
from locscale.include.emmer.ndimage.map_utils import load_map
import time
import matplotlib.pyplot as plt
import os
import random
random.seed(42)
from visualization_tools import project_map, plot_projections, plot_density_contour_map, plot_density_contour_map
from tqdm import tqdm
emdb_info_full_json = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_dataset_low_pass_filter/create_test_set/5_emdb_full_info_with_micelle.json"

with open(emdb_info_full_json, "r") as f:
    emdb_info_full = json.load(f)

micelle_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/micelle_remove_extra_floating_objects_test_set_low_pass"

pdb_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/aligned_pdb_from_opm"
unmodelled_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/unmodelled_regions_test_set_low_pass"

rows = 3
cols = 6
fig, ax = plt.subplots(rows,cols, dpi=300)

map_nr    = 0
index_col = 0
index_row = 0

emdb_ids = [emd for emd in emdb_info_full]
num_plots = rows*cols

sampled_emdb_ids = random.sample(emdb_ids, num_plots)
for emdb_id in tqdm(sampled_emdb_ids):
    if index_col == cols:
        index_col = 0
        index_row += 1
    if index_row == rows:
        break
        
    micelle = load_map(os.path.join(micelle_folder, f"emd_{emdb_id}_cleaned_micelle.mrc"))[0]
    projection = project_map(micelle, "x", projection_type="mean")
    ax[index_row,index_col].imshow(projection)

    ax[index_row,index_col].get_xaxis().set_visible(False)
    ax[index_row,index_col].get_yaxis().set_visible(False)

    ax[index_row,index_col].set_title("{}".format(emdb_id), fontsize=8)

    index_col += 1

 



plt.tight_layout()

output_directory = os.path.dirname(emdb_info_full_json)
save_path = os.path.join(output_directory, "1d_micelle_projections_test_set_x.png")
plt.savefig(save_path, dpi=300)
