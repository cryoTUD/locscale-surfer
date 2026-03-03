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
from visualization_tools import project_map, plot_projections, plot_density_contour_map, plot_density_contour_map
from tqdm import tqdm
import random

emdb_info_full_json = "/home/abharadwaj1/scratch/dev/emmernet_training/training_cubes/hybrid_model_map_MA_new/5_emdb_full_info_with_micelle.json"

with open(emdb_info_full_json, "r") as f:
    emdb_info_full = json.load(f)

micelle_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/micelle"

statistics = {}
# for each micelle, extract basic statistics to gauge the quality of the micelle

# percentage of voxels that are not zero
for emdb_id in tqdm(emdb_info_full):
    micelle = load_map(os.path.join(micelle_folder, f"emd_{emdb_id}_micelle.mrc"))[0]
