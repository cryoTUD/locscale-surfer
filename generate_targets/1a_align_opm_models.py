import os 
import numpy as np 
import gemmi 
import matplotlib.pyplot as plt
import json
import sys 
sys.path.append('/home/abharadwaj1/soft/students/segmentation_of_micelles/segmentation/utility')

from locscale.include.emmer.pdb.pdb_utils import compute_rmsd_two_pdb, get_coordinates
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc, convert_pdb_to_mrc_position
from locscale.include.emmer.pdb.pdb_to_map import pdb2map


# Custom functions
from useful_functions import extract_dummy_residues_from_pdb
from visualization_tools import project_map, plot_projections, plot_density_contour_map

def calculate_number_of_DUM_residues(st):
    num_res = 0
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == "DUM":
                    num_res += 1
    return num_res
    
def align_pdb_structures(reference_pdb_path, target_pdb_path, aligned_pdb_path):
    ''' Align the target pdb structure to the reference pdb structure using gemmi library
    Args:
        reference_pdb_path (str): Path to the reference pdb file
        target_pdb_path (str): Path to the target pdb file
    '''
    import gemmi 
    # Load the reference and target pdb structures
    # calculate number of DUM residues in the reference pdb
 
    reference_structure = gemmi.read_structure(reference_pdb_path)
    target_structure = gemmi.read_structure(target_pdb_path)

    reference_model = reference_structure[0]
    target_model = target_structure[0]

    chain_1_ref = reference_model[0]
    chain_1_target = target_model[0]

    # Create a transformation matrix 
    polymer_ref = chain_1_ref.whole()
    polymer_target = chain_1_target.whole()
    polymer_type = polymer_ref.check_polymer_type()
    sup = gemmi.calculate_superposition(polymer_ref, polymer_target, polymer_type, gemmi.SupSelect.All)

    # Apply the transformation matrix to the target structure
    target_model.transform_pos_and_adp(sup.transform)

    # Save the aligned pdb file
    aligned_structure = gemmi.Structure()
    aligned_structure.add_model(target_model)
    aligned_structure.write_pdb(aligned_pdb_path)

    print("Number of DUM residues in the aligned pdb: ", calculate_number_of_DUM_residues(aligned_structure), " path: ", aligned_pdb_path)
    # Calculate the RMSD between the reference and target structures
    rmsd_superposition = sup.rmsd
    print(f"RMSD between the reference and target structures: {rmsd_superposition:.2f} Å")
    return rmsd_superposition



emdb_info_full = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_dataset_low_pass_filter/create_test_set/2a_emdb_full_info_with_lowpass_mask.json"

pdbs_from_opm = "/home/abharadwaj1/scratch/dev/segmentation_micelle/corrected_pdb"
aligned_pdb_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/aligned_pdb_from_opm"
os.makedirs(aligned_pdb_folder, exist_ok=True)

with open(emdb_info_full, "r") as f:
    emdb_dict = json.load(f)

emdb_dict_aligned = {}
for i, emd in enumerate(emdb_dict):
    print(f"Processing {emd} ({i+1}/{len(emdb_dict)})")
    pdb_id = emdb_dict[emd]["pdb_ids"][0]
    pdb_path = emdb_dict[emd]["pdb_path"]

    non_aligned_pdb_path = os.path.join(pdbs_from_opm, f"{pdb_id}.pdb")
    aligned_pdb_path = os.path.join(aligned_pdb_folder, f"{pdb_id}.pdb")
    if not os.path.exists(aligned_pdb_path):
        
        if not os.path.isfile(non_aligned_pdb_path):
            print(f"{non_aligned_pdb_path} does not exist")
            continue

        rmsd_superposition = align_pdb_structures(
            pdb_path, 
            non_aligned_pdb_path, 
            aligned_pdb_path
        )
    else:
        rmsd_superposition = -1
        
    emdb_dict[emd]["aligned_pdb_path"] = aligned_pdb_path
    emdb_dict[emd]["rmsd_superposition"] = rmsd_superposition

# Save the dictionary to a JSON file
output_file = os.path.join(os.path.dirname(emdb_info_full), "3_emdb_full_info_with_aligned_pdb.json")

# Print statistics of rmsd superposition
from scipy.stats import describe
rmsd_values = [emdb_dict[emd]["rmsd_superposition"] for emd in emdb_dict]
rmsd_stats = describe(rmsd_values)
min_rmsd = min(rmsd_values)
max_rmsd = max(rmsd_values)
print(f"RMSD statistics: {rmsd_stats}")
# print quartiles at 25, 50, 75 percentiles
quartiles = np.percentile(rmsd_values, [25, 50, 75])
print("Min RMSD: ", min_rmsd)
print(f"RMSD quartiles: {quartiles}")
print("Max RMSD: ", max_rmsd)


with open(output_file, "w") as f:
    json.dump(emdb_dict, f, indent=4)




