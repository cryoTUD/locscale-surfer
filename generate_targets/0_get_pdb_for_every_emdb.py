import os 
import requests
from bs4 import BeautifulSoup

def get_cleaned_pdb_ids_from_emdb(emdb_id):
    """
    Retrieves the associated PDB ID(s) for a given EMDB ID and cleans the output.

    Args:
        emdb_id (str): The EMDB ID (e.g., "5778").

    Returns:
        list: A cleaned list of associated PDB IDs, or None if none are found.
    """
    base_url = f"https://www.ebi.ac.uk/emdb/EMD-{emdb_id}"
    response = requests.get(base_url)

    if response.status_code != 200:
        return None  # Return None if the page doesn't load

    soup = BeautifulSoup(response.text, 'html.parser')
    pdb_ids = []

    # Extract PDB IDs from the webpage
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Identify and clean valid PDB IDs
        if 'fitted_pdbs' in href or '.cif' in href or href.strip().isalnum():
            pdb_id = href.split('/')[-1].split('.')[0]
            if pdb_id.isalnum() and len(pdb_id) == 4:
                pdb_ids.append(pdb_id)

    return sorted(set(pdb_ids)) if pdb_ids else None

import json
import os
from tqdm import tqdm
model_folder_main = "/home/abharadwaj1/scratch/dev/emmernet_training/training_cubes/deepEmhancer_223"
training_cubes_folder = os.path.join(model_folder_main, "cubedata_directory", "cubedata_training")

copied_file_path_json = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_datasets/create_test_set/copied_file_paths.json"
output_directory = os.path.dirname(copied_file_path_json)

# Load the copied file paths
with open(copied_file_path_json, "r") as f:
    copied_file_paths = json.load(f)


emdb_ids = [x.split("_")[1] for x in os.listdir(training_cubes_folder)]

opm_pdb_main_directory = "/home/abharadwaj1/scratch/dev/emmernet_training/pdb"
pdb_ids_in_opm = [x.split(".")[0] for x in os.listdir(opm_pdb_main_directory)]

emdb_pdb_dict = {}
emdb_ids_with_membrane = []
copied_file_paths_with_pdb = {}
for emdb_id in tqdm(emdb_ids, desc="Retrieving PDB IDs"):
    pdb_ids = get_cleaned_pdb_ids_from_emdb(emdb_id)
    if pdb_ids is None:
        continue
    
    if pdb_ids[0] in pdb_ids_in_opm:
        emdb_pdb_dict[emdb_id] = pdb_ids
        emdb_ids_with_membrane.append(emdb_id)
        


# Save the dictionary to a JSON file
output_file = os.path.join(model_folder_main, "0_emdb_pdb_dict.json")
emdb_ids_with_membrane_json_path = os.path.join(model_folder_main, "0_emdb_ids_with_membrane_proteins_dict.json")
emdb_ids_in_test_set_path = os.path.join(model_folder_main, "0_emdb_ids_in_test_set.json")
emdb_full_info_json_path = os.path.join(output_directory, "1_emdb_full_info.json")
# Check which EMDB IDs which are part of training set 
emd_ids_used_in_training_path = "/home/abharadwaj1/scratch/dev/emmernet_training/training_cubes/hybrid_model_map_MA_new/6_emdb_full_info_with_micelle_closed.json"
with open(emd_ids_used_in_training_path, "r") as f:
    emd_ids_used_in_training = json.load(f)

emdb_ids_in_test_set = [x for x in emdb_ids_with_membrane if x not in emd_ids_used_in_training.keys()]

for emdb_id in emdb_ids_in_test_set:
    copied_file_paths_with_pdb[emdb_id] = copied_file_paths[f"emd_{emdb_id}"].copy()
    copied_file_paths_with_pdb[emdb_id]['pdb_ids'] = emdb_pdb_dict[emdb_id]

with open(output_file, "w") as f:
    json.dump(emdb_pdb_dict, f, indent=4)

with open(emdb_ids_with_membrane_json_path, "w") as f:
    json.dump(emdb_ids_with_membrane, f, indent=4)

with open(emdb_ids_in_test_set_path, "w") as f:
    json.dump(emdb_ids_in_test_set, f, indent=4)

with open(emdb_full_info_json_path, "w") as f:
    json.dump(copied_file_paths_with_pdb, f, indent=4)



# Print some statistics
print(f"EMDB Ids with PDB found : {len(emdb_pdb_dict)}")
print(f"EMDB Ids with Membrane Proteins : {len(emdb_ids_with_membrane)}")
print(f"EMDB Ids in Test Set : {len(emdb_ids_in_test_set)}")
print(f"EMDB Full Info : {len(copied_file_paths_with_pdb)}")

print(f"EMDB PDB saved to : {output_file}")
print(f"EMDB IDs with Membrane Proteins saved to : {emdb_ids_with_membrane_json_path}")
print(f"EMDB IDs in Test Set saved to : {emdb_ids_in_test_set_path}")
print(f"EMDB Full Info saved to : {emdb_full_info_json_path}")                                

