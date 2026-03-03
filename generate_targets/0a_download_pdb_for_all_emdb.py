
import os 
import requests
from bs4 import BeautifulSoup

def download_pdb_file(pdb_id, output_dir):
    """
    Downloads a PDB file from the RCSB PDB database.

    Args:
        pdb_id (str): The PDB ID (e.g., "1abc").
        output_dir (str): The directory to save the PDB file.

    Returns:
        str: The path to the downloaded PDB file, or None if the download fails.
    """
    base_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(base_url)

    if response.status_code != 200:
        return None  # Return None if the download fails

    output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    with open(output_file, "wb") as f:
        f.write(response.content)

    return output_file

import json
import os
from tqdm import tqdm
model_folder_main = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_datasets/create_test_set"
emdb_dictionary = os.path.join(model_folder_main, "1_emdb_full_info.json")

output_dir = os.path.join(model_folder_main, "pdb_files")
os.makedirs(output_dir, exist_ok=True)

with open(emdb_dictionary, "r") as f:
    emdb_dict = json.load(f)

emdb_ids = list(emdb_dict.keys())
emdb_pdb_dict = {}
for emdb_id in tqdm(emdb_ids, desc="Downloading PDB IDs"):
    pdb_id = emdb_dict[emdb_id]["pdb_ids"]
    pdb_file_path = download_pdb_file(pdb_id[0], output_dir)
    if pdb_file_path is not None and os.path.isfile(pdb_file_path):
        emdb_pdb_dict[emdb_id] = emdb_dict[emdb_id].copy()
        emdb_pdb_dict[emdb_id]['pdb_path'] = pdb_file_path
    else:
        print(f"Download failed for {emdb_id}")

# Save the dictionary to a JSON file
output_file = os.path.join(model_folder_main, "2_emdb_full_info_with_pdb.json")
with open(output_file, "w") as f:
    json.dump(emdb_pdb_dict, f, indent=4)

        


    
        