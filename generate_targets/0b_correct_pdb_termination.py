import os
from tqdm import tqdm
# make small script to remove end if in pdb file
outp_folder = "/home/abharadwaj1/scratch/dev/segmentation_micelle/corrected_pdb_test_set"
inp_folder = "/home/abharadwaj1/scratch/dev/emmernet_training/pdb"
os.makedirs(outp_folder, exist_ok=True)
emdb_info_json = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_datasets/create_test_set/2_emdb_full_info_with_pdb.json"
import json 
with open(emdb_info_json, "r") as f:
    emdb_info = json.load(f)

pdb_ids = [emdb_info[emd]["pdb_ids"][0] for emd in emdb_info]
for file in tqdm(os.listdir(inp_folder)):
    pdb_id = file.split(".")[0]
    if pdb_id not in pdb_ids:
        continue
    inp_path  = os.path.join(inp_folder, pdb_id + ".pdb")
    outp_path = os.path.join(outp_folder, pdb_id + ".pdb")

    with open(inp_path, "r") as in_file:
        lines = in_file.readlines()
    
    with open(outp_path, "w") as out_file:
        for line in lines:
            if "END" in line and len(line.strip()) == 3:
                line = line.replace("END", "")
                print(pdb_id)
            out_file.write(line)
