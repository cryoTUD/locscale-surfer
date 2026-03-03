
import os
import json
emmap_path = '/home/tnw-nb4020-03/bep_lotte_micelle/data/unsharpened_maps/'

emdb_list = []
for emmap in os.listdir(emmap_path):
    emdb_id = emmap.split("_")[1]
    emdb_list.append(emdb_id)

file_path = '/home/tnw-nb4020-03/bep_lotte_micelle/emdb_list.json'

with open(file_path, 'w') as output_file:
	json.dump(emdb_list, output_file)