import json
import os
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
from locscale.include.emmer.ndimage.map_tools import find_unmodelled_mask_region


# with open("/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/emdb_to_pdb.json", "r") as file:
#     emdb_to_pdb = json.load(file)

with open("/home/tnw-nb4020-03/dev/bep_lotte_micelle/emdb_pdb_tmp_dict.json", "r") as file:
    emdb_to_pdb = json.load(file)

# with open("/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/pdb_to_emdb.json", "r") as file:
#     pdb_to_emdb = json.load(file)


def find_unmodelled_region(atomic_path, fdr_mask_path, unsharpened_map_path, unmodelled_region_path):
    regions = find_unmodelled_mask_region(fdr_mask_path, atomic_path)
    _, apix = load_map(unsharpened_map_path)
    save_as_mrc(regions, unmodelled_region_path, apix)

def find_unmodelled_regions(atomic_folder, fdr_mask_folder, unsharpened_map_folder, unmodelled_region_folder, emdb_list):
    for emdb_id in emdb_list:
        print(emdb_id)
        pdb_id = emdb_to_pdb[emdb_id]

        atomic_path = os.path.join(atomic_folder, f'pdb_{pdb_id}.pdb')
        fdr_mask_path = os.path.join(fdr_mask_folder, f'emd_{emdb_id}_FDR_confidence_final_lp.map')
        unsharpened_map_path = os.path.join(unsharpened_map_folder, f'EMD_{emdb_id}_unsharpened_fullmap.mrc')
        unmodelled_region_path = os.path.join(unmodelled_region_folder, f'emd_{emdb_id}_unmodelled_region_lp.mrc')

        find_unmodelled_region(atomic_path, fdr_mask_path, unsharpened_map_path, unmodelled_region_path)

def main():
    # emdbs = ['0825', '11922', '13940', '14139', '14452', '14633', '14650', '14792', '15010', '21972', '23749', '27000', '27132', '27134']
    
    emdbs = [
    "28779", "22806", "11810", "21152", "12128", "28487", "27894", "33365",
    "14764", "28066", "26597", "26489", "30627", "0926", "13972",
    "33615", "11925", "27655", "25825", "28498", "25849", "0774",
    "40500", "21454", "31037", "33803", "28584", "28498", "40510", "12271",
    "0719", "14761"]

    atomic_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/atomic_models'
    fdr_mask_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/test_fdr_masks_lpf'
    unsharpened_map_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/unsharpened_maps'
    unmodelled_region_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/test_ums_lpf'
    emdb_list = emdbs

    find_unmodelled_regions(atomic_folder, fdr_mask_folder, unsharpened_map_folder, unmodelled_region_folder, emdb_list)

if __name__ == "__main__":
    main()