import json
import os
import sys
from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
from find_micelles import extract_dummy_residues_from_pdb, find_number_of_membranes, get_membrane
from concurrent.futures import ProcessPoolExecutor

# with open("/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/emdb_to_pdb.json", "r") as file:
#     emdb_to_pdb = json.load(file)

with open("/home/tnw-nb4020-03/dev/bep_lotte_micelle/emdb_pdb_tmp_dict.json", "r") as file:
    emdb_to_pdb = json.load(file)


def segment_all_maps(emdb_list, pdb_folder, unmodelled_region_folder, unmodelled_region_format, dest_folder, dest_name_format): 
    '''parallelized version to significantly speedup the process'''     
    N = len(emdb_list)                         
    
    with ProcessPoolExecutor(max_workers=10) as pool:
        '''second and later arguments must be iterable'''
        pool.map(segmentation_routine, emdb_list, [pdb_folder]*N, [unmodelled_region_folder]*N, [unmodelled_region_format]*N, [dest_folder]*N, [dest_name_format]*N)
    # for emdb_id in emdb_list:
    #     segmentation_routine(emdb_id, pdb_folder, unmodelled_region_folder, unmodelled_region_format, dest_folder, dest_name_format)

def segmentation_routine(emdb_id, pdb_folder, unmodelled_region_folder, unmodelled_region_format, dest_folder, dest_name_format):
    '''procedure to be done for every image'''
    print(emdb_id)
    pdb_id    = emdb_to_pdb[emdb_id]
    # print("step 1")
    # open unmodelled region
    file_path = unmodelled_region_format.replace('*', emdb_id)
    unmodelled_region_path = os.path.join(unmodelled_region_folder, file_path)
    # print("step 2")
    unmodelled_region, apix = load_map(unmodelled_region_path)
    print(unmodelled_region.min(), unmodelled_region.max())
    print('finished opening unmodelled region')

    # get coordinates
    pdb_path = os.path.join(pdb_folder, f"pdb_{pdb_id}_aligned.pdb")
    N_coord, O_coord = extract_dummy_residues_from_pdb(pdb_path)

    # find number of membranes
    nr_membranes = find_number_of_membranes(pdb_path)
    print(nr_membranes)
    # get membrane
    imsize = len(unmodelled_region)
    membrane = get_membrane(N_coord, O_coord, unmodelled_region, apix, nr_membranes, imsize)
    print(membrane.min(), membrane.max())
    # select the unmodelled_region within the membrane region as the micelle
    selection = unmodelled_region * membrane

    # save the selection
    file_path = dest_name_format.replace('*', emdb_id)
    dest_path = os.path.join(dest_folder, file_path)
    save_as_mrc(selection, dest_path, apix)

    print(f'EMD-{emdb_id} done')

def segment_lpf():
    pdb_folder  = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/aligned_opm_models'
    unmodelled_region_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/unmodelled_regions/lpf_output'
    unmodelled_region_format = 'emd_*_unmodelled_region_lp.mrc'
    dest_folder      = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF'
    dest_name_format = 'emd_*_micelle_lp.mrc'

    with open("/home/tnw-nb4020-03/dev/bep_lotte_micelle/emdb_list.json", "r") as file:
        emdb_list = json.load(file)
    
    segment_all_maps(emdb_list, pdb_folder, unmodelled_region_folder, unmodelled_region_format, dest_folder, dest_name_format)

def segment_no_lpf():
    pdb_folder  = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/aligned_opm_models'
    unmodelled_region_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/unmodelled_regions/output'
    unmodelled_region_format = 'emd_*_unmodelled_region.mrc'
    dest_folder      = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles_final_no_LPF'
    dest_name_format = 'emd_*_micelle.mrc'

    with open("/home/tnw-nb4020-03/dev/bep_lotte_micelle/emdb_list.json", "r") as file:
        emdb_list = json.load(file)
    
    segment_all_maps(emdb_list, pdb_folder, unmodelled_region_folder, unmodelled_region_format, dest_folder, dest_name_format)


def segment_test():
    pdb_folder  = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/aligned_opm_models'
    unmodelled_region_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/test_ums_lpf'
    unmodelled_region_format = 'emd_*_unmodelled_region_lp.mrc'
    dest_folder      = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF'
    dest_name_format = 'emd_*_micelle_lp.mrc'

    # emdb_list = ['0825', '11922', '13940', '14139', '14452', '14633', '14650', '14792', '15010', '21972', '23749', '27000', '27132', '27134']
    
    # import json
    # with open('/home/tnw-nb4020-03/dev/bep_lotte_micelle/new_LR_test_set2.json') as file:
    #     LR_test = json.load(file)
    # with open('/home/tnw-nb4020-03/dev/bep_lotte_micelle/new_test_set2.json') as file:
    #     NR_test = json.load(file)

    # emdb_list =  list(LR_test.keys()) + list(NR_test.keys())
    
    emdb_list = [
    "28779", "22806", "11810", "21152", "12128", "28487", "27894", "33365",
    "14764", "28066", "26597", "26489", "30627", "0926", "13972",
    "33615", "11925", "27655", "25825", "28498", "25849", "0774",
    "40500", "21454", "31037", "33803", "28584", "28498", "40510", "12271",
    "0719", "14761"]
    emdb_list = ["11810", "21152", "26489", "0926", "27655", "14761"]

    segment_all_maps(emdb_list, pdb_folder, unmodelled_region_folder, unmodelled_region_format, dest_folder, dest_name_format)

if __name__ == '__main__':
    # segment_no_lpf()

    # emdb_id = '11922'
    # emdb_list = ['13940', '14139', '14452', '14633', '14650', '14792', '15010', '21972', '23749', '27000', '27132', '27134']
    
    # pdb_folder  = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/aligned_opm_models'
    # unmodelled_region_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/unmodelled_regions/lpf_output'
    # unmodelled_region_format = 'emd_*_unmodelled_region_lp.mrc'
    # dest_folder      = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF'
    # dest_name_format = 'emd_*_micelle_lp.mrc'

    # for emdb_id in emdb_list:
    #     segmentation_routine(emdb_id, pdb_folder, unmodelled_region_folder, unmodelled_region_format, dest_folder, dest_name_format)

    segment_test()

    # pdb_folder  = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/aligned_opm_models'
    # unmodelled_region_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/unmodelled_regions/lpf_output'
    # unmodelled_region_format = 'emd_*_unmodelled_region_lp.mrc'
    # dest_folder      = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF'
    # dest_name_format = 'emd_*_micelle_lp.mrc'

    # emdb_id = '9610'
    # segmentation_routine(emdb_id, pdb_folder, unmodelled_region_folder, unmodelled_region_format, dest_folder, dest_name_format)