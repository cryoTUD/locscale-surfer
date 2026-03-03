import matplotlib.pyplot as plt
from locscale.include.emmer.ndimage.map_utils import load_map
from visualization_tools import project_map
import math
import json
import os
import glob

def project_folder(data_folder, data_name_format, emdb_list, figure_path, figure_name, projection_axis):
    '''project all images in the data set on a certain axis, for easy visualization'''
    rows = 5
    cols = math.ceil(len(emdb_list)/rows) # make sure all projections fit in the nr of subplots
    fig, ax = plt.subplots(rows,cols, dpi=300)

    for map_nr, emdb_id in enumerate(emdb_list):
        index_col = map_nr // rows
        index_row = map_nr % rows

        # make sure it works for any file format
        file_name    = data_name_format.replace('*', emdb_id)
        micelle_path = os.path.join(data_folder, file_name)
        micelle, _   = load_map(micelle_path)

        # project map
        projection = project_map(micelle, projection_axis, projection_type="mean")
        ax[index_row,index_col].imshow(projection)

        ax[index_row,index_col].get_xaxis().set_visible(False)
        ax[index_row,index_col].get_yaxis().set_visible(False)

        ax[index_row,index_col].set_title("{}".format(emdb_id), fontsize=8)

    # create empty spots when there are more spots in the figure than images
    for subplot_nr in range(len(emdb_list), rows*cols):
        index_col = subplot_nr // rows
        index_row = subplot_nr % rows
        ax[index_row,index_col].set_visible(False)

    # save the figure in a 
    save_fig_path = os.path.join(figure_path, figure_name)
    fig.savefig(save_fig_path)

def plot_lpf(projection_axis):
    micelle_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF'
    figure_path    = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/figures'
    # figure_name    = '*_projection_final_micelle2_selection_lpf.png'.replace('*', projection_axis)
    figure_name    = '*_projection_micelles_test_set_nr_lr.png'.replace('*', projection_axis)
    micelle_file_format = 'emd_*_micelle_lp.mrc'

    # with open("/home/tnw-nb4020-03/dev/bep_lotte_micelle/emdb_list.json", "r") as file:
    #     emdb_list = json.load(file)

    emdb_list = [
    "28779", "22806", "12128", "28487", "27894", "33365",
    "14764", "28066", "26597", "30627", "13972",
    "33615", "11925", "25825", "28498", "25849", "0774",
    "40500", "21454", "31037", "33803", "28584", "28498", "40510", "12271",
    "0719"]

    project_folder(micelle_folder, micelle_file_format, emdb_list, figure_path, figure_name, projection_axis)

def plot_no_lpf(projection_axis):
    micelle_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles_final_no_LPF'
    figure_path    = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/figures'
    figure_name    = '*_projection_final_micelle_selection_no_lpf.png'.replace('*', projection_axis)
    micelle_file_format = 'emd_*_micelle.mrc'

    with open("/home/tnw-nb4020-03/bep_lotte_micelle/emdb_list.json", "r") as file:
        emdb_list = json.load(file)

    project_folder(micelle_folder, micelle_file_format, emdb_list, figure_path, figure_name, projection_axis)


if __name__ == '__main__':
    micelle_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF'
    figure_path    = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/figures/projections'
    # figure_name    = '*_projection_test_set_nr_lr.png'.replace('*', projection_axis)
    # micelle_file_format = 'emd_*_micelle_lp.mrc'

    # emdb_list = ['4288', '10049', '20986', '3885', '4032', '9941', '4646', '9695', '0282']
    # emdb_list = ['0094', '4593', '20145', '9610', '20146', '7009', '4589', '10418', '0257', '10279', '7127', '0093', '0415', '7133', '8702', '4746', '8960', '4611', '4997', '4733', '9931', '9935', '4588', '0193', '9112', '7882', '4272', '8958', '9939', '0499', '9934', '4789']
    emdb_list = ['0093', '0094', '0193', '0234', '0257', '0282', '0415', '0499', '3885', '4032', '4272', '4272', '4288', '4588', '4589', '4593', '4611', '4646', '4733', '4746', '4789', '4997', '7009', '7127', '7133', '7882', '8702', '8958', '8960', '9112', '9610', '9695', '9931', '9934', '9935', '9939', '9941', '10049', '10279', '10418', '20145', '20146', '20849', '20986']
    # emdb_list = ['0825', '11922', '13940', '14139', '14452', '14633', '14650', '14792', '15010', '21972', '23749', '27000', '27132', '27134']
    # emdb_list = [   "28779", "22806", "11810", "21152", "12128", "28487", "27894", "33365",
    #                 "14764", "28066", "26597", "26489", "30627", "0926", "13972",
    #                 "33615", "11925", "27655", "25825", "28498", "25849", "0774",
    #                 "40500", "21454", "31037", "33803", "28584", "28498", "40510", "12271",
    #                 "0719", "14761"]

    micelle_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/unmodelled_regions/lpf_output'
    micelle_file_format = 'emd_*_unmodelled_region_lp.mrc'
    project_folder(micelle_folder, micelle_file_format, emdb_list, figure_path, 'ums_x_projection_train_set.png', 'x')
    project_folder(micelle_folder, micelle_file_format, emdb_list, figure_path, 'ums_z_projection_train_set.png', 'z')

    # plot_lpf('z')
    # plot_lpf('x')