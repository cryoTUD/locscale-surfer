######################################################## IMPORTS ##################################################################

# external imports
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import shutil
import atexit
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import shuffle
from tensorboard.plugins.hparams import api as hp
from scipy.ndimage import laplace
# internal imports
# Add the path to the sys.path
import sys
sys.path.append('/home/abharadwaj1/soft/emmernet/scripts')
from EMmerNet_utils import *
from EMmerNet_models import define_model, define_model_regularized, define_model_large, define_model_dropout, define_model_two_channel
import random
### set random seed for reproducibility
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42) 
# Use deterministic convolution for reproducibility
#os.environ['TF_DETERMINISTIC_OPS'] = '1'

########################################################### ARG PARSER ###################################################################
parser = argparse.ArgumentParser(description="produces neural network sharpened cryo-EM maps, trained on LocScale sharpened maps")

## MACRO VARIABLES
# type of run
parser.add_argument("-run", "--run_configuration", nargs='+', help="run configuration, options: 'data_preparation' or 'neural_network' or both, this argument is required", default=None, required=False)

# directory names
parser.add_argument("--training_targets_json", "-training_targets_json", type=str, help="Path to json file with input and training targets", required=False, default="/home/abharadwaj1/dev/map_sharpening/emmernet/emmernet_training/cubedata_modelangelo_dataset/cubedata_directory/cubedata_training/training_emdb_ids_random_sample.json")
parser.add_argument("--num_maps_training", "-num_maps_training", type=int, help="Number of maps to use for training", default=None)
parser.add_argument("--num_maps_validation", "-num_maps_validation", type=int, help="Number of maps to use for validation", default=None)

## DATASETS CHARACTERISTICS
# mapdata selection
parser.add_argument("-tri", "--training_ids", nargs='+', help="maps selected for training, format: 'EMDBid_PDBid_first' ... 'EMDBid_PDBid_last' defaults to standard training dataset", default=EMDB_PDB_ids_training)
parser.add_argument("-vai", "--validation_ids", nargs='+', help="maps selected for validation, format: 'EMDBid_PDBid_first' ... 'EMDBid_PDBid_last', defaults to standard validation dataset", default=EMDB_PDB_ids_validation)
parser.add_argument("-tei", "--test_ids", nargs='+', help="maps selected for test, format: 'EMDBid_PDBid_first' ... 'EMDBid_PDBid_last', defaults to standard test dataset", default=EMDB_PDB_ids_test)

# basics
parser.add_argument("-cz", "--cube_size", type=int, help="size of map cubes, options: '64', '32' or '16', defaults to 32", default=32)

## HYPERPARAMETERS: DATA PREPARATION
# basics
parser.add_argument("-pntr", "--percent_noise_cubes_trainval", type=float, help="percent of noise cubes that are selected for the training and validation datasets, range: [0, 1], defaults to 0.02", default=0.02)
parser.add_argument("-num_cubes_training", "--num_cubes_training", type=int, help="Number of cubes to use for training", default=None)
parser.add_argument("-num_cubes_validation", "--num_cubes_validation", type=int, help="Number of cubes to use for validation", default=None)

## HYPERPARAMETERS: NEURAL NETWORK
# basics: train and test
parser.add_argument("-mn", "--model_name", type=str, help="model name, format: [model_][name]", required=False)
parser.add_argument("-a", "--append_text", type=str, help="Append text for model name", default=None)
parser.add_argument("-rt", "--run_type", type=str, help="run type, options: 'train', 'train_test', 'test', 'test_custom', 'test_predict', defaults to 'train'", default='train')
parser.add_argument("-st", "--start_type", type=str, help="start type, options: 'from_scratch', 'continue', 'specific', defaults to 'None'", default=None)
parser.add_argument("-le", "--load_epoch", type=int, help="load_epoch, defaults to 'None'", default=None)
parser.add_argument("-ne", "--num_epochs", type=int, help="number of epochs, defaults to '15'", default=15)
parser.add_argument("-no_aug", "--no_augmentation", action='store_true', help="no augmentation, defaults to 'False'", default=False)
parser.add_argument("-use_dropout", "--use_dropout", action='store_true', help="use dropout, defaults to 'False'", default=False)
# basics: train
parser.add_argument("-bs", "--batch_size", type=int, help="batch size, defaults to '8'", default=8)
parser.add_argument("-lr", "--nn_learning_rate", type=float, help="learning rate parameter, defaults to '0.001'", default=0.001)
parser.add_argument("-ld", "--nn_learning_rate_dev_name", type=str, help="type of learning rate development, options: 'constant' or 'reduce'", default="constant")
parser.add_argument("-op", "--nn_optimizer_name", type=str, help="type of optimization algorithm, options: 'SGD' or 'Adam', defaults to 'Adam'", default="Adam")
parser.add_argument("-lo", "--nn_loss_name", type=str, help="type of loss functions, options: 'MAE', 'MAE_phase' or 'MSE', defaults to 'MAE'", default="MAE")
parser.add_argument("-me", "--nn_metric_name", type=str, help="type of trainig and validation metric, options: 'MAE' or 'MSE', defaults to 'MSE'", default="MSE")
parser.add_argument("-nn_l1_reg", "--nn_l1_reg", type=float, help="L1 regularization parameter, defaults to 'None'", default=None)
parser.add_argument("-nn_l2_reg", "--nn_l2_reg", type=float, help="L2 regularization parameter, defaults to 'None'", default=None)
parser.add_argument("-training_cube_size","--training_cube_size", type=int, help="Length of training cubes", default=60000)
parser.add_argument("--use_physics_based_loss", action='store_true', help="Use physics based loss", default=False)
parser.add_argument("--run_emmernet_test_after_epoch", action='store_true', help="Run emmernet test after each epoch", default=False)
parser.add_argument("--segmentation_network", action='store_true', help="Use segmentation network", default=False)

# GPUs
parser.add_argument("-gpus", "--GPU_nums", nargs='+', help="numbers of the selected GPUs, format: '1 2 3'", default=[None])

# CPUs
parser.add_argument("-np", "--num_processes", type=int, help="number of processes, defaults to '10'", default=10)

################################################## SET RUN SPECIFIC VARIABLES ############################################################


def set_variables(args):
    
    ## MACRO VARIABLES
    # type of run
    global run_configuration; run_configuration = args.run_configuration
    
    ## DATASETS CHARACTERISTICS
    # mapdata selection
    # global emdb_pdb_training_id; emdb_pdb_training_id = args.training_ids
    # global emdb_pdb_validation_id; emdb_pdb_validation_id = args.validation_ids
    # global emdb_pdb_test_id; emdb_pdb_test_id = args.test_ids

    # cubedata names
    
    # basics
    global cube_size; cube_size = args.cube_size
    
    ## HYPERPARAMETERS: DATA PREPARATION
    # basics
    global step_size_trainval; step_size_trainval = int(cube_size / 4 * 3)
    global step_size_test; step_size_test = int(cube_size / 4 * 3)
    global percent_noise_cubes_trainval; percent_noise_cubes_trainval = args.percent_noise_cubes_trainval
    global percent_noise_cubes_test; percent_noise_cubes_test = 1
    global max_sample_size_training; max_sample_size_training = args.training_cube_size
    global max_sample_size_validation; max_sample_size_validation = int(max_sample_size_training / 6)
    global max_cubes_training; max_cubes_training = args.num_cubes_training
    global max_cubes_validation; max_cubes_validation = args.num_cubes_validation
    global num_maps_training; num_maps_training = args.num_maps_training
    global num_maps_validation; num_maps_validation = args.num_maps_validation

    
    ## HYPERPARAMETERS: NEURAL NETWORK
    # basics
    global model_name; model_name = args.model_name
    global append_text; 
    if args.append_text is not None:
        append_text = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.append_text
    else:
        append_text = datetime.now().strftime("%Y%m%d_%H%M%S")
    global run_type; run_type = args.run_type
    global start_type; start_type = args.start_type
    global load_epoch; load_epoch = args.load_epoch
    global num_epochs; num_epochs = args.num_epochs
    global batch_size; batch_size = args.batch_size
    global nn_learning_rate; nn_learning_rate = args.nn_learning_rate
    global nn_learning_rate_dev_name; nn_learning_rate_dev_name = args.nn_learning_rate_dev_name
    global nn_optimizer_name; nn_optimizer_name = args.nn_optimizer_name
    global nn_loss_name; nn_loss_name = args.nn_loss_name
    global nn_loss
    global nn_metric_name; nn_metric_name = args.nn_metric_name
    global nn_l1_reg; nn_l1_reg = args.nn_l1_reg
    global nn_l2_reg; nn_l2_reg = args.nn_l2_reg
    global no_augmentation; no_augmentation = args.no_augmentation
    global use_dropout; use_dropout = args.use_dropout
    global use_physics_based_loss; use_physics_based_loss = args.use_physics_based_loss
    global run_emmernet_test_after_epoch; run_emmernet_test_after_epoch = args.run_emmernet_test_after_epoch
    global SEGMENTATION_NETWORK; SEGMENTATION_NETWORK = args.segmentation_network

    # GPUs
    global GPU_nums; GPU_nums = ""
    global gpu_id_input; gpu_id_input = args.GPU_nums
    global GPU_names; GPU_names = []
    GPU_nums_length = len(args.GPU_nums)
    for i in np.arange(GPU_nums_length):
        GPU_num = args.GPU_nums[i]
        if i < (GPU_nums_length - 1):
            GPU_nums += (str(GPU_num) + ",")
        elif i == (GPU_nums_length - 1):
            GPU_nums += str(GPU_num)
        GPU_names.append(("/gpu:"+ str(GPU_num)))
    global num_processes; num_processes = args.num_processes
    ####################################################### GENERAL DIRS ##################################################################
    
    global parent_data_dir; parent_data_dir = "/home/abharadwaj1/scratch/dev/"
    global model_data_dir; model_data_dir = os.path.join("/tmp", model_name)

    if not os.path.exists(model_data_dir):
        print("Creating model data directory: {}".format(model_data_dir))
        os.makedirs(model_data_dir)
    
    global training_targets_json; training_targets_json = os.path.abspath(args.training_targets_json)

    #assert os.path.exists(training_targets_json), "training_targets_json does not exist: {}".format(training_targets_json)
    
    global collection_data_dir; collection_data_dir = os.path.join(model_data_dir, "collection_directory")
    if not os.path.exists(collection_data_dir):
        print("Creating collection data directory: {}".format(collection_data_dir))
        os.makedirs(collection_data_dir)
    
    global cubedata_dir; cubedata_dir = os.path.join(model_data_dir, "cubedata_directory")
    if not os.path.exists(cubedata_dir):
        print("Creating cubedata directory: {}".format(cubedata_dir))
        os.makedirs(cubedata_dir)
    
    global cubedata_random_cubes_dir; cubedata_random_cubes_dir = os.path.join(cubedata_dir, "cubedata_random_cubes")
    if not os.path.exists(cubedata_random_cubes_dir):
        print("Creating cubedata random cubes directory: {}".format(cubedata_random_cubes_dir))
        os.makedirs(cubedata_random_cubes_dir)
    
    global cubedata_training_dir; cubedata_training_dir = os.path.join(cubedata_dir, "cubedata_training")
    if not os.path.exists(cubedata_training_dir):
        print("Creating cubedata training directory: {}".format(cubedata_training_dir))
        os.makedirs(cubedata_training_dir)

    global cubedata_validation_dir; cubedata_validation_dir = os.path.join(cubedata_dir, "cubedata_validation")
    if not os.path.exists(cubedata_validation_dir):
        print("Creating cubedata validation directory: {}".format(cubedata_validation_dir))
        os.makedirs(cubedata_validation_dir)
    

    # outputs
    global outputdata_dir; outputdata_dir = os.path.join(parent_data_dir, model_name, "outputdata")
    if not os.path.exists(outputdata_dir):
        print("Creating outputdata directory: {}".format(outputdata_dir))
        os.makedirs(outputdata_dir)
    
    
    ########################################################## SUB DIRS ####################################################################
    
    # cubedata


    # output data
    if model_name != None:
        global model_name_dir; model_name_dir = os.path.join(outputdata_dir, append_text)
        global saved_models_dir; saved_models_dir = os.path.join(model_name_dir, "saved_models")
        if not os.path.exists(saved_models_dir):
            print("Creating saved models directory: {}".format(saved_models_dir))
            os.makedirs(saved_models_dir)
        global training_performance_dir; training_performance_dir = os.path.join(model_name_dir, "training_performance")
        if not os.path.exists(training_performance_dir):
            print("Creating training performance directory: {}".format(training_performance_dir))
            os.makedirs(training_performance_dir)

    ######################################################### DECISION TREES #################################################################
    
    # nn_optimizer decision tree
    global nn_optimizer
    if nn_optimizer_name == "Adam":
        nn_optimizer = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    elif nn_optimizer_name == "SGD":
        nn_momentum = 0.0
        nn_optimizer = tf.keras.optimizers.SGD(learning_rate=nn_learning_rate, momentum=nn_momentum)
    elif nn_optimizer_name == "RMSprop":
        nn_momentum = 0.0
        nn_optimizer = tf.keras.optimizers.RMSprop(learning_rate=nn_learning_rate, rho=0.9, momentum=nn_momentum, epsilon=1e-07, centered=False, name='RMSprop')
    elif nn_optimizer_name == "Adamax":
        nn_optimizer = tf.keras.optimizers.Adamax(learning_rate=nn_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')
    elif nn_optimizer_name == "Adadelta":
        nn_optimizer = tf.keras.optimizers.Adadelta(learning_rate=nn_learning_rate, rho=0.95, epsilon=1e-07, name='Adadelta')
    elif nn_optimizer_name == "Adagrad":
        nn_optimizer = tf.keras.optimizers.Adagrad(learning_rate=nn_learning_rate, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad')
    elif nn_optimizer_name == "Ftrl":
        nn_optimizer = tf.keras.optimizers.Ftrl(learning_rate=nn_learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0,
                                                l2_regularization_strength=0.0, name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
    elif nn_optimizer_name == "Nadam":
        nn_optimizer = tf.keras.optimizers.Nadam(learning_rate=nn_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
    else:
        print("ERROR: please specify 'nn_optimizer' as 'SGD' or 'Adam'")
        
    # nn_loss decision tree
    
    class reducePhaseDifference():
        """ custom loss function that reduces phase difference loss
        """

        def __init__(self):
            pass


        def reduce_phase_diff_loss(self, y_pred, y_true):

            import sys

            # compute the mean squared error
            mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(y_pred, y_true)))

            # flatten 3d tensor to 1d tensor
            y_pred_flat = tf.reshape(y_pred, [-1])
            y_true_flat = tf.reshape(y_true, [-1])
            
            # compute the phase difference
            y_pred_fft = tf.signal.rfft(y_pred_flat)
            y_true_fft = tf.signal.rfft(y_true_flat)

            y_pred_angle = tf.math.angle(y_pred_fft)
            y_true_angle = tf.math.angle(y_true_fft)

            phase_diff = tf.math.subtract(y_pred_angle, y_true_angle)
            phase_diff = tf.math.abs(phase_diff)
            max_phase_diff = tf.math.reduce_max(phase_diff)
            phase_diff = tf.math.reduce_mean(phase_diff)

            # compute the amplitude difference
            y_pred_amp = tf.math.abs(y_pred_fft)
            y_true_amp = tf.math.abs(y_true_fft)

            amp_diff = tf.math.subtract(y_pred_amp, y_true_amp)
            amp_diff = tf.math.abs(amp_diff)
            amp_diff = tf.math.reduce_mean(amp_diff)

            # compute the loss
            mode = "both"

            if mode == "only_phase_diff":
                loss = phase_diff
            elif mode == "only_mae":
                loss = mae
            elif mode == "both":
                loss = tf.math.add(mae, tf.math.multiply(phase_diff, 100))
            elif mode == "amp_phase":
                loss = tf.math.add(tf.math.multiply(amp_diff, 0.1), tf.math.multiply(phase_diff, 100))    

            # print
            tf.print("MAE: ", mae, "-- amp_diff: ", amp_diff, "-- phase diff: ", phase_diff, "-- max phase diff: ", max_phase_diff, " -- total loss: ", loss, output_stream=sys.stdout)

            return loss

        def __call__(self, y_pred, y_true):
            return self.reduce_phase_diff_loss(y_pred, y_true)

    
    if nn_loss_name == "MAE":
        nn_loss = tf.keras.losses.MeanAbsoluteError()
    elif nn_loss_name == "MSE":
        nn_loss = tf.keras.losses.MeanSquaredError()
    elif nn_loss_name == "MAE_phase":
        nn_loss = reducePhaseDifference()
    else:
        print("ERROR: please specify 'nn_loss' as 'MAE' or 'MSE'")
        
    # nn_metric decision tree
    global nn_metric
    if nn_metric_name == "MAE":
        nn_metric = ['mae']
    elif nn_metric_name == "MSE":
        nn_metric = ['mse']
    else:
        print("ERROR: please specify 'nn_metric' as 'MAE' or 'MSE'")

######################################################## FUNCTIONS ##################################################################

##################################################### PRINT FUNCTIONS ###############################################################

def print_hyperparameters():
    """ prints all hyperparameters to user
    """ 
    import yaml 
    import json 
    import pickle
    print("Hyperparameters:")
    XY_filenames_training_dataset_pickle = os.path.join(cubedata_training_dir, "XY_filenames_dataset.pickle")
    XY_filenames_validation_dataset_pickle = os.path.join(cubedata_validation_dir, "XY_filenames_dataset.pickle")

    # cube_filenames_training_X_dic = pd.read_json(X_filenames_training_dataset_json).to_dict(orient="index")
    # cube_filenames_training_Y_dic = pd.read_json(Y_filenames_training_dataset_json).to_dict(orient="index")
    # cube_filenames_validation_X_dic = pd.read_json(X_filenames_validation_dataset_json).to_dict(orient="index")
    # cube_filenames_validation_Y_dic = pd.read_json(Y_filenames_validation_dataset_json).to_dict(orient="index")
    
    # with open(XY_filenames_training_dataset_pickle, "rb") as f:
    #     XY_filenames_dataset_training = pickle.load(f)
    # with open(XY_filenames_validation_dataset_pickle, "rb") as f:
    #     XY_filenames_dataset_validation = pickle.load(f)

    emdb_ids_used_for_training = [x for x in os.listdir(cubedata_training_dir) if os.path.isdir(os.path.join(cubedata_training_dir, x))]
    emdb_ids_used_for_validation = [x for x in os.listdir(cubedata_validation_dir) if os.path.isdir(os.path.join(cubedata_validation_dir, x))]
    
    
    hyperparameters_dictionary = {
        "run_configuration": run_configuration,
        "training_targets_json" : training_targets_json,
        "append_text" : append_text,
        "model_name_dir" : model_name_dir,
        "cubedata_directory" : cubedata_dir,
        "dataset_characteristics": {
            "training_id": emdb_ids_used_for_training,
            "validation_id": emdb_ids_used_for_validation,
        },
        # "mapdata_sizes": {
        #     "training_size": len(emdb_pdb_training_id),
        #     "validation_size": len(emdb_pdb_validation_id),
        #     "test_size": len(emdb_pdb_test_id),
        # },
        "basics": {
            "cube_size": cube_size,
        },
        "data_preparation": {
            "step_size_trainval": step_size_trainval,
            "step_size_test": step_size_test,
            "percent_noise_cubes_trainval": percent_noise_cubes_trainval,
            "percent_noise_cubes_test": percent_noise_cubes_test,   
        },
        "neural_network": {
            "model_name": model_name,
            "run_type": run_type,
            "start_type": start_type,
            "load_epoch": load_epoch,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": nn_learning_rate,
            "learning_rate_dev_name": nn_learning_rate_dev_name,
            "optimizer_name": nn_optimizer_name,
            "loss_name": nn_loss_name,
            "metric_name": nn_metric_name,
            "GPU_nums": GPU_nums,
            "GPU_names": GPU_names,
        },
    }
    
    print(yaml.dump(hyperparameters_dictionary, default_flow_style=False))
    # dump hyperparameters to json file
    json_file_hyperparameters = os.path.join(model_name_dir, model_name+"_"+"hyperparameters.json")
    with open(json_file_hyperparameters, 'w') as fp:
        json.dump(hyperparameters_dictionary, fp, indent=4)
    
    return hyperparameters_dictionary
    
    
def save_training_info():
    """ writes all training and validation (hyper)parameters to text file in training performance directory
    """
    
    text_file_path = os.path.join(training_performance_dir, model_name+"_"+"training_info.txt")
    f = open(text_file_path, "w")
    f.write("################# TRAINING INFO ################")
    f.write("\n####### HYPERPARAMETERS: DATASETS CHARACTERISTICS #######")
    f.write("\n### mapdata selection")
    # f.write("\n- training set = {}".format(emdb_pdb_training_id))
    # f.write("\n- validation set = {}".format(emdb_pdb_validation_id))
    f.write("\n### mapdata sizes")
    # f.write("\n- training set size = {}".format(len(emdb_pdb_training_id)))
    # f.write("\n- validation set size = {}".format(len(emdb_pdb_validation_id)))
    f.write("\n### cubedata names")
    f.write("\n### cubedata sizes")
    f.write("\n### basics")
    f.write("\n- cube size = {}".format(cube_size))
    f.write("\n####### HYPERPARAMETERS: DATA PREPARATION #######")
    f.write("\n### basics")
    f.write("\n- step size training / validation = {}".format(step_size_trainval))
    f.write("\n- percent noise trainval: {}".format(percent_noise_cubes_trainval))
    f.write("\n######## HYPERPARAMETERS: NEURAL NETWORK ########")
    f.write("\n### basics")
    f.write("\n- model name = {}".format(model_name))
    f.write("\n- run type: {}".format(run_type))
    f.write("\n- start type: {}".format(start_type))
    f.write("\n- load epoch: {}".format(load_epoch))
    f.write("\n- number of epochs: {}".format(num_epochs))
    f.write("\n- batch size = {}".format(batch_size))
    f.write("\n- learning rate: {}".format(nn_learning_rate))
    f.write("\n- learning rate development: {}".format(nn_learning_rate_dev_name))
    f.write("\n- optimizer: {}".format(nn_optimizer_name))
    f.write("\n- loss function: {}".format(nn_loss_name))
    f.write("\n### GPUs")
    f.write("\n- GPU numbers: {}".format(GPU_nums))
    f.write("\n- GPU names: {}".format(GPU_names))
    f.close()
    

def save_cubedata_info(text_file_path, dataset_name, EMDB_PDB_ids, step_size, percent_noise_cubes, num_cubes_X):
    """ writes cubedata (hyper)parameters to text file in cubedata info dir
    """
    
    f = open(text_file_path, "w")
    f.write("############# PARAMETERS: CUBEDATA ##############")    
    f.write("\ncube_size: {}".format(cube_size))
    f.write("\nstep size: {}".format(step_size))
    f.write("\npercent noise cubes: {}".format(percent_noise_cubes))
    f.write("\nmapdata selection: {}".format(EMDB_PDB_ids))
    f.write("\nmapdata size: {}".format(len(EMDB_PDB_ids)))
    f.write("\ncubedata name: {}".format(dataset_name))
    f.write("\ncubedata X cubes: {}".format(num_cubes_X))
    f.close()

################################################### LOW LEVEL FUNCTIONS #############################################################

def run_emmernet_test(emmernet_model, epoch_num, model_save_path):
    import os
    import shutil
    import subprocess
    from locscale.automate.tools import get_defaults_dictionary
    from locscale.emmernet.run_emmernet import run_emmernet_batch, \
        start_preprocessing_data, prepare_inputs_for_network, \
        predict_cubes_and_assemble, assemble_cubes_in_right_place
    from locscale.include.emmer.ndimage.map_utils import load_map, save_as_mrc
    import argparse
    emmernet_inputs = get_defaults_dictionary("emmernet")
    
    emmap_path = "/home/abharadwaj1/dev/map_sharpening/emmernet/emmernet_training/test_data/EMD_3061_unfiltered.mrc"
    mask_path = "/home/abharadwaj1/dev/map_sharpening/emmernet/emmernet_training/test_data/EMD_3061_unfiltered_confidenceMap.mrc"
    model_folder = os.path.dirname(model_save_path)
    test_map_folder = os.path.join(model_folder, "test_epochs") 
    if not os.path.exists(test_map_folder):
        os.makedirs(test_map_folder)
    epoch_folder = os.path.join(test_map_folder, "epoch_{}".format(epoch_num))
    if not os.path.exists(epoch_folder):
        os.makedirs(epoch_folder)
        
    copied_emmap_path = os.path.join(epoch_folder, "EMD_3061_unfiltered.mrc")
    copied_mask_path = os.path.join(epoch_folder, "EMD_3061_unfiltered_confidenceMap.mrc")
    output_file_path = os.path.join(epoch_folder, "emd_3061_epoch_{}_emmernet_output.mrc".format(epoch_num))
    if not os.path.exists(copied_emmap_path):
        copied_emmap_path = shutil.copyfile(emmap_path, copied_emmap_path)
    
    if not os.path.exists(copied_mask_path):
        copied_mask_path = shutil.copyfile(mask_path, copied_mask_path)
    emmap, apix  = load_map(copied_emmap_path)
    
    emmernet_inputs["emmap_path"] = copied_emmap_path
    emmernet_inputs["emmap_folder"] = epoch_folder
    emmernet_inputs["apix"] = apix
    emmernet_inputs["xyz_mask_path"] = copied_mask_path
    emmernet_inputs["verbose"] = True
    emmernet_inputs["outfile"] = output_file_path
    emmernet_inputs["gpu_ids"] = gpu_id_input
    emmernet_inputs["monte_carlo"] = True
    
    try: 
        print("Running emmernet for epoch {}".format(epoch_num))
        emmernet_inputs = start_preprocessing_data(emmernet_inputs)
        print("Preprocessing done")
        emmernet_inputs = prepare_inputs_for_network(emmernet_inputs)
        print("Inputs prepared")
        emmernet_inputs["emmernet_model"] = emmernet_model
        emmernet_inputs["cuda_visible_devices_string"] = ""
        input_dictionary = run_emmernet_batch(emmernet_inputs, emmernet_model, "cpu")
        predicted_map_mean = assemble_cubes_in_right_place(input_dictionary, input_dictionary["cubes_predicted_mean"])
        predicted_map_var = assemble_cubes_in_right_place(input_dictionary, input_dictionary["cubes_predicted_var"])
        predicted_map_total = assemble_cubes_in_right_place(input_dictionary, input_dictionary["cubes_predicted_total"])
        
        outfile_mean = os.path.join(epoch_folder, "emd_3061_epoch_{}_emmernet_output_mean.mrc".format(epoch_num))
        outfile_var = os.path.join(epoch_folder, "emd_3061_epoch_{}_emmernet_output_var.mrc".format(epoch_num))
        save_as_mrc(predicted_map_mean, outfile_mean, apix)
        save_as_mrc(predicted_map_var, outfile_var, apix)
        save_as_mrc(predicted_map_total, output_file_path, apix)
        print("EMmerNet test finished successfully")
    except Exception as e:
        print("EMmerNet test failed")
        print(e)
        raise e

class save_weights_on_epoch(tf.keras.callbacks.Callback):

    def __init__(self):
        super(save_weights_on_epoch, self).__init__()
        

    def on_epoch_end(self, epoch, logs=None):
    
        # self.model.save_weights(os.path.join(saved_models_dir, "{}_epoch-{}".format(model_name, str(epoch))))
        # save model checkpoint as hdf5 file
        model_save_path = os.path.join(saved_models_dir, "{}_epoch-{}.hdf5".format(model_name, str(epoch)))
        self.model.save(model_save_path)
        
        # run EMmerNet test with current model on a test map 
        if run_emmernet_test_after_epoch:
            run_emmernet_test(self.model, epoch, model_save_path)

def get_cube_statistics(cube_filenames):
    number_of_sample_cubes = 100
    sample_cube_filenames = np.random.choice(cube_filenames, number_of_sample_cubes, replace=False)
    cubes = [np.reshape(np.load(cube_filename), (cube_size, cube_size, cube_size)) for cube_filename in sample_cube_filenames]
    statistics = {}
    statistics["means"] = [x.mean() for x in cubes]
    statistics["stds"] = [x.std() for x in cubes]
    statistics["mins"] = [x.min() for x in cubes]
    statistics["maxs"] = [x.max() for x in cubes]
    return statistics



# def create_datagenerators(cubedata_top_directory):
#     """ creates training and validation datagenerator objects

#     Returns:
#         training_data_generator (Custom_Datagenerator): training data generator object
#         validation_data_generator (Custom_Datagenerator): validation data generator object
#     """
#     import pandas as pd
#     import pickle 

#     print("\n>>> CREATE DATAGENERATORS")
    
#     class Custom_Datagenerator(tf.keras.utils.Sequence):
  
#         def __init__(self, cube_filenames_X, cube_filenames_Y, batch_size) :
#             self.cube_filenames_X = cube_filenames_X  
#             self.cube_filenames_Y = cube_filenames_Y 
#             self.batch_size = batch_size
#             #self.cubes_dir = cubes_dir
            
#         def __len__(self) :
#             return (np.ceil(len(self.cube_filenames_X) / float(self.batch_size))).astype(np.int)
        
#         def __getitem__(self, idx) :
#             batch_cube_filenames_X = self.cube_filenames_X[idx * self.batch_size : (idx+1) * self.batch_size]
#             batch_cube_filenames_Y = self.cube_filenames_Y[idx * self.batch_size : (idx+1) * self.batch_size]
            
#             X_data = np.empty((batch_size, cube_size, cube_size, cube_size, 1))
#             Y_data = np.empty((batch_size, cube_size, cube_size, cube_size, 1))
            
#             i = 0
#             for filename in batch_cube_filenames_X:
#                 tempcube = np.load(filename)
#                 X_data[i,:,:,:,:] = tempcube
#                 i += 1
            
#             j = 0
#             for filename in batch_cube_filenames_Y:
#                 tempcube = np.load(filename)
#                 Y_data[j,:,:,:,:] = tempcube
#                 j += 1
            
#             return X_data, Y_data
#     cubedata_directory_training = os.path.join(cubedata_top_directory, "cubedata_training")
#     cubedata_directory_validation = os.path.join(cubedata_top_directory, "cubedata_validation")
    
#     X_filenames_training_dataset_json = os.path.join(cubedata_directory_training, "X_filenames_dataset.json")
#     Y_filenames_training_dataset_json = os.path.join(cubedata_directory_training, "Y_filenames_dataset.json")
#     X_filenames_validation_dataset_json = os.path.join(cubedata_directory_validation, "X_filenames_dataset.json")
#     Y_filenames_validation_dataset_json = os.path.join(cubedata_directory_validation, "Y_filenames_dataset.json")
#     XY_filenames_training_dataset_pickle = os.path.join(cubedata_directory_training, "XY_filenames_dataset.pickle")
#     XY_filenames_validation_dataset_pickle = os.path.join(cubedata_directory_validation, "XY_filenames_dataset.pickle")

#     # cube_filenames_training_X_dic = pd.read_json(X_filenames_training_dataset_json).to_dict(orient="index")
#     # cube_filenames_training_Y_dic = pd.read_json(Y_filenames_training_dataset_json).to_dict(orient="index")
#     # cube_filenames_validation_X_dic = pd.read_json(X_filenames_validation_dataset_json).to_dict(orient="index")
#     # cube_filenames_validation_Y_dic = pd.read_json(Y_filenames_validation_dataset_json).to_dict(orient="index")
    
#     with open(XY_filenames_training_dataset_pickle, "rb") as f:
#         XY_filenames_dataset_training = pickle.load(f)
#     with open(XY_filenames_validation_dataset_pickle, "rb") as f:
#         XY_filenames_dataset_validation = pickle.load(f)

#     emdb_ids_used_for_training = list(set([get_emdb_id_from_cube_path(x[0]) for x in XY_filenames_dataset_training]))
#     emdb_ids_used_for_validation = list(set([get_emdb_id_from_cube_path(x[0]) for x in XY_filenames_dataset_validation]))
    
#     print("EMDB IDs used for training:")
#     for emdb_id in emdb_ids_used_for_training:
#         print("EMDB ID: {}".format(emdb_id))
    
#     print("EMDB IDs used for validation:")
#     for emdb_id in emdb_ids_used_for_validation:
#         print("EMDB ID: {}".format(emdb_id))
        
#     if no_augmentation:
#         print("Removing cubes with B-factor augmentation")
#         print("Number of training cubes before removing: {}".format(len(XY_filenames_dataset_training)))
#         print("Number of validation cubes before removing: {}".format(len(XY_filenames_dataset_validation)))
#         XY_filenames_dataset_training = remove_augmented_cubes(XY_filenames_dataset_training)
#         XY_filenames_dataset_validation = remove_augmented_cubes(XY_filenames_dataset_validation)
#         print("Number of training cubes after removing: {}".format(len(XY_filenames_dataset_training)))
#         print("Number of validation cubes after removing: {}".format(len(XY_filenames_dataset_validation)))
#         print("=============================================")
        
        
#     # Print the number of training and validation samples and the type of data
#     print("Number of training samples: {}".format(len(XY_filenames_dataset_training)))
#     print("Number of validation samples: {}".format(len(XY_filenames_dataset_validation)))
#     print("Type of data train: {}".format(type(XY_filenames_dataset_training)))
#     print("Type of data validation: {}".format(type(XY_filenames_dataset_validation)))

    
#     MAX_SAMPLE_SIZE_TRAINING = max_sample_size_training  if len(XY_filenames_dataset_training) > max_sample_size_training\
#                                                                 else len(XY_filenames_dataset_training)
#     MAX_SAMPLE_SIZE_VALIDATION = max_sample_size_validation if len(XY_filenames_dataset_validation) > max_sample_size_validation\
#                                                                 else len(XY_filenames_dataset_validation)

#     # print one example of the training data
#     print("Example of training data: {}".format(XY_filenames_dataset_training[0]))
#     # print one example of the validation data
#     print("Example of validation data: {}".format(XY_filenames_dataset_validation[0]))

#     # Randomly shuffle the training and validation data
#     p = np.random.permutation(len(XY_filenames_dataset_training))
#     # Choose a subset of the training data

#     XY_filenames_dataset_training_sampled = [XY_filenames_dataset_training[i] for i in p[:MAX_SAMPLE_SIZE_TRAINING]]
#     q = np.random.permutation(len(XY_filenames_dataset_validation))
#     XY_filenames_dataset_validation_sampled = [XY_filenames_dataset_validation[i] for i in q[:MAX_SAMPLE_SIZE_VALIDATION]]

#     print("Number of training samples: {}".format(len(XY_filenames_dataset_training_sampled)))
#     print("Number of validation samples: {}".format(len(XY_filenames_dataset_validation_sampled)))
#     print("Example of training data: {}".format(XY_filenames_dataset_training_sampled[0]))
#     print("Example of validation data: {}".format(XY_filenames_dataset_validation_sampled[0]))

#     cube_filenames_training_X = np.array([filename[0] for filename in XY_filenames_dataset_training_sampled])
#     cube_filenames_training_Y = np.array([filename[1] for filename in XY_filenames_dataset_training_sampled])
#     cube_filenames_validation_X = np.array([filename[0] for filename in XY_filenames_dataset_validation_sampled])
#     cube_filenames_validation_Y = np.array([filename[1] for filename in XY_filenames_dataset_validation_sampled])

#     from locscale.include.emmer.ndimage.map_utils import save_as_mrc
#     # load random 5 cubes and save them as mrc files
#     print("Saving 5 random cubes as mrc files...")
#     print("Location: {}".format(cubedata_random_cubes_dir))

#     for i in range(5):
#         random_index = np.random.randint(0, len(cube_filenames_training_X))
#         random_index_val = np.random.randint(0, len(cube_filenames_validation_X))
#         cube_filename = cube_filenames_training_X[random_index]
#         cube_X_train = np.load(cube_filenames_training_X[random_index])
#         cube_Y_train = np.load(cube_filenames_training_Y[random_index])
#         cube_X_val = np.load(cube_filenames_validation_X[random_index_val])
#         cube_Y_val = np.load(cube_filenames_validation_Y[random_index_val])
#         print("Shape of X_train: {}".format(cube_X_train.shape))
#         # reshape the cubes to 3D
#         cube_X_train = cube_X_train.reshape(cube_size, cube_size, cube_size)
#         cube_Y_train = cube_Y_train.reshape(cube_size, cube_size, cube_size)
#         cube_X_val = cube_X_val.reshape(cube_size, cube_size, cube_size)
#         cube_Y_val = cube_Y_val.reshape(cube_size, cube_size, cube_size)
#         print("Shape of X_train: {}".format(cube_X_train.shape))

#         cube_X_train_new_filepath = os.path.join(cubedata_random_cubes_dir,  os.path.basename(cube_filenames_training_X[random_index])[:-4] + "_X_train.mrc")
#         cube_Y_train_new_filepath = os.path.join(cubedata_random_cubes_dir,  os.path.basename(cube_filenames_training_Y[random_index])[:-4] + "_Y_train.mrc")
#         cube_X_val_new_filepath = os.path.join(cubedata_random_cubes_dir,  os.path.basename(cube_filenames_validation_X[random_index_val])[:-4] + "_X_val.mrc")
#         cube_Y_val_new_filepath = os.path.join(cubedata_random_cubes_dir,  os.path.basename(cube_filenames_validation_Y[random_index_val])[:-4] + "_Y_val.mrc")

#         print("saving cube_X_train to: {}".format(cube_X_train_new_filepath))
#         print("saving cube_Y_train to: {}".format(cube_Y_train_new_filepath))
#         print("saving cube_X_val to: {}".format(cube_X_val_new_filepath))
#         print("saving cube_Y_val to: {}".format(cube_Y_val_new_filepath))

#         save_as_mrc(cube_X_train, cube_X_train_new_filepath, apix=1)
#         save_as_mrc(cube_Y_train, cube_Y_train_new_filepath, apix=1)
#         save_as_mrc(cube_X_val, cube_X_val_new_filepath, apix=1)
#         save_as_mrc(cube_Y_val, cube_Y_val_new_filepath, apix=1)
#         print("============================================")
    

#     # cube_statistics_X_train = get_cube_statistics(cube_filenames_training_X)
#     # cube_statistics_Y_train = get_cube_statistics(cube_filenames_training_Y)
#     # cube_statistics_X_val = get_cube_statistics(cube_filenames_validation_X)
#     # cube_statistics_Y_val = get_cube_statistics(cube_filenames_validation_Y)

#     training_cubes_length = len(cube_filenames_training_X)
#     validation_cubes_length = len(cube_filenames_validation_X)

#     training_data_generator = Custom_Datagenerator(cube_filenames_training_X, cube_filenames_training_Y, batch_size)
#     validation_data_generator = Custom_Datagenerator(cube_filenames_validation_X, cube_filenames_validation_Y, batch_size)
    
#     return training_data_generator, validation_data_generator, training_cubes_length, validation_cubes_length

def create_hdf5_datagenerators(cubedata_top_directory):
    """ creates training and validation datagenerator objects

    Returns:
        training_data_generator (Custom_Datagenerator): training data generator object
        validation_data_generator (Custom_Datagenerator): validation data generator object
    """
    import pandas as pd
    import pickle 

    print("\n>>> CREATE DATAGENERATORS")
    import numpy as np
    import h5py
    import tensorflow as tf

    class HDF5CubeDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, parent_h5_path, key_list, batch_size, cube_size=48):
            """
            HDF5CubeDataGenerator: Custom data generator for loading cube-shaped data from HDF5 files with external links.

            :param parent_h5_path: Path to the parent HDF5 file containing external links to the cubes.
            :param key_list: List of top-level keys to use for data generation.
            :param batch_size: Number of samples per batch.
            :param cube_size: The size of the cubes, assuming the shape is (cube_size, cube_size, cube_size, 1).
            """
            self.parent_h5_path = parent_h5_path
            self.key_list = key_list
            self.batch_size = batch_size
            self.cube_size = cube_size

        def __len__(self):
            # Return the number of batches per epoch
            return int(np.ceil(len(self.key_list) / float(self.batch_size)))

        def __getitem__(self, idx):
            # Calculate which keys to retrieve for this batch
            batch_keys = self.key_list[idx * self.batch_size:(idx + 1) * self.batch_size]

            # Initialize arrays to store the batch data
            X_data = np.empty((len(batch_keys), self.cube_size, self.cube_size, self.cube_size, 1))
            Y_data = np.empty((len(batch_keys), self.cube_size, self.cube_size, self.cube_size, 1))

            # Open the parent HDF5 file and retrieve data for each key in the batch
            with h5py.File(self.parent_h5_path, 'r') as h5_file:
                for i, key in enumerate(batch_keys):
                    # Retrieve the X and Y cube datasets for the current key
                    x_cube_key = list(h5_file[key].keys())[0]  # Assume the first key is the X data
                    y_cube_key = list(h5_file[key].keys())[1]  # Assume the second key is the Y data

                    X_data[i] = h5_file[key][x_cube_key][:]
                    Y_data[i] = h5_file[key][y_cube_key][:]

            return X_data, Y_data

        def on_epoch_end(self):
            # Optionally shuffle the key list at the end of each epoch if required
            np.random.shuffle(self.key_list)
    
    cubedata_directory_training = os.path.join(cubedata_top_directory, "cubedata_training")
    cubedata_directory_validation = os.path.join(cubedata_top_directory, "cubedata_validation")

    # Fetch the HDF5 file paths for the training and validation datasets
    h5_file_training = os.path.join(cubedata_directory_training, "combined_training_dataset_2026.h5")
    h5_file_validation = os.path.join(cubedata_directory_validation, "combined_validation_dataset_2026.h5")

    print(f"Training HDF5 file: {h5_file_training}")
    # Open the HDF5 files to retrieve the keys for the training and validation datasets
    with h5py.File(h5_file_training, 'r') as h5_file:
        key_list_training = list(h5_file.keys())

    print(f"Validation HDF5 file: {h5_file_validation}")
    with h5py.File(h5_file_validation, 'r') as h5_file:
        key_list_validation = list(h5_file.keys())
    
    # Dump keys to json files
    with open(os.path.join(cubedata_top_directory, "keys.json"), 'w') as f:
        json.dump({"training": key_list_training, "validation": key_list_validation}, f)

    training_cubes_length = len(key_list_training)
    validation_cubes_length = len(key_list_validation)
    print(f"Number of training cubes: {training_cubes_length}")

    training_data_generator = HDF5CubeDataGenerator(h5_file_training, key_list_training, batch_size)
    validation_data_generator = HDF5CubeDataGenerator(h5_file_validation, key_list_validation, batch_size)

    return training_data_generator, validation_data_generator, training_cubes_length, validation_cubes_length

        

def fit_UNet_model(UNet_model, training_data_generator, validation_data_generator, run_type, training_cubes_length, validation_cubes_length, model_epoch=None):
    """ fits UNet model, while loading the dataset dynamically with the data generators

    Args:
        UNet_model (tf.keras.Model): UNet model object
        training_data_generator (Custom_Datagenerator): training data generator object
        validation_data_generator (Custom_Datagenerator): validation data generator object
        run_type (string): specifies the run type. Options: ["train", "train_test"]
        model_epoch (int): specifies the saved models epoch number for the train_test method. Defaults to None.

    Returns:
        history (History object): contains information about the fitting process, like the loss and metric performance per epoch
    """
    
    print("\n>>> TRAIN MODEL")
    
    # define size of training and validation datasets
    training_data_size = training_cubes_length
    validation_data_size = validation_cubes_length

    # save log files for tensorboard
    log_dir = os.path.join(saved_models_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # run type and learning rate development decision tree
    if run_type == "train":
        epochs = num_epochs
        if nn_learning_rate_dev_name == "constant":
            nn_callbacks = [save_weights_on_epoch()]
        elif nn_learning_rate_dev_name == "reduce":
            nn_callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1), save_weights_on_epoch()]
    elif run_type == "train_test":
        epochs = 1
        if nn_learning_rate_dev_name == "constant":
            nn_callbacks = [tf.keras.callbacks.ModelCheckpoint(os.path.join(saved_models_dir, "{}_epoch-{}.hdf5".format(model_name, str(model_epoch).zfill(2))))]
        elif nn_learning_rate_dev_name == "reduce":
            nn_callbacks = [tf.keras.callbacks.ModelCheckpoint(os.path.join(saved_models_dir, "{}_epoch-{}.hdf5".format(model_name, str(model_epoch).zfill(2)))), nn_reduce_learning_rate]
        
    # fit model
    nn_callbacks.append(tensorboard_callback)

    history = UNet_model.fit(x = training_data_generator, steps_per_epoch = int(training_data_size // batch_size),
                             epochs = epochs, verbose = 1, validation_data = validation_data_generator,
                             validation_steps = int(validation_data_size // batch_size), callbacks = nn_callbacks)
    
    # Save model
    UNet_model.save(os.path.join(saved_models_dir, "{}_final_epoch_{}.hdf5".format(model_name, str(epochs).zfill(2))))
    
    return history
    
    
#################################################### MEDIUM LEVEL FUNCTIONS ############################################################    

def train_UNet_model(UNet_model):
    """ fits model and processes output data to excel and figure (pdf) format

    Args:
        UNet_model (tf.keras.Model): UNet model object
        start_type (string): specifies the start type. Options: ["from_scratch", "continue"]
    """
    
    print("\n### TRAIN UNET MODEL ###")
    
    # create datagenerators from cubedata
    #cubedata_directory_training = "/home/abharadwaj1/dev/map_sharpening/emmernet/emmernet_training/emmernet_epsilon_custom/cubedata_training"
    #cubedata_directory_validation = "/home/abharadwaj1/dev/map_sharpening/emmernet/emmernet_training/emmernet_epsilon_custom/cubedata_validation"
    training_data_generator, validation_data_generator, length_training, length_validation = create_hdf5_datagenerators(cubedata_dir)
    

    history = fit_UNet_model(
        UNet_model=UNet_model, training_data_generator=training_data_generator, validation_data_generator=validation_data_generator,\
        run_type="train", training_cubes_length=length_training, validation_cubes_length=length_validation, model_epoch=None,
    )
    
    # save training characteristics
    training_history_json_filename = os.path.join(saved_models_dir, "training_history.json")
    try:
        with open(training_history_json_filename, 'w') as fp:
            json.dump(jsonify_dictionary(history.history), fp)
    except:
        print("Could not save training history to json file.")


#################################################### HIGH LEVEL FUNCTIONS ############################################################

def prepare_data():
    """ prepares training, validation and/or test datasets for the neural network 

    """
    from sklearn.model_selection import train_test_split
    # print to user
    print("\n### COLLECTING AND PREPARING TRAINING AND VALIDATION DATA ###")
    # EMDB_ids_during_training = emdb_pdb_training_id + emdb_pdb_validation_id
    
    
    
    collected_data = collect_all_data(collection_data_dir, training_targets_json=training_targets_json, \
                                      num_maps_training=num_maps_training, num_maps_validation=num_maps_validation)

    emdb_keys = collected_data["emdb_keys"]

    # Prepare the dataset for training and validation
    emdb_training_id, emdb_validation_id = train_test_split(emdb_keys, test_size=0.15, random_state=42, shuffle=True)

    XY_filenames_dataset_training = prepare_dataset_for_all_emdbs_parallel(emdb_training_id, \
        cubedata_directory=cubedata_training_dir, \
        collection_directory=collection_data_dir,
        step_size=step_size_trainval, cube_size=cube_size, n_jobs=num_processes, max_cubes=max_cubes_training)

    XY_filenames_dataset_validation = prepare_dataset_for_all_emdbs_parallel(emdb_validation_id, \
        cubedata_directory=cubedata_validation_dir, \
        collection_directory=collection_data_dir,
        step_size=step_size_trainval, cube_size=cube_size, n_jobs=num_processes, max_cubes=max_cubes_validation)

    
    # final print
    print("\n##### THE DATA PREPARATION IS SUCCESSFULLY FINISHED #####")
    

def run_UNet_model(run_type, start_type=None, load_epoch=None):
    """ runs training or test run of the UNet model 

    Args:
        run_type (string): specifies the run type. Options: ["train", "train_test", "test", "test_custom"]
        start_type (string): specifies the start type. Option: ["from_scratch", "continue"]. Defaults to None.
        load_epoch (int): specifies the start epoch, needed if the start type is continue. Defaults to None.
    """
    
    # GPUs
    print("Setting CUDA_VISIBLE_DEVICES to {}".format(GPU_nums))
    print("run_type: {}".format(run_type))
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_nums  
    mirrored_strategy = tf.distribute.MirroredStrategy()
            

    for folder in [model_name_dir, saved_models_dir, training_performance_dir]:
        if not os.path.isdir(folder):
            os.mkdir(folder)
            
    # Select the right type of model

    if use_physics_based_loss:
        model_definition_function = define_model_two_channel
    else:
        if nn_l1_reg is not None and nn_l2_reg is not None:
            model_definition_function = define_model_regularized

        elif use_dropout:
            model_definition_function = define_model_dropout
        else:    
            if cube_size == 32:
                model_definition_function = define_model
            elif cube_size == 64 or cube_size == 48:
                model_definition_function = define_model_large
            else:
                raise ValueError("Cube size {} not supported.".format(cube_size))


    print(tf.keras.backend.image_data_format())
    ## RUN: run type and start type decision tree
    # train / train_test
    if run_type in ["train", "train_test"]:
        # start type
        with mirrored_strategy.scope():
            if use_physics_based_loss:
                class reducePhysicsBasedLoss(tf.keras.losses.Loss):
                    """ custom loss function that reduces physics based loss
                    """
                    def __init__(self):
                        super().__init__(name="reducePhysicsBasedLoss")
                    
                    def laplacian_tf(self, tensor):  
                        laplace_kernel = tf.constant([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=tf.float32)

                        laplace_kernel = tf.reshape(laplace_kernel, [3, 3, 3, 1, 1])
                        tensor = tf.expand_dims(tensor, -1)
                        laplacian = tf.nn.conv3d(tensor, laplace_kernel, [1, 1, 1, 1, 1], "SAME")
                        laplacian = tf.squeeze(laplacian, -1)    
                        
                        return laplacian
            
                            
                    # Then, in your loss function, you can use this layer to compute the Laplacian:
                    def physics_based_loss(self, y_pred, y_true):
                        potential_tf, charge_density_tf = tf.split(y_pred, num_or_size_splits=2, axis=-1)
                        #laplacian_layer = LaplacianLayer()
                        laplacian_potential_tf = -1 * self.laplacian_tf(potential_tf)
                        return tf.reduce_mean(tf.abs(laplacian_potential_tf - charge_density_tf))
                    
                    def data_based_loss(self, y_pred, y_true):
                        # return MAE of y_pred and y_true
                        return tf.reduce_mean(tf.abs(y_pred - y_true))
                        
                    def __call__(self, y_true, y_pred, sample_weight=None):
                        import sys
                        return self.physics_based_loss(y_pred=y_pred, y_true=y_true) + self.data_based_loss(y_pred=y_pred, y_true=y_true)
                
                class PhysicsBasedMetric(tf.keras.metrics.Metric):
                    def __init__(self, name='PhysicsBasedLoss', **kwargs):
                        super(PhysicsBasedMetric, self).__init__(name=name, **kwargs)
                        self.physics_based_loss = self.add_weight(name='pb_loss', initializer='zeros')
                        self.batch_count = self.add_weight(name='batch_count', initializer='zeros')

                    def update_state(self, y_true, y_pred, sample_weight=None):
                        physics_loss = reducePhysicsBasedLoss().physics_based_loss(y_pred, y_true)
                        self.physics_based_loss.assign(physics_loss)
                        self.batch_count.assign_add(1)

                    def result(self):
                        return self.physics_based_loss / self.batch_count

                class DataBasedMetric(tf.keras.metrics.Metric):
                    def __init__(self, name='DataBasedLoss', **kwargs):
                        super(DataBasedMetric, self).__init__(name=name, **kwargs)
                        self.data_based_loss = self.add_weight(name='db_loss', initializer='zeros')
                        self.batch_count = self.add_weight(name='batch_count', initializer='zeros')
                        
                    def update_state(self, y_true, y_pred, sample_weight=None):
                        data_loss = reducePhysicsBasedLoss().data_based_loss(y_pred, y_true)
                        self.data_based_loss.assign(data_loss)
                        self.batch_count.assign_add(1)

                    def result(self):
                        return self.data_based_loss / self.batch_count
                nn_loss = reducePhysicsBasedLoss()
                nn_metric = [PhysicsBasedMetric(), DataBasedMetric()]
            else:
                nn_loss = tf.keras.losses.MeanAbsoluteError()
                nn_metric = ['mse']
            if nn_l1_reg is not None and nn_l2_reg is not None:
                UNet_model = model_definition_function(cube_size, l1_weight=nn_l1_reg, l2_weight=nn_l2_reg)
                UNet_model.compile(optimizer=nn_optimizer, loss=nn_loss, metrics=nn_metric)                
            else:
                UNet_model = model_definition_function(cube_size)
                UNet_model.compile(optimizer=nn_optimizer, loss=nn_loss, metrics=nn_metric)

            print(UNet_model.summary())
            print(model_definition_function.__name__)
        
        # run type
        if run_type == "train":
            train_UNet_model(UNet_model)
        # elif run_type == "train_test":
        #     train_test_UNet_model(UNet_model, start_type) 


    atexit.register(mirrored_strategy._extended._collective_ops._pool.close)
    
    # final print
    print("\n##### EMMERNET HAS SUCCESSFULLY FINISHED RUNNING #####")
    
######################################################## RUN SCRIPT ################################################################
    
def main():
    
    # parse input arguments from user
    args_default = parser.parse_args()
    # Change the default values of the arguments
    # args_dict = vars(args_default)
    # args_dict["run_configuration"] = "neural_network"
    # args_dict["GPU_nums"] = [str(x) for x in range(1,7)]
    # args_dict["num_epochs"] = 15
    # args_dict["batch_size"] = 12*len(args_dict["GPU_nums"])
    # args_dict["model_name"] = "combined_DE_MA_dataset_LS223"
    # args_dict["append_text"] = "TF_EMmerNet_combined_dataset_2"
    # args_dict["cube_size"] = 48
    # args_dict["use_dropout"] = True


    # Change the default values of arguments for segmenting the data
    args_dict = vars(args_default)
    args_dict["run_configuration"] = "data_preparation"
    args_dict["model_name"] = "segmentation_with_curated_micelle"
    args_dict["append_text"] = "scunet_cz48"
    args_dict["cube_size"] = 48
    args_dict["training_targets_json"] = "/home/abharadwaj1/scratch/dev/emmernet_training/segmentation_micelle_dataset_low_pass_filter/create_training_set/7_emdb_full_info_with_curated_micelle.json"
    args_dict["num_processes"] = 10
    # args = argparse.Namespace(**args_dict)
    # set program variables to parsed arguments

    # args_dict = vars(args_default)
    # args_dict["run_configuration"] = "neural_network"
    # args_dict["GPU_nums"] = [str(x) for x in range(1,1)]
    # args_dict["num_epochs"] = 20
    # args_dict["batch_size"] = 64
    # args_dict["model_name"] = "hybrid_model_map_MA_new_2"
    # args_dict["append_text"] = "TF_EMmerNet_MA_dataset_2"
    # args_dict["cube_size"] = 48
    # args_dict["use_dropout"] = True
    # args_dict["nn_learning_rate"] = 1e-4

    args = argparse.Namespace(**args_dict)

    set_variables(args)

    
    # run configuration decision tree
    if "data_preparation" in run_configuration:
        prepare_data()
    elif "neural_network" in run_configuration:
        print("Running neural network")
        print_hyperparameters()
        run_UNet_model(run_type, start_type, load_epoch)
        # print hyperparameters
        
    
    elif "both" in run_configuration:
        prepare_data()
        print_hyperparameters()
        run_UNet_model(run_type, start_type, load_epoch)
    else:
        print("Error: please specify '--run_configuration' as 'data_preparation' or 'neural_network' or 'both', this argument is required")


# run main function
if __name__ == '__main__':
    main()
