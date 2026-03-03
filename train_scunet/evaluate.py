import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.insert(1, '/home/tnw-nb4020-03/dev/bep_lotte_micelle/EMReady_v2.0')
from scunet import SCUNet

from students.segmentation_of_micelles.training_scripts.old.locscale_emmernet_utils import load_map, save_as_mrc, preprocess_emmap, extract_all_cube_centers, resample_map, cube_emmap, reassemble_map
from eval_utils import get_all_emds_from_dir
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
random.seed(42)
import seaborn as sns
torch.manual_seed(42)

from EVAL import calc_fbeta, calc_stats

# used (parts of) functions from emmernet_utils.py
def cube_map(unsharp_map_path, cube_size=48, step_size = 32,
             standardize=True):
    '''   
    consists of the following steps:
            (a) load unsharpened map
            (b) preprocess (norm., stand.) unsharpened map
            (c) calculate cube centers
            (d) extract cubes from cube centers
    '''

    # (a) load unsharpened map
    unsharp_map, unsharp_apix = load_map(unsharp_map_path)

    # (b) preprocess (normalize, standardize) unsharpened map
    prepro_unsharp_map = preprocess_emmap(unsharp_map, unsharp_apix, standardize)

    # (c) calculate cube centers with confidence mask
    cubecenters = extract_all_cube_centers(prepro_unsharp_map, step_size, cube_size)

    # (d) extract cubes from cube centers
    cubed_unsharp_map = cube_emmap(prepro_unsharp_map, cubecenters, cube_size)
    cubed_unsharp_map = rearrange(cubed_unsharp_map, 'b h w l c -> b c h w l')

    return cubed_unsharp_map, cubecenters, unsharp_apix, prepro_unsharp_map.shape, unsharp_map.shape

def save_confusion_matrix(save_path, y_true, y_pred):
    cm = confusion_matrix((y_true>0.5).flatten(), (y_pred>0.5).flatten())

    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

    cm_display.plot()
    plt.savefig(save_path)

    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def eval_model_state(model_arch: str = 'scunet', unsharp_map_path: str = None, segm_path: str = None, smoothening: bool = False, hist_flag: bool = False, cm_flag: bool = False, save_pred: bool = True, cube_size: int = 48, batch_size: int = 8, gpu_ids: list = [0]):
    import sys
    
    '''preprocess input map'''
    cubed_unsharp_map, cubecenters, unsharp_apix, prepro_unsharp_shape, unsharp_map_shape = cube_map(unsharp_map_path, cube_size)

    '''load the data'''
    print(f'batch_size ={batch_size}')
    eval_dataloader = DataLoader(cubed_unsharp_map, batch_size=batch_size, shuffle=False)

    '''set the correct model'''
    try:
        if model_arch == 'scunet':
            sys.path.insert(1, '/home/tnw-nb4020-03/dev/bep_lotte_micelle/EMReady_v2.0')
            from scunet import SCUNet

            # best scunet models state
            # model_state_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/20240416/saved_models/model_20240416_193402_4.pt' # full dataset
            model_state_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/20240415/saved_models/model_20240415_150921_9.pt' # small dataset
            # model_state_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/20240415/saved_models/model_20240415_143512_9.pt' # small dataset, f1 ~ 0.23
            # model_state_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/20240418/saved_models/model_20240418_023911_9.pt' # small dataset, f0.5 score ~ 0.553

            model = SCUNet( 
                    in_nc=1, 
                    config=[2,2,2,2,2,2,2], 
                    dim=32, 
                    drop_path_rate=0.0, 
                    input_resolution=48, 
                    head_dim=16, 
                    window_size=3,
                    )
        elif model_arch == 'emmernet':
            from emmernet_pytorch import emmernet

            # best emmernet model state
            model_state_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/emmernet_cz48/outputdata/20240429/saved_models/model_20240429_202534_3.pt' # full dataset
            # model_state_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/emmernet_cz48/outputdata/20240427/saved_models/model_20240427_023805_9.pt' # small dataset
            # model_state_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/emmernet_cz48/outputdata/20240426/saved_models/model_20240426_205524_9.pt'

            model = emmernet(
                    in_nc=1,
                    filters=32,
                    )

    except Exception as e:
        print(f'Failed with error : {e}')

    model_name = os.path.basename(model_state_path).split('.')[0]
    
    '''load model for evaluation'''
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    use_gpu = torch.cuda.is_available()
    model_state_dict = torch.load(model_state_path)

    model.load_state_dict(model_state_dict)

    if use_gpu:
        torch.cuda.empty_cache()
        model = model.cuda()
    model.eval()

    '''make prediction and reassemble output cubes'''
    prediction = np.empty((0, 1, cube_size, cube_size, cube_size))
    unnorm_pred = np.empty((0, 1, cube_size, cube_size, cube_size))

    num_batches = len(eval_dataloader)

    with torch.no_grad():
        for i, emmap in enumerate(eval_dataloader):

            emmap = emmap.to(device)

            outputs = model(emmap)
            unnorm = outputs
            outputs = torch.sigmoid(outputs)

            print(f'{i+1}/{num_batches}')
            sys.stdout.flush()

            if torch.cuda.is_available():
                outputs = outputs.cpu()
                unnorm = unnorm.cpu()

            outputs = outputs.numpy()
            unnorm = unnorm.numpy()
            
            prediction = np.append(prediction, outputs, axis=0)
            unnorm_pred = np.append(unnorm_pred, unnorm, axis=0)

    '''plot histogram of predicted values'''
    if hist_flag==True:
        fig, ax = plt.subplots()
        sns.histplot(unnorm_pred.flatten(), ax=ax, bins=100)
        fig.savefig(f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/figures/histograms/hist_{model_name}_{eval_id}.png')

    '''load correct output'''
    segm, _ = load_map(segm_path)

    '''reassemble prediction'''
    prediction = reassemble_map(prediction, cubecenters, cube_size, prepro_unsharp_shape)

    '''resample reassembly'''
    prediction = resample_map(prediction, emmap_size_new=unsharp_map_shape, order=2)

    sys.stdout.flush()

    if smoothening == True:
        from scipy.ndimage import uniform_filter
        prediction = uniform_filter(prediction, size=3)

    '''evaluate prediction'''
    if cm_flag == True:
        cm_path = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/figures/confusion_matrices/pred_{model_name}_emd_{eval_id}.png'
        tn, fp, fn, tp = save_confusion_matrix(cm_path, segm, prediction)
    else: 
        tp, fp, fn, tn = calc_stats(segm, prediction)

    if save_pred == True:
        eval_id    = os.path.basename(unsharp_map_path).split('_')[1]
        print(f'saving prediction {eval_id}...')
        pred_folder = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/predictions/{model_name}'
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
            print(f"Directory '{pred_folder}' created successfully.")
        else:
            print(f"Directory '{pred_folder}' already exists.")
        
        pred_path = os.path.join(pred_folder, f'pred_{model_name}_{eval_id}.mrc')
        save_as_mrc(prediction, pred_path, apix=unsharp_apix)

    return prediction, tn, fp, fn, tp

def calc_precision(tp, fp):
    return tp / (tp + fp)

def calc_recall(tp, fn):
    return tp / (tp + fn)

def main():
    # Set device to cuda if possible
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 24
    eval_id = '4288'

    ## Define all paths
    # input
    input_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/unsharpened_maps/'
    input_file = f'EMD_{eval_id}_unsharpened_fullmap.mrc'
    input_path = input_folder + '/' + input_file

    # correct output
    target_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF/' 
    target_file = f'emd_{eval_id}_micelle_lp.mrc'
    target_path = target_folder + '/' + target_file

    # set model architecture
    model_arch = 'scunet'

    _, tn, fp, fn, tp = eval_model_state(model_arch, input_path, target_path, eval_id, smoothening=False, hist_flag=True, cm_flag=True, save_pred=True, batch_size=batch_size)

    precision = calc_precision(tp, fp)
    recall    = calc_recall(tp, fn)
    print(eval_id)
    print(f'Precision is: {precision}')
    print(f'Recall is:    {recall}')
    print(tn)
    print(fp)
    print(fn)
    print(tp)

def make_predictions(model_arch: str = 'scunet', input_folder:str = None, target_folder: str = None, emdb_list: list = None, output_file: str = None, save_pred: bool = True, gpu_ids: list = [0]):
    batch_size = 8*3

    nr_preds = len(emdb_list)

    with open(output_file, 'a') as file:
        file.write(f'emdb_id, tn, fp, fn, tp, precision, recall, f1\n')

    for i, emdb_id in enumerate(emdb_list):
        print(emdb_id)
        sys.stdout.flush()

        ## Define input and target paths 
        # input
        input_file = f'EMD_{emdb_id}_unsharpened_fullmap.mrc'
        input_path = os.path.join(input_folder, input_file)

        # target
        target_file = f'emd_{emdb_id}_micelle_lp.mrc'
        target_path = os.path.join(target_folder, target_file)

        ## Evaluate the model state
        _, tn, fp, fn, tp = eval_model_state(model_arch=model_arch, unsharp_map_path=input_path, segm_path=target_path, save_pred=True, smoothening=False, cm_flag=False, cube_size=48, batch_size=batch_size, gpu_ids=gpu_ids)
                                
        precision = calc_precision(tp, fp)
        recall    = calc_recall(tp, fn)
        f1        = calc_fbeta(tp, fp, fn, 1.0)

        #### WRITE RESULTS TO A FILE
        with open(output_file, 'a') as file:
            file.write(f'{emdb_id}, {tn}, {fp}, {fn}, {tp}, {precision}, {recall}, {f1}\n')

        print(f'{i+1}/{nr_preds} Finished prediction for {emdb_id}.')
        sys.stdout.flush()

    print('All done.')
    sys.stdout.flush()


if __name__ == '__main__':

    # emdb_list = ["28779",  "12128", "27894", "33365",
    #             "14764", "28066", "26597", "13972",
    #             "33615", "11925", "25825", "25849", 
    #             "0774", "40500", "21454", "31037", 
    #                     "33803", "28584", "12271"]
    
    # test_ids  = ['0825', '11922', '13940', '14139', '14452', '14633', '14650', '14792', '15010', '21972', '23749', '27134', '27000']
    # emdb_list = ["28779", "12128", "27894", "33365", "14764", "28066", "26597", "13972", "33615", "11925", "25825", "25849", "0774",
    #                 "40500", "21454", "31037", "33803", "28584", "12271"]

    # emdb_list = ['0093', '0094', '0193', '0257', '0415', '0499', '4272', '4588', '4589', '4593', '4611', '4733', '4746', '4789', '4997', '7009', '7127', '7133', '7882', '8702', '8958', '8960', '9112', '9610', '9931', '9934', '9935', '9939', '10279', '10418', '20145', '20146']

    # emdb_list = test_ids + emdb_list
    
    # input_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/unsharpened_maps'
    # target_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF'
    # output_file   = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/predictions/model_20240416_193402_4/train_results.txt'
    # gpu_ids = [0,1,2,3]

    # make_predictions(model_arch='scunet', input_folder=input_folder, target_folder=target_folder, emdb_list=emdb_list, output_file=output_file, save_pred=True, gpu_ids=gpu_ids)

    # output_file = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/predictions/model_20240429_202534_3/train_results.txt'
    # make_predictions(model_arch='emmernet', input_folder=input_folder, target_folder=target_folder, emdb_list=emdb_list, output_file=output_file, save_pred=True, gpu_ids=gpu_ids)



    
    emdb_list = ['4288']

    input_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/unsharpened_maps'
    target_folder = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/micelles2_final_LPF'
    output_file   = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/data/predictions/model_20240427_023805_9/f0.5_results.txt'
    gpu_ids = [0]

    make_predictions(model_arch='scunet', input_folder=input_folder, target_folder=target_folder, emdb_list=emdb_list, output_file=output_file, save_pred=True, gpu_ids=gpu_ids)