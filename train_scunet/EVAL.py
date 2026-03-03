import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from students.segmentation_of_micelles.training_scripts.old.TRAIN import SimpleDataSet, define_model
torch.manual_seed(0)

import json
import random
random.seed(42)
import numpy as np

def calc_precision(tp, fp):
    return tp / (tp + fp)

def calc_recall(tp, fn):
    return tp / (tp + fn)

def calc_fbeta(tp, fp, fn, beta):
    precision = calc_precision(tp, fp)
    recall    = calc_recall(tp, fn)
    fbeta = (1 + beta*beta) * precision * recall / (beta*beta*precision + recall)
    return fbeta

def calc_stats(y_true, y_pred):
    '''convert to binary'''
    y_true_bool = (y_true > 0.5).astype(bool)
    y_pred_bool = (y_pred > 0.5).astype(bool)

    '''calculate confusion'''
    tp = np.sum(np.logical_and( y_true_bool, y_pred_bool))
    fp = np.sum(np.logical_and(~y_true_bool, y_pred_bool))
    fn = np.sum(np.logical_and( y_true_bool,~y_pred_bool))
    tn = np.sum(np.logical_and(~y_true_bool,~y_pred_bool))

    return tp, fp, fn, tn

def evaluate(args_dict, model_state_path, beta):
    '''Load Data'''
    valid_files = json.load(open(args_dict['valid_dir'],'r'))

    if args_dict['num_cubes_used'] is not None:
        _, valid_size = args_dict['num_cubes_used']
        valid_files = random.sample(valid_files, valid_size)

    valid_dataset = SimpleDataSet(valid_files)

    '''Divide data into batches'''
    batch_size = args_dict['batch_size']*len(args_dict['gpu_ids'])

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    '''Define and load the model'''
    model = define_model(args_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = nn.DataParallel(model, device_ids=args_dict['gpu_ids'])

    model_state_dict = torch.load(model_state_path)
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    '''Start evaluating'''
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            emmap, segm = data
            emmap = emmap.to(device)
            # segm  = segm.to(args_dict['device'])

            outputs = model(emmap)
            pred = torch.nn.functional.sigmoid(outputs)

            # calculate the statistics for batch and add them to total
            tp, fp, fn, tn = calc_stats(segm.numpy(), pred.cpu().detach().numpy())
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    '''Calculate the accuracy metric'''
    fb_score = calc_fbeta(total_tp, total_fp, total_fn, beta)
    
    return fb_score, total_tp, total_fp, total_fn, total_tn

from students.segmentation_of_micelles.training_scripts.old.TRAIN import parse_args, train
if __name__ == "__main__":
    args = parse_args()
    args['num_cubes_used'] = None
    args['batch_size'] = 8*5
    args['gpu_ids']    = [0]
    args['model_architecture'] = 'SCUNet'

    model_state_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/emmernet_cz48/outputdata/20240430/saved_models/model_20240430_133222_7.pt' # scunet + ememrnet params at epoch 8
    torch.cuda.empty_cache()
    f1_score, total_tp, total_fp, total_fn, total_tn = evaluate(args, model_state_path, 1.0)

    print(f'f1={f1_score}')
    print(f'tp {total_tp} fp {total_fp} fn {total_fn} tn {total_tn}')
    print(f'precision {calc_precision(total_tp, total_fp)}')
    print(f'recall {calc_recall(total_tp, total_fn)}')