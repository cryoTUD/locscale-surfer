import os
import sys
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import json 
import random
import numpy as np
from einops import rearrange
from datetime import datetime
import monai

import argparse
import ast

# Set all random seeds
random.seed(42)
torch.manual_seed(0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
date      = timestamp[:8]
DEF_TRAIN_DIR   = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/cubedata_directory/cubedata_training/XY_filenames_dataset.json'
DEF_VALID_DIR   = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/cubedata_directory/cubedata_validation/XY_filenames_dataset.json'
DEF_MDL_SV_DIR  = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/{date}/saved_models'
DEF_LOSS_DIR    = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/{date}'
DEF_LOSS_PARAMS = "{'alpha':0.25, 'gamma':2.0, 'reduction':'mean'}"
DEF_LR_PARAMS   = "{'mode':'min', 'threshold':0.01, 'factor':0.1, 'patience':240}"
# DEF_MDL_SV_DIR  = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/emmernet_cz48/outputdata/{date}/saved_models'
# DEF_LOSS_DIR    = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/emmernet_cz48/outputdata/{date}'
# DEF_MDL_SV_DIR  = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes_64/scunet_emmernet_swinunetr/outputdata/{date}/saved_models'
# DEF_LOSS_DIR    = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes_64/scunet_emmernet_swinunetr/outputdata/{date}'

# DEF_TRAIN_DIR   = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes_64/scunet_emmernet_swinunetr/cubedata_directory/cubedata_training/XY_filenames_dataset.json'
# DEF_VALID_DIR   = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes_64/scunet_emmernet_swinunetr/cubedata_directory/cubedata_validation/XY_filenames_dataset.json'

def parse_args():
    '''parse the arguments'''
    p = argparse.ArgumentParser(prog='Semantic Segmentation SCUNet',
                                description='Segments lipid belts in transmembrane proteins.'
                                )

    p.add_argument('--train_dir',           default=DEF_TRAIN_DIR,      type=str,               help="JSON file that contains the cube directories of the training input and target.")
    p.add_argument('--valid_dir',           default=DEF_VALID_DIR,      type=str,               help="JSON file that contains the cube directories of the training input and target.")
    p.add_argument('--num_cubes_used',      default=None,               type=int, nargs='+',              help="List of integer numbers specifying the size of the subset of the dataset used to train and validate respectively.\
                                                                                                      Default: all cubes")
    p.add_argument('--model_architecture',  default='SCUNet',           type=str,               help="Type of model architecure used to train. Default: SCUNet")
    p.add_argument('--optimizer',           default='Adam',             type=str,               help="Optimizer used during training, \
                                                                                                      options include: 'Adam'/'SGD'.")
    p.add_argument('--loss_fn',             default='FocalLoss',        type=str,               help="Name of loss function used during training.")
    p.add_argument('--loss_params',         default=DEF_LOSS_PARAMS,    type=ast.literal_eval,  help='Hyperparameters in the loss function. Give dictionary as: "dict".')
    p.add_argument('--loss_info_file_path', default=DEF_LOSS_DIR,       type=str,               help="Directory to save the loss per ...")
    p.add_argument('--l1_lambda',           default=0.0,                type=float,             help="L1 regularization weight. Factor to let the size of the weights count towards the loss.")
    p.add_argument('--track_every',         default=8,                  type=int,               help="Number of batches to pass before the average loss over those batches is returned.")
    p.add_argument('--batch_size',          default=8,                  type=int,               help="Number of cubes trained on per batch per GPU. Default = 8")
    p.add_argument('--cube_size',           default=48,                 type=int,               help="Yet to be defined")
    # p.add_argument('--activ_fn',            default='Sigmoid',          type=str,   help="Type of activation function used on the output of the model.")
    p.add_argument('--model_save_dir',      default=DEF_MDL_SV_DIR,     type=str,               help="A directory to save each epoch's model state.")
    p.add_argument('--num_epochs',          default=10,                 type=int,               help="Specifies the number of training loops performed.")
    p.add_argument('--lr',                  default=0.001,              type=float,             help="Any float between e-8 and 1.0, default=0.001.")
    p.add_argument('--lr_scheduler_params', default=DEF_LR_PARAMS,      type=ast.literal_eval,  help="Hyperparameters for the ReduceLROnPlateau scheduler, \
                                                                                                      default={'mode':'min', 'threshold':0.01, 'factor':0.1, 'patience':240}. \
                                                                                                      Patience refers to the number of batches waited before lr is adjusted.")
    p.add_argument('--gpu_ids',             default=[0, 1, 2],          type=int, nargs='+' ,     help="List of GPU ids (ints) ranging from 0-7.")
    p.add_argument('--timestamp',           default=timestamp,          type=str,               help='Timestamp to save the model+hyperparameters under the correct name.')
    '''store the arguments in a dictionary'''
    args_dict = vars(p.parse_args())
    
    return args_dict

class SimpleDataSet(Dataset):

    def __init__(self, list_of_filenames):
        super().__init__()
        self.filename = list_of_filenames

    def __getitem__(self, idx):
        xy_pair = self.filename[idx]
        x = np.load(xy_pair[0])
        x = rearrange(x, 'h w l c -> c h w l')

        y = np.load(xy_pair[1])
        y = rearrange(y, 'h w l c -> c h w l')
        y = (y>0.5).astype(np.float64)
        return x, y
    
    def __len__(self):
        return len(self.filename)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Assume inputs are raw logits from the final layer of your model
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        targets = targets.type(torch.long)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        pt = torch.exp(-BCE_loss)  # Converts BCE loss to probability
        p  = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        F_loss = at * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
def get_loss_fn(args_dict):
    if args_dict['loss_fn'] == 'FocalLoss':
        params = args_dict['loss_params']
        loss_fn   = FocalLoss(**params)

    if args_dict['loss_fn'] == 'BCEloss':
        params = args_dict['loss_params']
        loss_fn   = nn.BCEWithLogitsLoss(**params)

    if args_dict['loss_fn'] == 'DiceLoss':
        loss_fn   = monai.losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    return loss_fn

def get_optimizer(model, arg_dict):
    lr = arg_dict['lr']
    if arg_dict['optimizer'] == 'Adam':
        optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    if arg_dict['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return optimizer

def train_one_epoch(args_dict, model, epoch_idx, train_dataloader, loss_fn, optimizer, scheduler, loss_store_path):
    running_loss = 0.
    last_loss = 0.
    
    i = 0
    with open(loss_store_path, 'a') as f:
        sys.stdout = f
        # for i, data in enumerate(training_loader):
        for emmap, segm in tqdm(train_dataloader, leave=True):
            # Every data instance is an input + label pair
            # emmap, segm = data
            emmap = emmap.to(args_dict['device'])
            segm  = segm.to(args_dict['device'])

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(emmap)
            # outputs = torch.sigmoid(outputs)

            # Compute the loss and its gradients
            if args_dict['loss_fn'] == 'L1FocalLoss':
                loss = loss_fn(outputs, segm, model)
            else:
                loss = loss_fn(outputs, segm)

            all_parameters_in_model = torch.cat([x.flatten() for x in model.parameters()])

            l1_regularization = args_dict['l1_lambda'] * torch.norm(all_parameters_in_model, 1) / len(all_parameters_in_model)
            loss += l1_regularization
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if i % args_dict['track_every'] == args_dict['track_every']-1:
                last_loss = running_loss / args_dict['track_every'] # loss per batch
                print(f'>>> epoch {epoch_idx+1} batch {i+1} loss: {last_loss}'.format(epoch_idx + 1, i + 1, last_loss))
                sys.stdout.flush()
                running_loss = 0.

            # Step the learning rate scheduler based on the validation loss
            if args_dict['lr_scheduler_params'] is not None:
                scheduler.step(loss.item())

                patience = args_dict['lr_scheduler_params']['patience']
                if i % patience == patience - 1:
                    print(f"current lr={optimizer.param_groups[0]['lr']}")
                sys.stdout.flush()
            i += 1

        sys.stdout = sys.__stdout__


    return last_loss

def train_model(args_dict, model, train_dataloader, valid_dataloader):
    '''Define loss, optim, etc.'''
    loss_fn   = get_loss_fn(args_dict)
    optimizer = get_optimizer(model, args_dict)
    if args_dict['lr_scheduler_params'] is not None:
        params = args_dict['lr_scheduler_params']
        print(params)
        scheduler = ReduceLROnPlateau(optimizer, **params)
    
    '''Run the training'''
    loss_store_path = os.path.join(args_dict['loss_info_file_path'], f'batchloss_{args_dict["timestamp"]}.txt')

    epochs = args_dict['num_epochs']
    for epoch in range(epochs):
        with open(loss_store_path, 'a') as f:
            sys.stdout = f

            print(f'EPOCH {epoch+1}/{epochs}:')
            sys.stdout.flush()

            # set to training state
            model.train(True)

            avg_loss = train_one_epoch(args_dict, model, epoch, train_dataloader, loss_fn, optimizer, scheduler, loss_store_path)

            running_vloss = 0.0

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            # Validate model
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(valid_dataloader):
                    vemmap, vsegm = vdata
                    vemmap = vemmap.to(args_dict['device'])
                    vsegm  = vsegm.to(args_dict['device'])

                    voutputs = model(vemmap)
                    # voutputs = torch.sigmoid(voutputs)
                    
                    if args_dict['loss_fn'] == 'L1FocalLoss':
                        vloss = loss_fn(voutputs, vsegm, model)
                    else:
                        vloss = loss_fn(voutputs, vsegm)

                    all_parameters_in_model = torch.cat([x.flatten() for x in model.parameters()])

                    l1_regularization = args_dict['l1_lambda'] * torch.norm(all_parameters_in_model, 1) / len(all_parameters_in_model)
                    vloss += l1_regularization

                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            sys.stdout = f
            print(f'LOSS train {avg_loss} valid total loss {avg_vloss}')
            sys.stdout.flush()
        
            # save each epoch's model version
            model_name   = f'model_{args_dict["timestamp"]}_{epoch}.pt'
            model_folder = args_dict['model_save_dir']
            model_path   = os.path.join(model_folder, model_name)
            torch.save(model.state_dict(), model_path)

            sys.stdout = sys.__stdout__
    
    return model_path


def store_hyperparam(args_dict):
    output_dir  = args_dict['model_save_dir']
    output_path = os.path.join(output_dir, f'hyperparams_{args_dict["timestamp"]}.json')

    # Save the dictionary as a JSON file
    save_dict = args_dict.copy()
    save_dict.pop('device')
    with open(output_path, 'w') as json_file:
        json.dump(save_dict, json_file, indent=4)

def define_model(args_dict):

    if args_dict['model_architecture'] == 'SCUNet':
        sys.path.insert(1, '/home/tnw-nb4020-03/dev/bep_lotte_micelle/EMReady_v2.0')
        from scunet import SCUNet
        model = SCUNet( 
                in_nc=1, 
                config=[2,2,2,2,2,2,2], 
                dim=32, 
                drop_path_rate=0.0, 
                input_resolution=args_dict['cube_size'], 
                head_dim=16, 
                window_size=3,
                )
        
    if args_dict['model_architecture'] == 'emmernet':
        from emmernet_pytorch import emmernet
        model = emmernet(
                in_nc=1,
                filters=32,
                )
        
    if args_dict['model_architecture'] == 'SwinUNetR':
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
                img_size=(64,64,64),
                in_channels=1,
                out_channels=1,
                feature_size=48,
                use_checkpoint=False,
        )
    
    return model


def train(args):
    torch.manual_seed(0)

    if not os.path.exists(args['model_save_dir']):
        os.makedirs(args['model_save_dir'])
        print(f"Directory '{args['model_save_dir']}' created successfully.")
    else:
        print(f"Directory '{args['model_save_dir']}' already exists.")
    
    MDL_PTHS = os.path.join(args['model_save_dir'], f'model_{args["timestamp"]}_x.pt') 
    loss_store_path = os.path.join(args['loss_info_file_path'], f'batchloss_{args["timestamp"]}.txt')

    path = args['loss_info_file_path']
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")

    with open(loss_store_path, 'a') as f:
        sys.stdout = f
        print(f'Model states will be saved under: {MDL_PTHS}')
        sys.stdout.flush()
    sys.stdout = sys.__stdout__

    # Load Data
    train_files = json.load(open(args['train_dir'],'r'))
    valid_files = json.load(open(args['valid_dir'],'r'))

    if args['num_cubes_used'] is not None:
        train_size, valid_size = args['num_cubes_used']
        train_files = random.sample(train_files, train_size)
        valid_files = random.sample(valid_files, valid_size)

    train_dataset = SimpleDataSet(train_files)
    valid_dataset = SimpleDataSet(valid_files)

    batch_size = args['batch_size']*len(args['gpu_ids'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Define the right model, default is SCUNet
    model = define_model(args)

    # Set device to cuda if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['device'] = device

    model = nn.DataParallel(model, device_ids=args['gpu_ids'])
    model = model.to(device)

    # Create model folder if not present yet
    model_folder = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/{date}/saved_models'

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        print(f"Directory '{model_folder}' created successfully.")
    else:
        print(f"Directory '{model_folder}' already exists.")

    store_hyperparam(args)

    last_model_state_path = train_model(args, model, train_dataloader, valid_dataloader)

    with open(loss_store_path, 'a') as f:
        sys.stdout = f
        print('\nFinished training!')
        sys.stdout.flush()
    sys.stdout = sys.__stdout__

    return last_model_state_path

if __name__ == "__main__":
    args = parse_args()

    # # Run 1 - f1 0.0
    # args['num_cubes_used'] = [2400, 2400]
    # args['num_epochs']     = 10
    # args['batch_size'] = 2
    # args['track_every']= args['batch_size']
    # args['optimizer']  = 'Adam'
    # args['loss_fn']    = 'BCEloss'
    # args['l1_lambda']  = 0.0018568060455270872
    # args['lr']         = 0.028028827246444697
    # args['loss_params'] = {}
    # args['lr_scheduler_params'] = {'mode':'min', 'threshold':0.01733944613540257, 'factor':0.8707018115788627, 'patience':395}

    # Run 2 - f1 0.064
    # args['num_cubes_used'] = [2400, 2400]
    # args['num_epochs']     = 10
    # args['batch_size'] = 6
    # args['track_every']= args['batch_size']
    # args['optimizer']  = 'Adam'
    # args['loss_fn']    = 'DiceLoss'
    # args['l1_lambda']  = 0.03021838400032327
    # args['lr']         = 6.769000889063471e-07
    # args['loss_params'] = {}
    # args['lr_scheduler_params'] = {'mode':'min', 'threshold':0.009343963420901115, 'factor':0.4306712364907186, 'patience': 17}

    # # Run 3 - f1 0.238
    # args['num_cubes_used'] = [2400, 2400]
    # args['num_epochs']     = 10
    # args['batch_size'] = 4
    # args['track_every']= args['batch_size']
    # args['optimizer']  = 'Adam'
    # args['loss_fn']    = 'DiceLoss'
    # args['l1_lambda']  = 0.15383098239016044
    # args['lr']         = 0.00367845118087877
    # args['loss_params'] = {}
    # args['lr_scheduler_params'] = {'mode':'min', 'threshold':0.02854001971615474, 'factor': 0.4714890832173867, 'patience': 138}

    # # Run 4 - f1 0.5005
    # args['num_cubes_used'] = [2400, 2400]
    # args['num_epochs']     = 10
    # args['batch_size'] = 4
    # args['track_every']= args['batch_size']
    # args['optimizer']  = 'Adam'
    # args['loss_fn']    = 'BCEloss'
    # args['l1_lambda']  = 0.0009791447181934341
    # args['lr']         = 0.001577704294471637
    # args['loss_params'] = {}
    # args['lr_scheduler_params'] = {'mode':'min', 'threshold':0.004834803973385312, 'factor':0.6932199245385205, 'patience': 143}

    # Run 5 - f1 0.575
    # args['num_cubes_used'] = [2400, 2400]
    # args['num_epochs']     = 10
    # args['batch_size'] = 4
    # args['track_every']= args['batch_size']
    # args['optimizer']  = 'Adam'
    # args['loss_fn']    = 'BCEloss'
    # args['l1_lambda']  = 0.0025529627248198645
    # args['lr']         = 0.0005342900453956584
    # args['loss_params'] = {}
    # args['lr_scheduler_params'] = {'mode':'min', 'threshold':0.036816631240524665, 'factor':0.6806784587897274, 'patience': 141}

    # Run 6 - f1 0.5695 (emmernet)
    args['num_epochs'] = 1
    args['model_architecture'] = 'SwinUNetR'
    # args['batch_size'] = 
    args['track_every']= args['batch_size']*5
    args['optimizer']  = 'SGD'
    args['loss_fn']    = 'FocalLoss'
    args['l1_lambda']  = 0.0009238989118039449
    args['lr']         = 0.003306613255378132
    args['loss_params'] = {}
    args['lr_scheduler_params'] = {'mode':'min', 'threshold':0.04509819021630981, 'factor':0.7076064169843722, 'patience': 106*20}
    
    train(args)
