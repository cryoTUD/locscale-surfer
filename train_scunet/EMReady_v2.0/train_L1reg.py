import os
#os.environ['CUDA_VISIBLE_DEVICES']='2,3' ## CHANGE

from scunet import SCUNet
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json 
from einops import rearrange
from datetime import datetime
from torch import nn
import sys
sys.path.insert(1, '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_scripts')
from eval_utils import get_all_emds_from_dir
from torchvision.ops.focal_loss import sigmoid_focal_loss

from monai.losses import DiceLoss
import random
random.seed(42)
torch.manual_seed(0)
# from focal_loss.focal_loss import FocalLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        F_loss = at * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class CustomLoss(nn.Module):
    def __init__(self, l1_lambda):
        super(CustomLoss, self).__init__()
        self.l1_lambda = l1_lambda

    def forward(self, y_pred, y_true, model):
        # Compute main loss (e.g., cross-entropy)
        main_loss = FocalLoss()(y_pred, y_true)
        
        # Compute L1 regularization term
        # all_parameters_in_model = torch.cat([x.flatten() for x in model.parameters()])
        all_parameters = torch.cat([param.flatten() for param in self.parameters()])

        l1_regularization = self.l1_lambda * torch.norm(all_parameters_in_model, 1) / len(all_parameters_in_model)
                
        # Combine main loss and L1 regularization
        total_loss = main_loss + self.l1_lambda * l1_regularization
        
        return total_loss, main_loss, l1_regularization*self.l1_lambda

## MODEL DEFINITION
model = SCUNet( 
        in_nc=1, 
        config=[2,2,2,2,2,2,2], 
        dim=32, 
        drop_path_rate=0.0, 
        input_resolution=48, 
        head_dim=16, 
        window_size=3,
        )

## DATA SET
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
    
# used chatgpt for redirecting the print statements of the batch loss
def train_one_epoch(model, epoch_idx, training_loader, track_every, patience, timestamp):
    running_loss = 0.
    running_foc_loss = 0.
    running_l1_reg = 0.
    last_loss = 0.
    
    i = 0
    date = timestamp[:8]
    with open(f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/{date}/batchloss_{timestamp}.txt', 'a') as f:
        sys.stdout = f
        # for i, data in enumerate(training_loader):
        for emmap, segm in tqdm(training_loader, leave=True):
            # Every data instance is an input + label pair
            # emmap, segm = data
            emmap = emmap.to(device)
            segm  = segm.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(emmap)
            outputs = torch.sigmoid(outputs)

            # Compute the loss and its gradients
            # loss = loss_fn(outputs, segm)
            loss, foc_loss, l1_reg = loss_fn(outputs, segm, model)
            # loss = sigmoid_focal_loss(outputs, segm)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            running_foc_loss += foc_loss.item()
            running_l1_reg += l1_reg.item()

            if i % track_every == track_every-1:
                last_loss = running_loss / track_every # loss per batch
                last_foc_loss = running_foc_loss / track_every
                last_l1_reg = running_l1_reg / track_every
                print('>>> epoch {} batch {} loss: {} foc_loss: {} l1_reg: {}'.format(epoch_idx + 1, i + 1, last_loss, last_foc_loss, last_l1_reg))
                sys.stdout.flush()
                running_loss = 0.
                running_foc_loss = 0.
                running_l1_reg = 0.

            # Step the learning rate scheduler based on the validation loss
            scheduler.step(loss.item())

            if i % patience == patience - 1:
                print(f"current lr={optimizer.param_groups[0]['lr']}")
            sys.stdout.flush()
            i += 1

        sys.stdout = sys.__stdout__


    return last_loss
  
def train_model(model, epochs, training_loader, validation_loader, track_every, patience, model_folder, timestamp):
    date = timestamp[:8]
    for epoch in range(epochs):
        with open(f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/{date}/batchloss_{timestamp}.txt', 'a') as f:
            sys.stdout = f

            print(f'EPOCH {epoch+1}/{epochs}:')
            sys.stdout.flush()

            # set to training state
            model.train(True)

            avg_loss = train_one_epoch(model, epoch, training_loader, track_every, patience, timestamp)

            running_vloss = 0.0
            running_vfoc_loss = 0.0
            running_vl1_reg = 0.0

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            # Validate model
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vemmap, vsegm = vdata
                    vemmap = vemmap.to(device)
                    vsegm  = vsegm.to(device)

                    voutputs = model(vemmap)
                    voutputs = torch.sigmoid(voutputs)
                    
                    vloss, vfoc_loss, vl1_reg = loss_fn(voutputs, vsegm, model)
                    # vloss = sigmoid_focal_loss(voutputs, vsegm)

                    running_vloss += vloss.item()
                    running_vfoc_loss += vfoc_loss.item()
                    running_vl1_reg += vl1_reg.item()

            avg_vloss = running_vloss / (i + 1)
            avg_vfoc_loss = running_vfoc_loss / (i + 1)
            avg_vl1_reg = running_vl1_reg / (i + 1)
            sys.stdout = f
            print(f'LOSS train {avg_loss} valid total loss {avg_vloss} foc loss {avg_vfoc_loss} l1 reg {avg_vl1_reg}')
            sys.stdout.flush()
            # print(f'LOSS train {avg_loss}')
        
            # save each epoch's model version
            model_name = f'model_{timestamp}_{epoch}.pt'
            torch.save(model.state_dict(), model_folder + '/' + model_name)

            sys.stdout = sys.__stdout__


def save_model_state(model_state, file_dir, file_name):
    if os.path.exists(file_dir + '/' + file_name):
        print(f"Warning: File '{file_dir + '/' + file_name}' already exists.")
        overwrite = input("Do you want to overwrite it? (yes/no): ")
        if overwrite.lower() != 'yes':
            print("Model state not saved.")
            return
    
    # Save the model state
    print(f"Saving model state to '{file_dir + '/' + file_name}'...")

    torch.save(model_state, file_dir + '/' + file_name)
    print("Model state saved successfully.")

def save_hyperparameters(save_dir, train_cube_dir, valid_cube_dir, timestamp, cubedata_dir,
                         cube_size, step_size, epochs, batch_size, optimizer_name,
                         init_learning_rate, loss_name, num_GPUs, activation_name,
                        mode=None, threshold=None, factor=None, patience=None):
    """ prints all hyperparameters to user
    """ 
    import json
    
    emd_ids_train = get_all_emds_from_dir(train_cube_dir)
    emd_ids_valid = get_all_emds_from_dir(valid_cube_dir)

    hyperparameters_dictionary = {
        "network_directory" : '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48',
        "timestamp" : timestamp,
        "cubedata_directory" : cubedata_dir,
        "dataset_characteristics": {
            "training_id": emd_ids_train,
            "validation_id": emd_ids_valid
        },
        "mapdata_sizes": {
            "training_size": len(emd_ids_train),
            "validation_size": len(emd_ids_valid)
        },
        "basics": {
            "cube_size": cube_size,
            "stride": step_size
        },
        "neural_network": {
            "model_name": f'model_{timestamp}_x.pt',
            "num_epochs": epochs,
            "batch_size": batch_size,
            "optimizer_name": optimizer_name,
            "initial learning rate": init_learning_rate,
            "adap_lr_characteristics": {
                mode,
                threshold,
                factor,
                patience
                                        },
            "loss_name": loss_name,
            "num_GPUs": num_GPUs,
            "activation_function": activation_name
        },
    }
    
    hyperparameters_dictionary = json.dumps(hyperparameters_dictionary, indent=4)

    with open(save_dir + '/' + f'model_summary_{timestamp}.txt', 'a') as f:
        json.dump(hyperparameters_dictionary, f)


if __name__ == "__main__":
    # SET TIME
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    ## Create the dataset
    train_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/cubedata_directory/cubedata_training/XY_filenames_dataset.json'
    print(train_path)
    train_files = json.load(open(train_path,'r'))
    train_files = random.sample(train_files, 48*50)
    train_dataset = SimpleDataSet(train_files)
    
    valid_path = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/cubedata_directory/cubedata_validation/XY_filenames_dataset.json'
    valid_files = json.load(open(valid_path,'r'))
    valid_files = random.sample(valid_files, 48*50)
    valid_dataset = SimpleDataSet(valid_files)

    ## Training loop
    from torch import nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from tqdm import tqdm
    
    # Specify hyperparameters
    init_lr    = 0.001
    optimizer  = Adam(model.parameters(), lr=init_lr)
    optim_name = 'Adam'
    # Define the learning rate scheduler
    mode, threshold, factor, patience = 'min', 0.01, 0.1, 240
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, threshold=threshold, factor=factor, patience=patience)
    # loss_fn   = nn.MSELoss()
    # loss_fn   = nn.BCEWithLogitsLoss()
    # loss_fn   = DiceLoss(to_onehot_y=False, sigmoid=False)
    # loss_fn   = torchvision.ops.focal_loss.sigmoid_focal_loss()
    # loss_fn   = FocalLoss(gamma=20.0, alpha=1.0)
    loss_fn   = CustomLoss(l1_lambda=0.1)
    loss_name = 'DiceLoss'
    num_gpus  = torch.cuda.device_count()
    epochs    = 10
    batch_size  = 8*num_gpus
    num_samples = batch_size
    activation_name = 'sigmoid'

    ## Load the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Set device to cuda if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = nn.DataParallel(model, device_ids=[0,1,2])
    model = model.to(device)

    date = datetime.now().strftime('%Y%m%d')
    model_folder = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/{date}/saved_models'

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        print(f"Directory '{model_folder}' created successfully.")
    else:
        print(f"Directory '{model_folder}' already exists.")

    # # STORE HYPERPARAMETERS
    # cubedata_dir = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/cubedata_directory/'
    # save_dir = f'/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/outputdata/{date}'
    # train_cube_dir = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/cubedata_directory/cubedata_training/'
    # valid_cube_dir = '/home/tnw-nb4020-03/dev/bep_lotte_micelle/training_cubes/scunet_cz48/cubedata_directory/cubedata_validation/'
    # save_hyperparameters(save_dir, train_cube_dir, valid_cube_dir, timestamp, cubedata_dir,
    #                      48, int(48/4*3), epochs, batch_size, optim_name,
    #                      init_lr, mode, threshold, factor, patience, loss_name,
    #                      num_gpus, activation_name)

    train_model(model, epochs, train_dataloader, valid_dataloader, batch_size, patience, model_folder, timestamp)