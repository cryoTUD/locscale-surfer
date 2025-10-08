import os
import sys
import argparse
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from scipy.ndimage import uniform_filter   
from scipy.ndimage import label
from src.surfer import predict
from src.locscale_emmernet_utils import (
    load_map,
    preprocess_emmap,
    extract_all_cube_centers,
    filter_cubecenters_by_mask,
    cube_emmap,
    save_as_mrc,
    resample_map,
    reassemble_map
)
from src.scunet import SCUNet

parent_directory = os.path.dirname(os.path.abspath(__file__))
SCUNET_MODEL_PATH = os.path.join(parent_directory, "src", "data", "SURFER_SCUNet.pt")

assert os.path.exists(SCUNET_MODEL_PATH), f"Model file not found at {SCUNET_MODEL_PATH}"

def cube_map(
    unsharp_map_path: str,
    cube_size: int = 48,
    step_size: int = 32,
    standardize: bool = True, 
    mask_path: str = None

):
    """
    Cubes the input map for processing.

    Steps:
    (a) Load unsharpened map.
    (b) Preprocess (normalize, standardize) unsharpened map.
    (c) Calculate cube centers.
    (d) Extract cubes from cube centers.
    """
    # (a) Load unsharpened map
    unsharp_map, unsharp_apix = load_map(unsharp_map_path)
    if mask_path is not None:
        mask, mask_apix = load_map(mask_path)
    
    # (b) Preprocess (normalize, standardize) unsharpened map
    prepro_unsharp_map = preprocess_emmap(unsharp_map, unsharp_apix, standardize)
    if mask_path is not None:
        preprocessed_mask = preprocess_emmap(mask, mask_apix, standardize=False)

    # (c) Calculate cube centers with confidence mask
    cubecenters = extract_all_cube_centers(prepro_unsharp_map, step_size, cube_size)
    
    # (c 1) Filter cube centers by mask
    if mask_path is not None:
        filtered_cube_centers = filter_cubecenters_by_mask(cubecenters, preprocessed_mask, cube_size, signal_to_noise_cubes=None)[0]
    else:
        filtered_cube_centers = cubecenters
    # (d) Extract cubes from cube centers
    cubed_unsharp_map = cube_emmap(prepro_unsharp_map, filtered_cube_centers, cube_size)
    cubed_unsharp_map = rearrange(cubed_unsharp_map, 'b h w l c -> b c h w l')

    return cubed_unsharp_map, cubecenters, unsharp_apix, prepro_unsharp_map.shape, unsharp_map.shape, filtered_cube_centers


def predict(
    input_map_path: str,
    target_map_path: str = None,
    prediction_path: str = None,
    batch_size: int = 8,
    cube_size: int = 48,
    gpu_ids: list = [0],
    model_state_path: str = None,
    standardize: bool = True,
    mask_path: str = None, # new parameter
    threshold: float = 0.5
):

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Load the data
    cubed_unsharp_map, cubecenters, unsharp_apix, prepro_unsharp_shape, unsharp_map_shape, filtered_cube_centers = cube_map(
        input_map_path, cube_size, standardize=standardize, mask_path=mask_path)

    eval_dataloader = DataLoader(cubed_unsharp_map, batch_size=batch_size, shuffle=False)
    if target_map_path is not None:
        target_map, apix = load_map(target_map_path)
    # Set the correct model
    
    model = SCUNet(
        in_nc=1,
        config=[2, 2, 2, 2, 2, 2, 2],
        dim=32,
        drop_path_rate=0.1,
        input_resolution=cube_size,
        head_dim=16,
        window_size=3,
    )


    # Load model for evaluation
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Computation device: {device}')
    print(f"GPU available: {use_gpu}")
    if device.type == 'cuda':
        print(f'Using GPU: {gpu_ids}')
        model = torch.nn.DataParallel(model, device_ids=["cuda:" + str(gpu_id) for gpu_id in gpu_ids])
    else:
        print('Using CPU. This may take a while...')  
        model = model.cpu()
    
    

    try:
        print(f"Loading model state from {model_state_path}...")
        checkpoint = torch.load(model_state_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        #model_state_dict = torch.load(model_state_path)
        model.load_state_dict(state_dict)
    except Exception as e:
        model_state_dict = torch.load(model_state_path)
        model.load_state_dict(model_state_dict)

    model.eval()

    # Make prediction
    prediction = []

    num_batches = len(eval_dataloader)
    with torch.no_grad():
        for emmap in tqdm(eval_dataloader, total=num_batches, desc='Predicting...'):
            emmap = emmap.to(device)
        
            outputs = model(emmap)
            outputs = torch.sigmoid(outputs)
            
            sys.stdout.flush()

            if torch.cuda.is_available():
                outputs = outputs.cpu()

            outputs = outputs.numpy()
            prediction.append(outputs)

    # Concatenate the predictions
    prediction = np.concatenate(prediction, axis=0)
    
    # Reassemble prediction
    prediction = reassemble_map(prediction, filtered_cube_centers, cube_size, prepro_unsharp_shape)


    # Resample reassembly
    prediction = resample_map(prediction, emmap_size_new=unsharp_map_shape, order=2)

    

    if prediction_path is not None:
        # check if directory exists
        # if not os.path.exists(os.path.dirname(prediction_path)):
        #     os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

        output_filename = prediction_path
    else:
        basename = os.path.basename(input_map_path)
        output_filename = os.path.join(os.path.dirname(input_map_path), f'{basename}_micelle_prediction.mrc')
    
    if target_map_path is not None:
        binarized_prediction = (prediction >= threshold).astype(np.float32)
        # smooth the binarized prediction
        smoothed_prediction = uniform_filter(binarized_prediction, size=3)
        # remove micelle
        target_map_without_micelle = target_map * (1 - smoothed_prediction)
        # save the target map without micelle
        basename_target = os.path.basename(target_map_path)
        output_target_filename = os.path.join(os.path.dirname(target_map_path), f'{basename_target}_without_micelle.mrc')
        save_as_mrc(target_map_without_micelle, output_target_filename, apix=apix)
        

    save_as_mrc(prediction, output_filename, apix=unsharp_apix)
    print(f'Prediction saved under: {output_filename}')
    
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict enhanced map using SCUNet/EmmerNet models.')

    parser.add_argument('-i', '--input', required=True, help='Path to the input unsharpened map (.mrc file).')
    parser.add_argument('-t', '--target', required=False, help='Path to the target map (.mrc file) for micelle subtraction')
    parser.add_argument('-o', '--output', required=False, help='Path to the output prediction (.mrc file).')
    parser.add_argument('-mask', '--mask_path', default=None, 
                        help='Path to the mask file (.mrc).')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='Threshold for binarizing the prediction (default: 0.5).')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for prediction (default: 32).')
    parser.add_argument('-cz', '--cube_size', type=int, default=48, help='Cube size (default: 48).')
    parser.add_argument('-g', '--gpu_ids', type=int, nargs='+', default=None,
                        help='List of GPU IDs to use (default: [0]).')

    args = parser.parse_args()

    input_path = args.input
    target_path = args.target
    prediction_path = args.output
    model_arch = "scunet"
    model_state_path = SCUNET_MODEL_PATH
    batch_size = args.batch_size
    cube_size = args.cube_size
    gpu_ids = args.gpu_ids
    threshold = args.threshold

    predict(
        input_map_path=input_path,
        target_map_path=target_path,
        prediction_path=prediction_path,
        batch_size=batch_size,
        cube_size=cube_size,
        gpu_ids=gpu_ids,
        model_state_path=model_state_path,
        mask_path=args.mask_path,
        threshold=threshold
    )
