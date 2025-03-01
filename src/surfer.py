
import sys 

from .utils import (
    preprocess_emmap,
    extract_all_cube_centers,
    filter_cubecenters_by_mask,
    cube_emmap,
    save_as_mrc,
    resample_map,
    reassemble_map
)

from .scunet import SCUNet

def cube_map(unsharp_map, unsharp_apix, mask, cube_size=48,step_size=32):
    """
    Cubes the input map for processing.

    Steps:
    (a) Load unsharpened map.
    (b) Preprocess (normalize, standardize) unsharpened map.
    (c) Calculate cube centers.
    (d) Extract cubes from cube centers.
    """
    from einops import rearrange

    # (a) Load unsharpened map
    
    # (b) Preprocess (normalize, standardize) unsharpened map
    prepro_unsharp_map = preprocess_emmap(unsharp_map, unsharp_apix, standardize=True)
    if mask is not None:
        preprocessed_mask = preprocess_emmap(mask, unsharp_apix, standardize=False)

    # (c) Calculate cube centers with confidence mask
    cubecenters = extract_all_cube_centers(prepro_unsharp_map, step_size, cube_size)
    
    # (c 1) Filter cube centers by mask
    if mask is not None:
        filtered_cube_centers = filter_cubecenters_by_mask(cubecenters, preprocessed_mask, cube_size, signal_to_noise_cubes=None)[0]
    else:
        filtered_cube_centers = cubecenters

    # (d) Extract cubes from cube centers
    cubed_unsharp_map = cube_emmap(prepro_unsharp_map, filtered_cube_centers, cube_size)
    cubed_unsharp_map = rearrange(cubed_unsharp_map, 'b h w l c -> b c h w l')

    return cubed_unsharp_map, cubecenters, unsharp_apix, prepro_unsharp_map.shape, unsharp_map.shape, filtered_cube_centers

def predict(
    input_map,
    apix,
    mask, 
    batch_size: int = 8,
    cube_size: int = 48,
    step_size: int = 32,
    gpu_ids: list = [0],
    model_state_path: str = None,
):
    """
    Function to predict an enhanced map.

    Inputs:
    - input_map_path: Path to the unsharpened cryo-EM map.
    - model_arch: 'scunet' or 'emmernet'.
    - prediction_path: Path where the prediction is to be saved.
    - model_state_path: Path to the model state file (.pt).
    """
    # Set random seeds for reproducibility
    from collections import OrderedDict

    import numpy as np 
    import random
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm


    random.seed(42)
    torch.manual_seed(42)

    # Load the data
    cubed_unsharp_map, cubecenters, unsharp_apix, prepro_unsharp_shape, unsharp_map_shape, filtered_cube_centers = cube_map(
        input_map, apix, mask, cube_size, step_size)

    eval_dataloader = DataLoader(cubed_unsharp_map, batch_size=batch_size, shuffle=False)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        print(f'Using GPU: {gpu_ids}')
        model = torch.nn.DataParallel(model, device_ids=["cuda:" + str(gpu_id) for gpu_id in gpu_ids])
    elif device == 'cpu':
        print('Using CPU.')  
        model = torch.nn.DataParallel(model, map_location=device)
    
    use_gpu = torch.cuda.is_available()
  
    checkpoint = torch.load(model_state_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    #model_state_dict = torch.load(model_state_path)
    # remove the module. prefix from the keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)

    if use_gpu:
        torch.cuda.empty_cache()
        model = model.to(device)
    model.eval()

    # Make prediction
    prediction = []

    num_batches = len(eval_dataloader)
    with torch.no_grad():
        for i, emmap in enumerate(eval_dataloader):
            emmap = emmap.to(device)
            outputs = model(emmap)
            outputs = torch.sigmoid(outputs)
            
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
    print(f"Prediction done. Shape: {prediction.shape}")

    return prediction

