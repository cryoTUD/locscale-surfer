from .scunet import SCUNet
from .utils import (
    cube_emmap,
    extract_all_cube_centers,
    filter_cubecenters_by_mask,
    preprocess_emmap,
    reassemble_map,
    resample_map,
)


def cube_map(unsharp_map, unsharp_apix, mask, cube_size=48, step_size=32):
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
        filtered_cube_centers = filter_cubecenters_by_mask(
            cubecenters, preprocessed_mask, cube_size, signal_to_noise_cubes=None
        )[0]
    else:
        filtered_cube_centers = cubecenters

    # (d) Extract cubes from cube centers
    cubed_unsharp_map = cube_emmap(prepro_unsharp_map, filtered_cube_centers, cube_size)
    cubed_unsharp_map = rearrange(cubed_unsharp_map, "b h w l c -> b c h w l")

    return (
        cubed_unsharp_map,
        cubecenters,
        unsharp_apix,
        prepro_unsharp_map.shape,
        unsharp_map.shape,
        filtered_cube_centers,
    )


def select_device_based_on_os():
    # if linux use torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def predict(
    input_map,
    apix,
    mask,
    batch_size: int = 8,
    cube_size: int = 48,
    step_size: int = 32,
    gpu_ids: list | None = None,
    model_state_path: str = None,
    progress_cb = None, 
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
    import random
    from collections import OrderedDict

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    random.seed(42)
    torch.manual_seed(42)
    
    if gpu_ids is None:
        gpu_ids = [0]

    # Load the data
    (
        cubed_unsharp_map,
        cubecenters,
        unsharp_apix,
        prepro_unsharp_shape,
        unsharp_map_shape,
        filtered_cube_centers,
    ) = cube_map(input_map, apix, mask, cube_size, step_size)

    eval_dataloader = DataLoader(
        cubed_unsharp_map, batch_size=batch_size, shuffle=False
    )
    n_batches = len(eval_dataloader) 

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
    device = select_device_based_on_os()
    if device.type == "cuda":
        print(f"Using CUDA GPU(s): {gpu_ids}")
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    elif device.type == "mps":
        print("Using Apple Silicon GPU (MPS).")
        model = model.to(device)
    else:
        print("Using CPU.")

    use_gpu = device.type in ["cuda", "mps"]

    checkpoint = torch.load(model_state_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # remove the module. prefix from the keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)

    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Make prediction
    prediction = []

    with torch.no_grad():
        for i, emmap in enumerate(eval_dataloader):
            emmap = emmap.to(device)
            outputs = torch.sigmoid(model(emmap))

            if use_gpu:
                outputs = outputs.detach().cpu()

            prediction.append(outputs.numpy())
            
            if progress_cb is not None:
                pct = int((i + 1) / n_batches * 100)
                progress_cb(pct)

    # Concatenate the predictions
    prediction = np.concatenate(prediction, axis=0)

    # Reassemble prediction
    prediction = reassemble_map(
        prediction, filtered_cube_centers, cube_size, prepro_unsharp_shape
    )

    # Resample reassembly
    prediction = resample_map(prediction, emmap_size_new=unsharp_map_shape, order=2)
    print(f"Prediction done. Shape: {prediction.shape}")

    return prediction
