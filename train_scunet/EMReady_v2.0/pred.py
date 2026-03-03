'''
Copyright (C) 2023 Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science and Technology

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import os
import sys
import json
import torch
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from Bio.PDB import PDBParser
from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning
from torch import nn
from tqdm import tqdm
from math import ceil
from torch import FloatTensor as FT
from torch.autograd import Variable as V
from scunet import SCUNet
from utils import parse_map, pad_map, chunk_generator, get_batch_from_generator, map_batch_to_map, write_map, inverse_map

def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--in_map", "-i", type=str, required=True, help="Input EM density map in MRC2014 format")
    parser.add_argument("--out_map", "-o", type=str, required=True, help="File name of the output processed map")
    parser.add_argument("--mask_map", "-m", type=str, default=None, help="Input mask map in MRC2014 format")
    parser.add_argument("--mask_contour", "-c", type=float, default=0.0, help="Set the contour level of the mask")
    parser.add_argument("--mask_str", "-p", type=str, default=None, help="Input structure mask files in PDB or CIF format")
    parser.add_argument("--mask_str_radius", "-r", type=float, default=4.0, help="Zone radius in angstroms")
    parser.add_argument("--inverse_mask", action='store_true', default=False, help="Whether to select the inverse mask")
    parser.add_argument("--gpu_id", "-g", type=str, default="0", help='ID(s) of GPU devices to use.  e.g. "0" for GPU #0, and "2,3,6" for GPUs #2, #3, and #6')
    parser.add_argument("--batch_size", "-b", type=int, default=10, help="Number of boxes input into EMReady in one batch. Users can adjust `batch_size` according to the VRAM of their GPU devices. Empirically, a GPU with 40 GB VRAM can afford a `batch_size` of 30")
    parser.add_argument("--stride", "-s", type=int, default=12, help="The step of the sliding window for cutting the input map into overlapping boxes. Its value should be an integer within [12,48]")
    parser.add_argument("--use_cpu", action='store_true', default=False, help="Whether to use CPU instead of GPU")
    parser.add_argument("--model_dir", "-md", type=str, required=True, help="Directory name of the state dictionary files for parameters of the trained model (The directory should include two files: 'model_grid_size_0.5.pth' and 'model_grid_size_1.0.pth')")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    in_map = args.in_map
    out_map = args.out_map
    mask_map = args.mask_map
    mask_contour = args.mask_contour
    mask_str = args.mask_str
    mask_str_radius = args.mask_str_radius
    inverse_mask = args.inverse_mask
    gpu_id = args.gpu_id
    batch_size = args.batch_size
    stride = args.stride
    use_cpu = args.use_cpu
    model_dir = args.model_dir

    box_size = 48

    ''' check GPU ''' 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print("# Running on {} GPU(s)".format(n_gpus))
        else:
            print("Error. CUDA not available")
            sys.exit()
    else:
        n_gpus = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("# Running on CPU")

    try:
        assert 48 >= stride >= 12
    except AssertionError:
        print ("Error. `stride` must be in the range of [12, 48]", file=sys.stderr)
        sys.exit(0)

    print("# Loading pre-trained model...")

    _, _, _, voxel_size = parse_map(in_map, ignorestart=False)
    print(f"# Voxel size of input map: {voxel_size}")
    if voxel_size.min() >= 1.0:
        print("# Using the EMReady model with grid size of 1.0 Angstrom")
        apix = 1.0
        model_state_dict_file = f"{model_dir}/model_grid_size_1.0.pth"
    else:
        print("# Using the EMReady model with grid size of 0.5 Angstrom")
        apix = 0.5
        model_state_dict_file = f"{model_dir}/model_grid_size_0.5.pth"
    ### choose a fixed model
    #apix = 1.0
    #model_state_dict_file = f"{model_dir}/model_grid_size_1.0.pth"

    if not use_cpu:
        model_state_dict = torch.load(model_state_dict_file)
    else:
        model_state_dict = torch.load(model_state_dict_file, map_location=torch.device('cpu'))

    model = SCUNet( 
        in_nc=1, 
        config=[2,2,2,2,2,2,2], 
        dim=32, 
        drop_path_rate=0.0, 
        input_resolution=48, 
        head_dim=16, 
        window_size=3,
    )

    model.load_state_dict(model_state_dict)
    if not use_cpu:
        torch.cuda.empty_cache()
        model = model.cuda()
        if n_gpus > 1:
            model = nn.DataParallel(model)

    model.eval()

    print("# Loading map data...")    
    map, origin, nxyz, voxel_size = parse_map(in_map, ignorestart=False, apix=apix)
    try:
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)
    except AssertionError:
        origin_shift =  ( np.round(origin / voxel_size) - origin / voxel_size ) * voxel_size
        map, origin, nxyz, voxel_size = parse_map(in_map, ignorestart=False, apix=apix, origin_shift=origin_shift)
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)
    print("# Map dimensions: {}".format(nxyz))
    map_volume = map.copy()
    del map

    _, _, old_nxyz, old_voxel_size = parse_map(in_map, ignorestart=False, apix=None)

    if mask_map != "none":
        map_mask = map_volume.copy()
        del map_volume
        print("# Loading mask map data...")
        mask, origin_mask, nxyz_mask, voxel_size_mask = parse_map(mask_map, ignorestart=False, apix=apix)
        try:
            assert np.all(np.abs(np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask) < 1e-4)
        except AssertionError:
            origin_shift_mask =  ( np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask ) * voxel_size_mask
            mask, origin_mask, nxyz_mask, voxel_size_mask = parse_map(mask_map, ignorestart=False, apix=apix, origin_shift=origin_shift_mask)
            assert np.all(np.abs(np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask) < 1e-4)
        nxyzstart_mask = np.round(origin_mask / voxel_size_mask).astype(np.int64)
        print("# Mask map dimensions: {}".format(nxyz_mask))

        assert np.all(nxyz_mask <= nxyz)
        try:
            assert np.all(nxyz_mask == nxyz)
        except AssertionError:
            pad_mask = np.zeros(nxyz[::-1]).astype(np.float32)
            nxyz_shift = nxyzstart_mask - nxyzstart
            pad_mask[nxyz_shift[2]:nxyz_shift[2]+nxyz_mask[2], nxyz_shift[1]:nxyz_shift[1]+nxyz_mask[1], nxyz_shift[0]:nxyz_shift[0]+nxyz_mask[0]] = mask
            mask = pad_mask
            origin_mask = origin
            nxyz_mask = nxyz
            nxyzstart_mask = nxyzstart
        if inverse_mask:
            map_volume = np.where(mask <= mask_contour, map_mask, 0).astype(np.float32).copy()
        else:
            map_volume = np.where(mask > mask_contour, map_mask, 0).astype(np.float32).copy()
        del map_mask

    if mask_str != "none":
        map_mask = map_volume.copy()
        del map_volume
        warnings.simplefilter('ignore', BiopythonWarning)
        if mask_str.split(".")[-1][-3:] == "pdb" or mask_str.split(".")[-1][-4:] == "pdb1":
            parser = PDBParser()
        elif mask_str.split(".")[-1][-3:] == "cif":
            parser = MMCIFParser()
        else:
            raise RuntimeError("Unknown type for structure file:", mask_str[-3:])
        structures = parser.get_structure("str", mask_str)
        coords = []
        ### multi-model
        #for structure in structures:
        #    for atom in structure.get_atoms():
        #        if atom.element == 'H':
        #            continue # always ignore Hydroge
        #        coords.append(atom.get_coord())
        structure = structures[0]
        for atom in structure.get_atoms():
            if atom.element == 'H':
                continue # always ignore Hydrogen
            coords.append(atom.get_coord())
        atoms = np.asarray(coords, dtype=np.float32)
        del coords

        map_volume = np.zeros(nxyz[::-1], dtype=np.float32)
        mask = np.zeros(nxyz[::-1], dtype=np.int16)
        for atom in atoms:
            atom_shifted = atom - origin
            lower = np.floor((atom_shifted - mask_str_radius) / voxel_size).astype(np.int32)
            upper = np.ceil ((atom_shifted + mask_str_radius) / voxel_size).astype(np.int32)
            for x in range(lower[0], upper[0] + 1):
                for y in range(lower[1], upper[1] + 1):
                    for z in range(lower[2], upper[2] + 1):
                        if 0 <= x < nxyz[0] and 0 <= y < nxyz[1] and 0 <= z < nxyz[2]:
                            if mask[z, y, x] == 0:
                                vector = np.array([x, y, z], dtype=np.float32) * voxel_size - atom_shifted
                                dist = np.sqrt(vector@vector)
                                if dist < mask_str_radius:
                                    mask[z, y, x] = 1
        if inverse_mask:
            mask = 1 - mask
        map_volume = map_mask * mask.astype(np.float32)
        map_volume = np.where(map_volume > 0.0, map_volume, 0.0).astype(np.float32)
        del map_mask

    map = map_volume.copy()
    del map_volume
    padded_map = pad_map(map, box_size, dtype=np.float32, padding=0.0)
    maximum = np.percentile(map[map > 0], 99.999)
    del map

    map_pred = np.zeros_like(padded_map, dtype=np.float32)
    denominator = np.zeros_like(padded_map, dtype=np.float32)

    print("# Start prediction")

    generator = chunk_generator(padded_map, maximum, box_size, stride)
    ncx, ncy, ncz = [ceil(nxyz[2-i] / stride) for i in range(3)]
    pbar = tqdm(total=ncx*ncy*ncz , ncols=100)

    with torch.no_grad():
        while True:
            positions, chunks = get_batch_from_generator(generator, batch_size, dtype=np.float32)
            
            if len(positions) == 0:
                pbar.close()
                break

            last = positions[-1]
            now = (last[0]-box_size+stride)//stride + (last[1]-box_size+stride)//stride*ncx + (last[2]-box_size+stride)//stride*ncx*ncy + 1
            pbar.update(now - pbar.n)

            X = V(FT(chunks), requires_grad=False).view(-1, 1, box_size, box_size, box_size)
            if not use_cpu:
                X = X.cuda()
        
            y_pred = model(X).view(-1, box_size, box_size, box_size)
            y_pred = y_pred.cpu().detach().numpy()
            map_pred, denominator = map_batch_to_map(map_pred, denominator, positions, y_pred, box_size)

    map_pred = (map_pred/denominator.clip(min=1))[box_size:box_size+nxyz[2], box_size:box_size+nxyz[1], box_size:box_size+nxyz[0]]

    print("# Reverse interpolate the voxel size from {} to {}".format(voxel_size, old_voxel_size))
    origin = nxyzstart * voxel_size
    origin_shift = [0.0, 0.0, 0.0]
    try:
        assert np.all(np.abs(np.round(origin / old_voxel_size) - origin / old_voxel_size) < 1e-4)
    except AssertionError:
        origin_shift =  ( np.round(origin / old_voxel_size) - origin / old_voxel_size ) * old_voxel_size
    map_pred, origin, nxyz, voxel_size = inverse_map(map_pred, nxyz, origin, voxel_size, old_voxel_size, origin_shift)
    assert np.all(np.abs(np.round(origin / old_voxel_size) - origin / old_voxel_size) < 1e-4)
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)

    write_map(out_map, map_pred, voxel_size, nxyzstart=nxyzstart)

if __name__ == "__main__":
    main()
