![GitHub Release](https://img.shields.io/github/v/release/cryotud/locscale-surfer)
[![Python 3.10](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/pypi/l/locscale.svg?color=orange)](https://gitlab.tudelft.nl/aj-lab/locscale/raw/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15488062.svg)](https://doi.org/10.5281/zenodo.15488062)

# LocScale-SURFER 
## _Python_ 

<img src="./docs/img/locscale_surfer.png" width="250" align="right">

`LocScale-SURFER python` is a python script for batch prediction of micellar regions using `LocScale-SURFER`.  

`LocScale-SURFER python` runs on Linux or Windows WSL platforms and is GPU-accelerated if GPU is available, but also runs on CPUs if required.

## Documentation

>[!IMPORTANT]
> Please visit [https://cryotud.github.io/locscale-surfer/](https://cryotud.github.io/locscale-surfer/) for comprehensive documentation, tutorials and troubleshooting.

## Installation

### From source code: 
1. Clone the repository and change to ```locscalesurfer-python``` branch
```bash
git clone https://github.com/cryoTUD/locscale-surfer.git
cd locscale-surfer
git checkout locscalesurfer-python
```
2. Install conda environment
```bash
conda env create -f environment.yml
```
3. Note the path to ```locscale_surfer.py``` script

## Usage 
### Activate the environment
```bash
conda activate locscalesurfer
```
### Command line argument
With GPU 
```bash
python /path/to/locscale_surfer.py -i unsharpened_map.mrc -t target_map.mrc -g 0
```
Without GPU 
```bash
python /path/to/locscale_surfer.py -i unsharpened_map.mrc -t target_map.mrc 
```
### Arguments
- ```-i``` or ```--input```: Input unsharpened map
- ```-o``` or ```--output``` : Output map filename (optional)
- ```-t``` or ```--target``` : Target map for micelle removal (optional)
- ```-g``` or ```--gpu_ids``` : List of GPU IDs to use  with spaces (optional, if GPUs needed)
Additional arguments
- ```-mask``` or ```--mask_path``` : Path to input mask to restrict computation within spatial boundaries
- ```-th``` or ```--threshold``` : Binarisation threshold for micelle subtraction (default: 0.5)
- ```-b``` or ```--batch_size``` : Batch size for prediction (default: 64)

## Credits
`LoScale-SURFER` is facilitated by a number of open-source projects.

- [`OPM database`](https://opm.phar.umich.edu/ppm_server): Orientations of Proteins in Membranes (OPM) database.
- [`Optuna`](https://github.com/optuna/optuna): Hyperparameter optimisation. [MIT license]
- [`SCUNet`](https://github.com/cszn/SCUNet): Semantic segmentation. [Apache 2.0]
- [`EMReady`](http://huanglab.phys.hust.edu.cn/EMReady/) : 3-D SCUNet architecture. [GNU 3.0]


