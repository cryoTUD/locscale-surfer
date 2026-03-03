#!/bin/bash
#Copyright (C) 2024 Hong cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science and Technology


# Users need to properly set the following variables after installation
#######################################################################
    #EMReady_home=""
    #activate=""
    #EMReady_env=""
    EMReady_home="/home/hcao/data_gu02/papers/emready_na/packages/20240131/EMReady2_use"
    activate="/home/hcao/data_gu02/anaconda3/bin/activate"
    EMReady_env="pth2"
#######################################################################


source $activate $EMReady_env

if [ $# -lt 2 ];then
	echo ""
	echo "EMReady 2.0 - Improvement of Cryo-EM maps by deep learning
Huang Lab @ HUST, http://huanglab.phys.hust.edu.cn/EMReady/"
	echo ""
	echo "USAGE: `basename $0` in_map.mrc out_map.mrc [options]"
	echo ""
	echo "Descriptions:"
	echo "    in_map.mrc  : Input EM density map in MRC2014 format"
	echo "    out_map.mrc : Filename of the output processed density map"
	echo ""
	echo "    -g          : ID(s) of GPU devices to use.  e.g. '0' for GPU #0, and '2,3,6' for GPUs #2, #3, and #6"
	echo "                  default: '0'"
    echo ""
	echo "    -s          : The step of the sliding window for cutting the input map into overlapping boxes. Its value should be an integer within [12, 48]. A smaller stride means a larger number of overlapping boxes"
	echo "                  default: '12'"
    echo ""
	echo "    -b          : Number of boxes input into EMReady in one batch. Users can adjust 'batch_size' according to the VRAM of their GPU devices. Empirically, a GPU with 40 GB VRAM can afford a 'batch_size' of 30"
	echo "                  default: '10'"
    echo ""
	echo "    --use_cpu   : Run EMReady on CPU instead of GPU"
	echo ""
    echo "    -m          : Input mask map in MRC2014 format."
    echo ""
    echo "    -c          : Set the contour level of the mask"
    echo "                  default: '0.0'"
    echo ""
    echo "    -p          : Input structure mask files in PDB or CIF format"
    echo ""
    echo "    -r          : Zone radius in angstroms"
    echo "                  default: '4.0'"
    echo ""
    echo "    --inverse   : Whether to select the inverse mask"
    echo ""

        exit 1
fi

in_map=$1
out_map=$2
mask_map=none
mask_contour=0
mask_str=none
mask_str_radius=4.0
inverse_mask=""
gpu_id="0"
stride=12
batch_size=10
use_cpu=""
model_state_dict_dir=$EMReady_home"/model_state_dicts"

while [ $# -gt 2 ];do
    case $3 in
    -m)
        shift
        mask_map=$3;;
    -c)
        shift
        mask_contour=$3;;
    -p)
        shift
        mask_str=$3;;
    -r)
        shift
        mask_str_radius=$3;;
    --inverse)
        inverse_mask="--inverse_mask";;
    -g)
        shift
        gpu_id=$3;;
    -b)
        shift
        batch_size=$3;;
    -s)
        shift
        stride=$3;;
    --use_cpu)
        use_cpu="--use_cpu";;
    *)
    	echo " ERROR: wrong command argument \"$3\" !!"
    	echo " Type \"$0\" for help !!"
        exit 2;;
    esac
    shift
done

python ${EMReady_home}/pred.py -i $in_map -o $out_map -m $mask_map -c $mask_contour -p $mask_str -r $mask_str_radius $inverse_mask -g $gpu_id -b $batch_size -s $stride $use_cpu -md $model_state_dict_dir
