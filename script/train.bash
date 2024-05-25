#!/bin/bash
# shellcheck disable=SC1068
cd ..
n_j=$1
n_m=$2
continue_model_dir=$3
python train_ppo.py --n_j="${n_j}" --n_m="${n_m}" --continue_model_dir="${continue_model_dir}"
echo "${n_j}*${n_m} model train loop is over!"
