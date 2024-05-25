#!/bin/bash
cd ..
n_j=$1
n_m=$2
continue_model_dir=$3

if [ -z "${continue_model_dir}" ]; then
    # continue_model_dir为空或未设定
    python train_ppo.py --n_j "${n_j}" --n_m "${n_m}"
else
    # continue_model_dir已设定
    python train_ppo.py --n_j "${n_j}" --n_m "${n_m}" --continue_model_dir "${continue_model_dir}"
fi
echo "${n_j}*${n_m} model train loop is over!"
