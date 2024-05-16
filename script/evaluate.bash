#!/bin/bash
# shellcheck disable=SC1068
cd ..
n_j=$1
n_m=$2
model_path=$3
python eval.py  --n_j="${n_j}" --n_m="${n_m}" --eval_model_path="${model_path}"
echo "${n_j}*${n_m} model evaluate loop is over!"
