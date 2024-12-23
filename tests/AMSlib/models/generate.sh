#!/bin/bash

if [[ $# -ne 2 ]]; then
  echo "Expecting at least 2 CLI argument"
  echo "$0 <directory> cpu|gpu"
  exit
fi

directory=$1
device=$2
root_dir=$(dirname $0)

mkdir -p $directory
echo $device

python ${root_dir}/generate.py single $device ${directory} duq_mean
python ${root_dir}/generate.py single $device ${directory} duq_max
python ${root_dir}/generate.py single $device ${directory} random 

python ${root_dir}/generate.py double $device ${directory} duq_mean
python ${root_dir}/generate.py double $device ${directory} duq_max
python ${root_dir}/generate.py double $device ${directory} random 


python ${root_dir}/generate_linear_model.py single $device ${directory} duq_mean 8 9
python ${root_dir}/generate_linear_model.py single $device ${directory} duq_max 8 9
python ${root_dir}/generate_linear_model.py single $device ${directory} random  8 9

python ${root_dir}/generate_linear_model.py double $device ${directory} duq_mean 8 9
python ${root_dir}/generate_linear_model.py double $device ${directory} duq_max 8 9
python ${root_dir}/generate_linear_model.py double $device ${directory} random  8 9


