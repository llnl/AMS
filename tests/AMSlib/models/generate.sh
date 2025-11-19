#!/bin/bash

if [[ $# -ne 1 ]]; then
  echo "Expecting at least 2 CLI argument"
  echo "$0 <directory>"
  exit
fi

directory=$1
device=$2
root_dir=$(dirname $0)

mkdir -p $directory
echo $device

python ${root_dir}/generate.py ${directory}

python ${root_dir}/generate_base_models.py --out-dir ${directory}

python ${root_dir}/generate_linear_model.py ${directory} 8 9

