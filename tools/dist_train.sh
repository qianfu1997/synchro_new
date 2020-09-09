#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
python3 -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}
