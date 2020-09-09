#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
python3 -m torch.distributed.launch --nproc_per_node=$3 $(dirname "$0")/panoptic_test.py $1 $2 --launcher pytorch ${@:4}