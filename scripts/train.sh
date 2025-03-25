#! /bin/bash

export REPO=`pwd`
export PYTHONPATH=${REPO}:${PYTHONPATH}

# Adapt all this to your setup
# export TMPDIR="/tmp"
export TMPDIR="/home/jovyan/data/temp"
export TORCH_HOME=${TMPDIR}
export HUGGINGFACE_HUB_CACHE=${TMPDIR}
export WANDB_HOME=${TMPDIR}
export DATAROOT="/home/jovyan/data/datasets"

# this is the config will be used
export CFG="train_v1_vitl14_mod.json"

# make the following input-dependent for multi-node
export NNODES=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES="0" # set yours

export MASTER_PORT=$((( RANDOM % 600 ) + 29400 ))
if [ $NNODES -gt 1 ]; then
    export MASTER_PORT=29400
fi

export GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
echo "Start script with python from: `which python`"
torchrun --rdzv-backend=c10d --nnodes=${NNODES} --nproc_per_node=${GPUS} --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} ${REPO}/scripts/train.py --config-file ${REPO}/configs/${CFG} --distributed
