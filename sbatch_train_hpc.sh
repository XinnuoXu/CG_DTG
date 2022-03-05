#!/bin/bash
#SBATCH -o /home/hpcxu1/Planning/Tree_enc_dec/logs/sl_tree.out
#SBATCH -e /home/hpcxu1/Planning/Tree_enc_dec/logs/sl_tree.out
#SBATCH -A TEAMER-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1	  # nodes requested
#SBATCH --gres=gpu:4  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH -t 24:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/hpcxu1/miniconda3/bin/activate Plan

# ====================
# Run training. Here we use src/gpu.py
# ====================
echo "Creating directory to save model weights"
BASE_DIR=/home/hpcxu1/Planning/Tree_enc_dec/outputs/
JSONS_DIR=${BASE_DIR}/jsons/
DATA_DIR=${BASE_DIR}/data/
MODEL_DIR=${BASE_DIR}/models/
LOG_DIR=${BASE_DIR}/logs/
mkdir -p ${BASE_DIR}
mkdir -p ${JSONS_DIR}
mkdir -p ${DATA_DIR}
mkdir -p ${MODEL_DIR}
mkdir -p ${LOG_DIR}

# This script does not actually do very much. But it does demonstrate the principles of training
#sh ./scripts.hpc/preprocess_shard.sh
#sh ./scripts.hpc/preprocess.sh
sh ./scripts.hpc/train_xsum.sh

echo "Job ${SLURM_JOB_ID} is done!"
