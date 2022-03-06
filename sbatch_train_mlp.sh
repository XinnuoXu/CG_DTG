#!/bin/bash
#SBATCH -o /home/s1687314/Planning/Tree_enc_dec/slogs/${SLURM_JOB_ID}.out
#SBATCH -e /home/s1687314/Planning/Tree_enc_dec/slogs/${SLURM_JOB_ID}.out
#SBATCH --nodes 1	  # nodes requested
#SBATCH --gres=gpu:4  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 24:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.
#SBATCH -n 4	  # tasks requested

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate Plan

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export MODEL_NAME=gumbel_softmax_3
export RAW_DATA_NAME=xsum_origin_greedy_sent.oracle

export RAW_DATA_DIR=/home/${USER}/Planning/Plan_while_Generate/Content_Selection/
export SCRATCH_HOME=/disk/scratch/${USER}/
export BASE_DIR=${SCRATCH_HOME}/Planning/xsum/${MODEL_NAME}/
mkdir -p ${BASE_DIR}
rsync --archive --update --compress --progress ${RAW_DATA_DIR}/${RAW_DATA_NAME} ${BASE_DIR}/${RAW_DATA_NAME}

# ====================
# Run training. Here we use src/gpu.py
# ====================
sh ./scripts/preprocess_shard.sh
sh ./scripts/preprocess.sh
sh ./scripts/train_xsum.sh

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
OUTPUT_HOME=/home/${USER}/Planning/Tree_enc_dec/outputs/${MODEL_NAME}
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${BASE_DIR} ${OUTPUT_HOME}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
rm -rf ${BASE_DIR}


echo "Job ${SLURM_JOB_ID} is done!"
