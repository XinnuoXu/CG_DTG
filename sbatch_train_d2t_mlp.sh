#!/bin/bash
#SBATCH -o /home/s1687314/Planning/Tree_enc_dec/slogs/plan.selfattn_meanpool
#SBATCH -e /home/s1687314/Planning/Tree_enc_dec/slogs/plan.selfattn_meanpool
#SBATCH --nodes 1	  # nodes requested
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 24:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.
#SBATCH --gres=gpu:3  # use 1 GPU
#SBATCH -n 1	  # tasks requested

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
# need to update
export RAW_DATA_DIR=/home/${USER}/Planning/Tree_enc_dec/outputs.webnlg/
export SCRATCH_DIR=/disk/scratch/${USER}/outputs.webnlg/

mkdir -p ${SCRATCH_DIR}
rsync --archive --update --compress --progress ${RAW_DATA_DIR}/data/ ${SCRATCH_DIR}/data/
echo "rsync --archive --update --compress --progress ${RAW_DATA_DIR}/data/ ${SCRATCH_DIR}/data/"

# ====================
# Run training. Here we use src/gpu.py
# ====================
#sh ./scripts_d2t.mlp/train_webnlg_base.sh 
#sh ./scripts_d2t.mlp/train_webnlg_plan.sh 
#sh ./scripts_d2t.mlp/train_webnlg_plan_maxpool.sh
#sh ./scripts_d2t.mlp/train_webnlg_plan_meanpool.sh
#sh ./scripts_d2t.mlp/train_webnlg_plan_selfattn_maxpool.sh
sh ./scripts_d2t.mlp/train_webnlg_plan_selfattn_meanpool.sh

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
rsync --archive --update --compress --progress ${SCRATCH_DIR}/* ${RAW_DATA_DIR}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
rm -rf /disk/scratch/${USER}

echo "Job ${SLURM_JOB_ID} is done!"
