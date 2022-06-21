#!/bin/bash
#SBATCH -A TEAMER-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1	  # nodes requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH -t 5:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.
#SBATCH --no-requeue

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

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

# This script does not actually do very much. But it does demonstrate the principles of training
#sh ./scripts.hpc/test_xsum_bartbase.sh
#sh ./scripts.hpc/test_xsum_gumbel_softmax.sh
#sh ./scripts.hpc/test_cnn_bartbase.sh
#sh scripts.hpc/test_cnn_gumbel_softmax.sh
#sh scripts.hpc/test_cnn_gumbel_softmax_with_ext_loss.sh
#sh scripts.hpc/test_cnn_abs_ext.sh 
#sh scripts.hpc/test_cnn_freeze_tmt.sh
#sh scripts.hpc/test_cnn_freeze_tmt_30percent.sh
#sh scripts.hpc/test_cnn_freeze_tmt_groundtruth.sh
#sh scripts.hpc/test_cnn_lead_3.sh
#sh scripts.hpc/test_cnn_not_lead_3.sh
#sh scripts.hpc/test_cnn_random.sh
#sh scripts.hpc/test_cnn_ext.sh
sh scripts_d2t.hpc/test_webnlg_base.sh

echo "Job ${SLURM_JOB_ID} is done!"
