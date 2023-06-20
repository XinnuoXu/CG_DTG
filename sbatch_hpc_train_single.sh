#!/bin/bash
#SBATCH -A BROADSEM-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1	  # nodes requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH -t 20:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=1  # number of cpus to use - there are 32 on each node.
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

#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 7
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 6
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 5
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 4
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 3
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 2

#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 7 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 6 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 5 1000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 4 500
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 3 500
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 2 500

echo "Job ${SLURM_JOB_ID} is done!"
