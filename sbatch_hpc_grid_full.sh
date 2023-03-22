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

#sh tool/grid_search_nfull_determin.sh 2 t5-base 10000 > ./tool/grid.res.ntriples.full/grid.res.2.determin
#sh tool/grid_search_nfull_determin.sh 3 t5-base 10000 > ./tool/grid.res.ntriples.full/grid.res.3.determin
#sh tool/grid_search_nfull_determin.sh 4 t5-base 15000 > ./tool/grid.res.ntriples.full/grid.res.4.determin
#sh tool/grid_search_nfull_determin.sh 7 t5-base 30000 > ./tool/grid.res.ntriples.full/grid.res.7.determin

#sh tool/grid_search_nfull_ffn_nosample.sh 2 t5-base 500 > ./tool/grid.res.ntriples.full/grid.res.2.nn
#sh tool/grid_search_nfull_ffn_nosample.sh 3 t5-base 1500 > ./tool/grid.res.ntriples.full/grid.res.3.nn
#sh tool/grid_search_nfull_ffn_nosample.sh 4 t5-base 2000 > ./tool/grid.res.ntriples.full/grid.res.4.nn
sh tool/grid_search_nfull_ffn_nosample.sh 7 t5-base 5500 > ./tool/grid.res.ntriples.full/grid.res.7.nn

echo "Job ${SLURM_JOB_ID} is done!"
