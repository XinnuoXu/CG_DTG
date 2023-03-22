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

#sh ./tool/grid_search_nshort_determin.sh 7 8000 > tool/grid.res.ntriples.single/grid.res.7.determin
#sh ./tool/grid_search_nshort_determin.sh 6 6000 > tool/grid.res.ntriples.single/grid.res.6.determin
#sh ./tool/grid_search_nshort_determin.sh 5 5000 > tool/grid.res.ntriples.single/grid.res.5.determin
#sh ./tool/grid_search_nshort_determin.sh 4 4000 > tool/grid.res.ntriples.single/grid.res.4.determin
#sh ./tool/grid_search_nshort_determin.sh 3 3000 > tool/grid.res.ntriples.single/grid.res.3.determin
#sh ./tool/grid_search_nshort_determin.sh 2 2000 > tool/grid.res.ntriples.single/grid.res.2.determin

#sh ./tool/grid_search_nshort_ffn_nosample.sh 7 2000 > tool/grid.res.ntriples.single/grid.res.7.nn
#sh ./tool/grid_search_nshort_ffn_nosample.sh 6 2000 > tool/grid.res.ntriples.single/grid.res.6.nn
#sh ./tool/grid_search_nshort_ffn_nosample.sh 5 1000 > tool/grid.res.ntriples.single/grid.res.5.nn
#sh ./tool/grid_search_nshort_ffn_nosample.sh 4 500 > tool/grid.res.ntriples.single/grid.res.4.nn
#sh ./tool/grid_search_nshort_ffn_nosample.sh 3 500 > tool/grid.res.ntriples.single/grid.res.3.nn
sh ./tool/grid_search_nshort_ffn_nosample.sh 2 500 > tool/grid.res.ntriples.single/grid.res.2.nn

echo "Job ${SLURM_JOB_ID} is done!"
