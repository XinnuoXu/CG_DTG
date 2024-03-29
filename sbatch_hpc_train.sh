#!/bin/bash
#SBATCH -A BROADSEM-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1	  # nodes requested
#SBATCH --gres=gpu:3  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH -t 15:00:00  # time requested in hour:minute:seconds
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
#BASE_DIR=/home/hpcxu1/Planning/Tree_enc_dec/outputs/
#JSONS_DIR=${BASE_DIR}/jsons/
#DATA_DIR=${BASE_DIR}/data/
#MODEL_DIR=${BASE_DIR}/models/
#LOG_DIR=${BASE_DIR}/logs/
#mkdir -p ${BASE_DIR}
#mkdir -p ${JSONS_DIR}
#mkdir -p ${DATA_DIR}
#mkdir -p ${MODEL_DIR}
#mkdir -p ${LOG_DIR}

# This script does not actually do very much. But it does demonstrate the principles of training
#sh ./scripts_d2t.hpc/train_webnlg_base.sh
#sh ./scripts_d2t.hpc/train_reinforce_encdec_base.sh
#sh ./scripts_d2t.hpc/train_reinforce_encdec_partial.sh
#sh ./scripts_d2t.hpc/train_reinforce_encdec_partial_numerical.sh
#sh ./scripts_d2t.hpc/train_reinforce_graph_fromscratch.sh
#sh ./scripts_d2t.hpc/train_reinforce_nn.sh
#sh ./scripts_d2t.hpc/train_reinforce_graph_nn.sh
#sh ./scripts_d2t.hpc/train_reinforce_graph_nn_sample.sh

#sh ./scripts_d2t.less_triple_single_shot/train_reinforce_encdec_base.sh
#sh ./scripts_d2t.less_triple_single_shot/train_reinforce_encdec_partial.sh
#sh ./scripts_d2t.less_triple_single_shot/train_reinforce_encdec_partial_numerical.sh 


# ====================
# N-triple fewshot
# ====================
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 2
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 4
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 5
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 6
#sh ./scripts_d2t.ntriple_single/train_reinforce_encdec_partial.sh 7

#sh scripts_d2t.ntriple_single/train_nn.sh 2 1000
#sh scripts_d2t.ntriple_single/train_nn.sh 3 2000
#sh scripts_d2t.ntriple_single/train_nn.sh 4 2000
#sh scripts_d2t.ntriple_single/train_nn.sh 5 3000
#sh scripts_d2t.ntriple_single/train_nn.sh 6 3000
#sh scripts_d2t.ntriple_single/train_nn.sh 7 3000

#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 2 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 3 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 4 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 5 3000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 6 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 7 3000

#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn_sample.sh 2 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn_sample.sh 3 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn_sample.sh 4 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn_sample.sh 5 3000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn_sample.sh 6 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn_sample.sh 7 3000

# Parameters tuning

#sh ./tool/grid_search_nshort.sh 2 2000 > tool/grid.res.2.nn
#sh ./tool/grid_search_nshort.sh 3 2000 > tool/grid.res.3.nn
#sh ./tool/grid_search_nshort.sh 4 2000 > tool/grid.res.4.nn
#sh ./tool/grid_search_nshort.sh 5 3000 > tool/grid.res.5.nn
#sh ./tool/grid_search_nshort.sh 6 2000 > tool/grid.res.6.nn
#sh ./tool/grid_search_nshort.sh 7 3000 > tool/grid.res.7.nn

#sh ./tool/grid_search_nshort_rl.sh 2 1000 > tool/grid.res.2.rl
#sh ./tool/grid_search_nshort_rl.sh 3 1000 > tool/grid.res.3.rl
#sh ./tool/grid_search_nshort_rl.sh 4 4500 > tool/grid.res.4.rl
#sh ./tool/grid_search_nshort_rl.sh 5 3000 > tool/grid.res.5.rl
#sh ./tool/grid_search_nshort_rl.sh 6 1000 > tool/grid.res.6.rl
#sh ./tool/grid_search_nshort_rl.sh 7 3000 > tool/grid.res.7.rl

#sh ./tool/grid_search_nshort_sample.sh 2 500 > tool/grid.res.2.sample
#sh ./tool/grid_search_nshort_sample.sh 3 500 > tool/grid.res.3.sample
#sh ./tool/grid_search_nshort_sample.sh 4 5000 > tool/grid.res.4.sample
#sh ./tool/grid_search_nshort_sample.sh 5 1500 > tool/grid.res.5.sample
#sh ./tool/grid_search_nshort_sample.sh 6 5000 > tool/grid.res.6.sample
#sh ./tool/grid_search_nshort_sample.sh 7 4000 > tool/grid.res.7.sample


# ====================
# N-triple full
# ====================

#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 2 t5-small
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 3 t5-small
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 4 t5-small
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 7 t5-small

#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 2 t5-base
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 3 t5-base
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 4 t5-base
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 7 t5-base

#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 2 t5-large
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 3 t5-large
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 4 t5-large
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial.sh 7 t5-large
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial_test.sh 7 t5-large
#sh ./scripts_d2t.ntriple_full/train_reinforce_encdec_partial_test2.sh 7 t5-large

#sh ./scripts_d2t.ntriple_full/train_webnlg_base.sh t5-large

#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 2 t5-small
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 3 t5-small
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 4 t5-small
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 7 t5-small

#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 2 t5-base
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 3 t5-base
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 4 t5-base
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 7 t5-base

#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 2 t5-large
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 3 t5-large
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 4 t5-large
#sh ./scripts_d2t.ntriple_full/validate_reinforce_encdec_partial.sh 7 t5-large

#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 2 t5-small 5000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 3 t5-small 10000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 4 t5-small 10000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 7 t5-small 15000 false

#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 2 t5-small 5000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 3 t5-small 10000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 4 t5-small 10000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 7 t5-small 15000 true

#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 2 t5-base 5000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 3 t5-base 5000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 4 t5-base 5000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 7 t5-base 10000 false

#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 2 t5-base 5000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 3 t5-base 5000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 4 t5-base 5000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 7 t5-base 10000 true

#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 2 t5-large 5000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 3 t5-large 5000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 4 t5-large 10000 false
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 7 t5-large 10000 false

#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 2 t5-large 5000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 3 t5-large 5000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 4 t5-large 10000 true
#./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 7 t5-large 10000 true

# ====================
# Traditional fewshot Bart-base
# ====================
#sh scripts_d2t.fewshot/train_reinforce_encdec_partial_bbase.sh 0.005
#sh scripts_d2t.fewshot/train_reinforce_encdec_partial_bbase.sh 0.01
#sh scripts_d2t.fewshot/train_reinforce_encdec_partial_bbase.sh 0.05
#sh scripts_d2t.fewshot/train_reinforce_encdec_partial_bbase.sh 0.1

# ====================
# Traditional fewshot Bart-large
# ====================
#sh scripts_d2t.fewshot/train_reinforce_encdec_partial_blarge.sh 0.005
#sh scripts_d2t.fewshot/train_reinforce_encdec_partial_blarge.sh 0.01
#sh scripts_d2t.fewshot/train_reinforce_encdec_partial_blarge.sh 0.05
#sh scripts_d2t.fewshot/train_reinforce_encdec_partial_blarge.sh 0.1

# ====================
# Traditional fewshot nn_rl
# ====================
#sh scripts_d2t.fewshot/train_reinforce_graph_nn_mini.sh 0.005 400
#sh scripts_d2t.fewshot/train_reinforce_graph_nn_mini.sh 0.01 1200
#sh scripts_d2t.fewshot/train_reinforce_graph_nn.sh 0.05 3500
#sh scripts_d2t.fewshot/train_reinforce_graph_nn.sh 0.1 5000

#sh scripts_d2t.fewshot/train_reinforce_graph_nn_sample.sh 0.05 3500
#sh scripts_d2t.fewshot/train_reinforce_graph_nn_sample.sh 0.1 5000


# Parameters tuning -- determin
#sh tool/grid_search_fewshot_determin.sh 0.005 500 > tool/grid.res_0.005.determin
#sh tool/grid_search_fewshot_determin.sh 0.01 1500 > tool/grid.res_0.01.determin
#sh tool/grid_search_fewshot_determin.sh 0.05 2500 > tool/grid.res_0.05.determin
#sh tool/grid_search_fewshot_determin.sh 0.1 4000 > tool/grid.res_0.1.determin

# Parameters tuning -- nn
#sh tool/grid_search_fewshot_nn.sh 0.005 400 > tool/grid.res_0.005.nn
#sh tool/grid_search_fewshot_nn.sh 0.01 1200 > tool/grid.res_0.01.nn
#sh tool/grid_search_fewshot_nn.sh 0.05 3500 > tool/grid.res_0.05.nn
#sh tool/grid_search_fewshot_nn.sh 0.1 5000 > tool/grid.res_0.1.nn

echo "Job ${SLURM_JOB_ID} is done!"
