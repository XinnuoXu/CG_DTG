#!/bin/bash
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=12700
#SBATCH --time=3:00
#SBATCH --cpus-per-task=4
#SBATCH -A TEAMER-SL2-GPU
nvidia-smi
