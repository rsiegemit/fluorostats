#!/bin/bash
#SBATCH -p devel
#SBATCH -t 00:28:00
#SBATCH -N 1
#SBATCH -J flstats_cp
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge
module load pytorch/2.7.1
export PYTHONPATH=/share/sw/ai/pytorch/2.7.1:$HOME/.local/lib/python3.9/site-packages
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_DISABLE_CACHE=1
python3 cluster_cellpose.py
echo DONE
