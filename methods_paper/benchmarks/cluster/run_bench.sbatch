#!/bin/bash
#SBATCH -p devel
#SBATCH -t 00:28:00
#SBATCH -N 1
#SBATCH -J flstats_dl
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
echo "node: $(hostname)"
echo "### CELLPOSE ###"
module purge
module load pytorch/2.7.1
export PYTHONPATH=/share/sw/ai/pytorch/2.7.1:$HOME/.local/lib/python3.9/site-packages
python3 cluster_cellpose.py
echo "### STARDIST ###"
module purge
module load rocm/6.3.1
export PYTHONPATH=/share/sw/ai/tensorflow/2.15.0:$HOME/.local/lib/python3.9/site-packages
python3 cluster_stardist.py
echo "ALL DONE"
