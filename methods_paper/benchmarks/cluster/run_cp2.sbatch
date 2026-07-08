#!/bin/bash
#SBATCH -p mi2101x
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -J fl_cp
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge
module load pytorch/2.7.1
export PYTHONPATH=/share/sw/ai/pytorch/2.7.1:$HOME/.local/lib/python3.9/site-packages
export FL_GPU=0
python3 cluster_infer.py cellpose
echo DONE
