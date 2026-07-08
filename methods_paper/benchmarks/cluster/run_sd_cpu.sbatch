#!/bin/bash
#SBATCH -p mi2101x
#SBATCH -t 00:40:00
#SBATCH -N 1
#SBATCH -J fl_sdcpu
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages
python3 cluster_infer.py stardist
echo DONE
