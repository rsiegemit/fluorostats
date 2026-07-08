#!/bin/bash
#SBATCH -p mi2101x
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -J fl_time
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge; module load pytorch/2.7.1
export PYTHONPATH=/share/sw/ai/pytorch/2.7.1:$HOME/.local/lib/python3.9/site-packages
python3 time_dl.py cellpose
module purge
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages
python3 time_dl.py stardist
echo DONE
