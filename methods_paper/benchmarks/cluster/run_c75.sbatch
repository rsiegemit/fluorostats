#!/bin/bash
#SBATCH -p mi2101x
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J fl_c75
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge; module load pytorch/2.7.1
export PYTHONPATH=/share/sw/ai/pytorch/2.7.1:$HOME/.local/lib/python3.9/site-packages
python3 infer_slices.py cellpose c75_slices/images c75_slices/labels cellpose_c75.csv
module purge
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages
python3 infer_slices.py stardist c75_slices/images c75_slices/labels stardist_c75.csv
echo DONE
