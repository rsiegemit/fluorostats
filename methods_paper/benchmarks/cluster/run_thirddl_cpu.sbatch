#!/bin/bash
#SBATCH -p mi2101x
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J fl_3dl
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge; module load pytorch/2.7.1
export PYTHONPATH=/share/sw/ai/pytorch/2.7.1:$HOME/.local/lib/python3.9/site-packages
pip install --user -q "cellpose[omni]" omnipose 2>&1 | tail -2
python3 cluster_thirddl.py
echo ALLDONE
