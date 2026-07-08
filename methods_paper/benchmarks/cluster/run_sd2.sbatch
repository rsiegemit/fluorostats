#!/bin/bash
#SBATCH -p devel
#SBATCH -t 00:28:00
#SBATCH -N 1
#SBATCH -J fl_sd
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge
export LD_LIBRARY_PATH=/opt/rocm-6.4.1/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/share/sw/ai/tensorflow/2.15.0:$HOME/.local/lib/python3.9/site-packages
python3 -c "import tensorflow as tf; print('TF OK', tf.__version__)"
python3 cluster_infer.py stardist
echo DONE
