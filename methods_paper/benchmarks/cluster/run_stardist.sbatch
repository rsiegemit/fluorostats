#!/bin/bash
#SBATCH -p devel
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -J flstats_sd
#SBATCH -o %x_%j.out
cd ~/fluorostats_bench
module purge
export LD_LIBRARY_PATH=/opt/rocm-6.3.1/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/share/sw/ai/tensorflow/2.15.0:$HOME/.local/lib/python3.9/site-packages
python3 -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"
python3 cluster_stardist.py
echo DONE
