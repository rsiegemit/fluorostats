#!/bin/bash
cd $WORK/vesselexpress
# 1. wait for download to finish (curl gone AND size stable)
while pgrep -f "curl.*VesselExpress" >/dev/null; do sleep 30; done
prev=0; while :; do cur=$(stat -c%s VesselExpress_Data.zip 2>/dev/null||echo 0); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 20; done
echo "DOWNLOAD_DONE size=$(stat -c%s VesselExpress_Data.zip)"
# 2. unzip only Figure2C_E Raw + Binary (the seg GT) blocks
mkdir -p data
unzip -o -q VesselExpress_Data.zip 'Figure2C_E/*/Raw/*' 'Figure2C_E/*/Binary/*' -d data 2>&1 | tail -2
echo "UNZIP_DONE files=$(find data -name '*.tif*'|wc -l)"
# 3. run benchmark
module load pytorch/2.7.1 2>/dev/null
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:/share/sw/ai/pytorch/2.7.1
python3 $WORK/bench_vesselexpress.py > $WORK/vesselexpress/ve_bench.log 2>&1
echo "BENCH_DONE"
