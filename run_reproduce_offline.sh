#!/usr/bin/env bash
set -euo pipefail

cd /home/algo/chunzhuang/gy/CV/YOLO-World
mkdir -p reproduce_runs
pkill -f my_reproduce_demo.py || true

nohup env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=. \
  /home/algo/anaconda3/envs/yoloworld/bin/python -u my_reproduce_demo.py \
  > reproduce_runs/run_basic_offline_20260608.log 2>&1 < /dev/null &

echo "started pid=$!"
