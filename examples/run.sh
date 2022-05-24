#!/bin/bash
python baseline_track1_navigation.py --run ppo --num-workers 10 --map-id 102 --base-worker-port 50100 &
python baseline_track1_navigation.py --run a3c  --num-workers 10 --map-id 102 --base-worker-port 50130 &
python baseline_track1_navigation.py --run impala  --num-workers 10 --map-id 102 --base-worker-port 50160 &