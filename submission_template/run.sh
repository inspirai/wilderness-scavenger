#!/bin/bash

set -e

echo "======== Copying evaluation scripts ========"
cp /root/submission_template/*.py . && ls
echo "================= Finished ================="

python eval.py --track $1 > /root/logs/${EVAL_ID}.txt 2>&1
