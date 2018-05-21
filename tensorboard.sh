#!/bin/bash
module load python-3.6.0
source activate pommerman_cpu
tensorboard --logdir="./rl_agent/logs"
