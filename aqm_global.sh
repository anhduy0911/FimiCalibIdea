#!/bin/bash

python3 forgan_multistep.py -ds aqm -t lstm -steps 5000 -n 3 -c 7 -p 3 -rg 32 -rd 128 -d_iter 6 -hbin 50 -type train -best=True -metric rmse -raw True -mul True