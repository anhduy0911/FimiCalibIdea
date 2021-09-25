#!/bin/bash

python3 forgan.py -ds aqm -t lstm -steps 5000 -n 4 -c 7 -rg 32 -rd 128 -d_iter 6 -hbin 50 -type train -best=True -metric rmse