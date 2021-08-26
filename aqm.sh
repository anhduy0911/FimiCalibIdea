#!/bin/bash

/Users/anhduy0911/miniforce_x86_64/envs/fimi/bin/python /Users/anhduy0911/Projects/Lab/Fimi/Calibration/ModelIdeas/forgan.py -ds aqm -t lstm -steps 5000 -n 4 -c 31 -rg 64 -rd 256 -d_iter 6 -hbin 50 -type test -best=False -metric rmse