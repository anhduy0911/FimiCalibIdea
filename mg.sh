#!/bin/bash

python3 forgan.py -ds mg -t lstm -steps 5000 -n 4 -c 32 -rg 64 -rd 256 -d_iter 6 -hbin 50 -hmin 0.4 -hmax 1.4