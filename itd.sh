#!/bin/bash

python3 forgan.py -ds itd -t gru -steps 25000 -n 16 -c 32 -rg 8 -rd 128 -d_iter 3 -hbin 50 -hmin 7e8 -hmax 9e9