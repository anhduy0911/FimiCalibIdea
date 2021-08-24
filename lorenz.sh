#!/bin/bash

python3 forgan.py -ds lorenz -t gru -steps 10000 -n 32 -c 24 -rg 8 -rd 64 -d_iter 2 -hbin 80 -hmin -11 -hmax 11