#!/bin/bash
exec ssh -T -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ClearAllForwardings=yes openlibing-DevEnv-661495 'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-9.1.0/lib64; exec python3 /opt/torch-baseline1135-20261103/supervisor_server1_v17.py server1'
