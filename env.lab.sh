#!/usr/bin/env bash
# source this file before make pc-direct / pc-hybrid to target the lab network Pi
PI_IP=192.168.11.3
export CAM_URL="http://$PI_IP:8899/stream.mjpg"
export BASE_URL="http://$PI_IP:5000"
export AGENT_URL="http://$PI_IP:8080"
echo "[lab] AGENT_URL=$AGENT_URL"
