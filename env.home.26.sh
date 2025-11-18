#!/usr/bin/env bash
# source this file when the home Pi temporarily uses 192.168.0.26 instead of 192.168.0.13
PI_IP=192.168.0.26
export CAM_URL="http://$PI_IP:8899/stream.mjpg"
export BASE_URL="http://$PI_IP:5000"
export AGENT_URL="http://$PI_IP:8080"
echo "[home-26] AGENT_URL=$AGENT_URL"
