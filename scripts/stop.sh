#!/bin/bash

REPO=`pwd`
CACHE="${DATA}/cache"
PID_FILE="${CACHE}/unidepth.pid"

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Sending SIGINT to PID $PID"
        kill -INT "$PID"
    else
        echo "Process $PID not running"
    fi
else
    echo "PID file $PID_FILE does not exist"
fi