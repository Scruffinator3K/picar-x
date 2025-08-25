#!/usr/bin/env bash
set -euo pipefail

# Move to project root (directory of this script)
cd "$(dirname "$0")"

# Ensure PYTHONPATH includes the project
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Stop any existing demo to free camera
pkill -f example/14.autonomous_demo.py || true

# Start demo in background and log to file
nohup python3 example/14.autonomous_demo.py > /tmp/picarx_autonomous_demo.log 2>&1 &
echo "started"
