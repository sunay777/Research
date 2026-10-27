#!/bin/zsh
# Double-click to start (or resume) the 100 normalised-input runs.
cd "$(dirname "$0")"
zsh cluster/run_norm_mac.sh 8
echo "Queue finished. You can close this window."
