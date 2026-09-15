#!/bin/bash
# Start the PR#17144-patched ollama build on :11435.
# usage: ./run_patched_server.sh <num_parallel>
NP="${1:-4}"
lsof -ti tcp:11435 | xargs kill -9 2>/dev/null
sleep 1
cd "$(dirname "$0")/ollama-src" || exit 1
mkdir -p ../logs
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_NUM_PARALLEL="$NP"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_DEBUG=1
export OLLAMA_MODELS="$HOME/.ollama/models"
# cwd is ollama-src, so LibOllamaPath resolves <cwd>/build/lib/ollama
nohup ./ollama-patched serve > "../logs/patched_np${NP}.log" 2>&1 &
echo "started patched server pid $! (NUM_PARALLEL=$NP) -> logs/patched_np${NP}.log"
