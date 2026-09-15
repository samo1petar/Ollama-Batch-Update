#!/bin/bash
# Start a controlled ollama serve on :11435 with a known OLLAMA_NUM_PARALLEL.
# usage: ./run_server.sh <num_parallel>
NP="${1:-1}"
pkill -f "OLLAMA_HOST=127.0.0.1:11435" 2>/dev/null
lsof -ti tcp:11435 | xargs kill -9 2>/dev/null
sleep 1
mkdir -p logs
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_NUM_PARALLEL="$NP"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_DEBUG=1
nohup /usr/local/bin/ollama serve > "logs/serve_np${NP}.log" 2>&1 &
echo "started pid $! with OLLAMA_NUM_PARALLEL=$NP -> logs/serve_np${NP}.log"
