#!/usr/bin/env bash
set -euo pipefail

# start_servers.sh — Launch 3 MLX servers for the 3-model arena comparison
# Requires: ~/.local/share/mlx-server/venv/ with mlx-lm installed
# RAM needed: ~85GB (122B ~65GB + 35B ~20GB + 0.8B ~0.5GB)

GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
BOLD='\033[1m'
RESET='\033[0m'

VENV="$HOME/.local/share/mlx-server/venv"

declare -A MODELS=(
    [8800]="arthurcollet/Qwen3.5-122B-A10B-mlx-nvfp4"
    [8801]="mlx-community/Qwen3.5-35B-A3B-4bit"
    [8802]="mlx-community/Qwen3.5-0.8B-4bit"
)

declare -A LABELS=(
    [8800]="122B"
    [8801]="35B"
    [8802]="0.8B"
)

echo -e "${BOLD}MLX 3-Model Arena — Starting Servers${RESET}"
echo "─────────────────────────────────────"

# Activate venv
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo -e "${RED}✗ Venv not found at $VENV${RESET}"
    echo "  Install: python3 -m venv $VENV && $VENV/bin/pip install mlx-lm"
    exit 1
fi
source "$VENV/bin/activate"

# Kill existing processes on target ports
for port in 8800 8801 8802; do
    lsof -ti :"$port" | xargs kill -9 2>/dev/null || true
done
sleep 1

# Launch each server in background
for port in 8800 8801 8802; do
    model="${MODELS[$port]}"
    label="${LABELS[$port]}"
    logfile="/tmp/mlx-server-${port}.log"

    echo -e "  Launching ${BOLD}${label}${RESET} on port ${port}..."
    nohup python -m mlx_lm.server --model "$model" --port "$port" \
        > "$logfile" 2>&1 &
done

# Wait for servers to initialize
echo ""
echo "Waiting for servers to load models..."
sleep 10

# Health check each server
ok=0
fail=0
echo ""
for port in 8800 8801 8802; do
    label="${LABELS[$port]}"
    if curl -s --max-time 5 "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${RESET} ${BOLD}${label}${RESET} (port ${port}) — ready"
        ((ok++))
    else
        echo -e "  ${RED}✗${RESET} ${BOLD}${label}${RESET} (port ${port}) — not responding (check /tmp/mlx-server-${port}.log)"
        ((fail++))
    fi
done

echo ""
echo "─────────────────────────────────────"
if [[ $fail -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All ${ok} servers ready.${RESET} Run the notebook!"
else
    echo -e "${YELLOW}${BOLD}${ok}/${((ok+fail))} servers ready.${RESET} Check logs for failures."
fi
