#!/usr/bin/env bash

# stop_servers.sh — Stop all running MLX servers on ports 8800-8809

GREEN='\033[92m'
RED='\033[91m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${BOLD}MLX Servers — Stopping${RESET}"
echo "─────────────────────────────────────"

stopped=0
for port in $(seq 8800 8809); do
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        echo -e "  ${GREEN}✓${RESET} Port ${port} — stopped"
        stopped=$((stopped + 1))
    fi
done

echo ""
if [[ $stopped -eq 0 ]]; then
    echo -e "${BOLD}No running servers found on ports 8800-8809.${RESET}"
else
    echo -e "${BOLD}Stopped ${stopped} server(s).${RESET}"
fi
