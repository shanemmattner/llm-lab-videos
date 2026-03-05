#!/usr/bin/env bash

# stop_servers.sh — Stop all 3 MLX arena servers

GREEN='\033[92m'
RED='\033[91m'
BOLD='\033[1m'
RESET='\033[0m'

declare -A LABELS=(
    [8800]="122B"
    [8801]="35B"
    [8802]="0.8B"
)

echo -e "${BOLD}MLX 3-Model Arena — Stopping Servers${RESET}"
echo "─────────────────────────────────────"

for port in 8800 8801 8802; do
    label="${LABELS[$port]}"
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        echo -e "  ${GREEN}✓${RESET} ${BOLD}${label}${RESET} (port ${port}) — stopped"
    else
        echo -e "  ${RED}—${RESET} ${BOLD}${label}${RESET} (port ${port}) — not running"
    fi
done

echo ""
echo -e "${BOLD}All servers stopped.${RESET}"
