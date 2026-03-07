#!/usr/bin/env bash
# Install the homebrew-py3 Jupyter kernel with correct PATH env.
# Run on any machine where llm-lab notebooks will execute.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/kernel-spec/kernel.json"
KERNEL_DEST="$HOME/Library/Jupyter/kernels/homebrew-py3"

# Verify python3.14 exists
PYTHON="/opt/homebrew/opt/python@3.14/bin/python3.14"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found. Install: brew install python@3.14"
    exit 1
fi

# Verify required packages
echo "Checking required packages..."
MISSING=()
for pkg in openai psutil markdown mlx ipykernel; do
    if ! "$PYTHON" -c "import $pkg" 2>/dev/null; then
        MISSING+=("$pkg")
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing packages: ${MISSING[*]}"
    echo "Install: $PYTHON -m pip install ${MISSING[*]}"
    exit 1
fi

# Install kernel spec
mkdir -p "$KERNEL_DEST"
cp "$KERNEL_SRC" "$KERNEL_DEST/kernel.json"
echo "Installed kernel spec to $KERNEL_DEST/kernel.json"

# Verify
echo "Verifying..."
"$PYTHON" -m jupyter kernelspec list 2>/dev/null | grep -q homebrew-py3 && \
    echo "OK: homebrew-py3 kernel registered" || \
    echo "WARN: kernel installed but not showing in kernelspec list"

echo "Done. Restart any running Jupyter kernels to pick up changes."
