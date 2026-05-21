#!/bin/zsh
# =============================================================================
# setup.sh — SNN Satellite Controller: Environment Setup
# =============================================================================
# Usage: source setup.sh
#
# This script:
#   1. Activates the Python virtual environment (.venv)
#   2. Verifies PyTorch and snntorch are installed
#   3. Prints the pyenv command to run once in MATLAB
#   4. Opens MATLAB
# =============================================================================

# --- CONFIGURATION (edit only this section) ----------------------------------

# Root of your project (where .venv lives)
PROJECT_ROOT="/Users/alejandromunozvazquez/Documents/Universidad/MII/POLIMI/2nd Course/Thesis/Code/StatelessSpacecraftControl/SNN_ControlSimulation"

# Folder containing snn_controller.py
SNN_MODULE_DIR="$PROJECT_ROOT/SNN/Networks"

# Your MATLAB version (run: ls /Applications | grep MATLAB to check)
MATLAB_APP="/Applications/MATLAB_R2024b.app"

# -----------------------------------------------------------------------------

echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│     SNN Satellite Controller — Setup        │"
echo "└─────────────────────────────────────────────┘"

# 1. Activate virtual environment
echo ""
echo "[1/3] Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate" || {
    echo "      ERROR: .venv not found at $PROJECT_ROOT"
    echo "      Fix:  python3 -m venv $PROJECT_ROOT/.venv"
    echo "            pip install torch snntorch"
    return 1
}
echo "      OK — $(which python3)"

# 2. Verify dependencies (python3 explicit — avoids any 'python' alias conflict)
echo ""
echo "[2/3] Checking dependencies..."
python3 -c "
import torch, snntorch
print(f'      PyTorch  {torch.__version__}')
print(f'      snntorch {snntorch.__version__}')
" || {
    echo "      ERROR: Missing packages."
    echo "      Fix:  pip install torch snntorch"
    return 1
}
echo "      OK"

# 3. Print the MATLAB setup command (must be run once per MATLAB session)
PYTHON_EXEC=$(which python3)
echo ""
echo "[3/3] Environment ready."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Run this ONCE in the MATLAB Command Window:"
echo ""
echo "  pyenv('Version', '$PYTHON_EXEC')"
echo ""
echo "  And this in the InitFcn callback:"
echo ""
echo "  insert(py.sys.path, int32(0), '$SNN_MODULE_DIR');"
echo "  py.importlib.reload(py.importlib.import_module('snn_controller'));"
echo "  py.snn_controller.controller_interface.reset_memory();"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 4. Open MATLAB
echo ""
echo "[>>] Launching MATLAB..."

if [ -d "$MATLAB_APP" ]; then
    open "$MATLAB_APP"
else
    # Auto-detect any installed MATLAB version
    MATLAB_FOUND=$(ls /Applications/ 2>/dev/null | grep -i "^MATLAB" | head -1)
    if [ -n "$MATLAB_FOUND" ]; then
        echo "     Found: $MATLAB_FOUND"
        open "/Applications/$MATLAB_FOUND"
    else
        echo "     WARNING: MATLAB not found. Open it manually."
    fi
fi

echo ""
echo "[OK] Done. MATLAB is opening in the background."
echo ""