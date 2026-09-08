#!/bin/bash
# =============================================================================
# setup_NVIDIA.sh — SNN Satellite Controller: Environment Setup (CUDA)
# =============================================================================
# Uso: source setup_NVIDIA.sh (en Linux/Mac) o bash setup_NVIDIA.sh
# =============================================================================

echo "┌─────────────────────────────────────────────┐"
echo "│  SNN Satellite Controller — NVIDIA Setup    │"
echo "└─────────────────────────────────────────────┘"

# 1. Crear el entorno virtual si no existe
if [ ! -d ".venv_nvidia" ]; then
    echo "[1/3] Creando entorno virtual (.venv_nvidia)..."
    
    # Si estamos en Ubuntu/Debian, instalamos el paquete venv automáticamente
    if command -v apt &> /dev/null; then
        echo "Detectado sistema Linux. Instalando dependencias del sistema (puede pedir contraseña)..."
        sudo apt update && sudo apt install python3-venv python3.10-venv -y
    fi

    # Intenta usar python3, python, o py (Windows)
    python3 -m venv .venv_nvidia || python -m venv .venv_nvidia || py -m venv .venv_nvidia
else
    echo "[1/3] Entorno virtual .venv_nvidia ya existe."
fi

# 2. Activar el entorno virtual (Soporte cruzado Linux / Git Bash Windows)
echo "[2/3] Activando entorno..."
source .venv_nvidia/bin/activate 2>/dev/null || source .venv_nvidia/Scripts/activate

# 3. Instalar dependencias con soporte CUDA 12.1
echo "[3/3] Instalando dependencias (PyTorch CUDA)..."
python3 -m pip install --upgrade pip
python3 -m pip install numpy h5py scipy matplotlib tqdm snntorch
python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[OK] ¡Entorno NVIDIA Listo y Activado!"
echo "Para arrancar el entrenamiento, ejecuta:"
echo "python supervised_train_NVIDIA.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

