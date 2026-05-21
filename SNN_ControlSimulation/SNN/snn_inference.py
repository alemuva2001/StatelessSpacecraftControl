"""
snn_inference.py
Inference script — SNN Satellite Controller
Connects the trained model to MATLAB/Simulink via Python Engine.

Usage from MATLAB:
  % Initialise once at simulation start
  py.importlib.import_module('snn_inference');
  py.snn_inference.initialize('../SNN/Weights/best_model.pth');

  % Each simulation step (Simulink MATLAB Function block)
  torque = py.snn_inference.step(event_matrix);   % event_matrix: [H, W, 2]
  Tx = torque{1}; Ty = torque{2}; Tz = torque{3};

  % End of simulation
  py.snn_inference.reset();

Event matrix format from Simulink:
  [H, W, 2]  = [256, 512, 2]  (sensor resolution: H=1024/4, W=1024/2)
  Channel 0 = ON events, Channel 1 = OFF events
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

_SNN_DIR = os.path.dirname(os.path.abspath(__file__))
if _SNN_DIR not in sys.path:
    sys.path.insert(0, _SNN_DIR)

from snn_controller import SpikingSatelliteController, _preprocess_frame

# =============================================================================
# CONFIGURATION — must match supervised_train.py
# =============================================================================
CONFIG = {
    'num_steps':   4,
    'hidden_gru':  128,
    'input_size':  64,        # square target resolution
    'data_dir':    '../Data',
    'weights_dir': '../SNN/Weights',
}
# =============================================================================

_model  = None
_state  = None
_stats  = None
_device = None


def initialize(model_path: str = None):
    """
    Load model and normalisation statistics.
    Call ONCE at the start of the simulation.
    """
    global _model, _state, _stats, _device

    _device = torch.device('cpu')

    stats_path = os.path.join(CONFIG['data_dir'], 'dataset_stats.npz')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"No se encontró '{stats_path}'.\n"
            f"Ejecuta primero: python3 compute_dataset_stats.py"
        )

    s      = np.load(stats_path)
    _stats = {k: torch.tensor(s[k], dtype=torch.float32) for k in s.files}
    print(f"[SNN] Stats cargadas desde {stats_path}")
    print(f"[SNN]   torque_mean = {_stats['torque_mean'].numpy()}")
    print(f"[SNN]   torque_peak = {_stats['torque_peak'].numpy()}")

    if model_path is None:
        model_path = os.path.join(CONFIG['weights_dir'], 'best_model.pth')

    _model = SpikingSatelliteController(
        num_steps=CONFIG['num_steps'],
        hidden_gru=CONFIG['hidden_gru'],
        learn_beta=True,
    ).to(_device)

    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.eval()
    print(f"[SNN] Modelo cargado desde {model_path}")

    reset()
    print("[SNN] Listo para inferencia.")


def reset():
    """Reset internal SNN state (LIF memories + GRU). Call at the start of each episode."""
    global _state
    if _model is None:
        raise RuntimeError("Llama a initialize() antes de reset().")
    _state = _model.init_state(batch_size=1, device=_device)
    print("[SNN] Estado interno reiniciado.")


def step(raw_event_matrix) -> list:
    """
    Process one event frame and return 3-axis torque command.

    Args:
        raw_event_matrix: array [H, W, 2] from Simulink (or any format
                          accepted by _preprocess_frame in snn_controller.py)
    Returns:
        [Tx, Ty, Tz]  in Nm
    """
    global _state

    if _model is None or _state is None:
        raise RuntimeError("Llama a initialize() antes de step().")

    arr = np.array(raw_event_matrix, dtype=np.float32)

    #print(f"[DEBUG SNN_INFERENCE] Shape recibido de Simulink: {arr.shape}     ", end='\r')

    # _preprocess_frame handles: squeeze, shape detection, H×W→square interpolation
    tensor_in = _preprocess_frame(arr, CONFIG['input_size'], _device)

    with torch.no_grad():
        torque_norm, _state = _model(tensor_in, _state)

    torque_real = (torque_norm.squeeze().cpu()
               * _stats['torque_peak']
               + _stats['torque_mean'])
    return torque_real.tolist()


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == '__main__':
    print("=== Test de inferencia ===")
    initialize()

    # Empty frame — torque should be ≈ 0
    dummy_frame = np.zeros((256, 512, 2), dtype=np.float32)
    torque = step(dummy_frame)
    print(f"\nFrame vacío → torque: {[f'{v:.6f}' for v in torque]} Nm")

    # Random sparse events
    np.random.seed(42)
    rand_on  = np.random.binomial(1, 0.05, (256, 512)).astype(np.float32)
    rand_off = np.random.binomial(1, 0.05, (256, 512)).astype(np.float32)
    random_frame = np.stack([rand_on, rand_off], axis=-1)   # [256, 512, 2]
    torque2 = step(random_frame)
    print(f"Frame aleatorio → torque: {[f'{v:.6f}' for v in torque2]} Nm")


    print("\n=== Test completado ===")
