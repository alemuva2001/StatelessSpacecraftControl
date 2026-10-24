"""
snn_inference.py
Inference script — SNN Satellite Controller
Connects the trained model to MATLAB/Simulink via Python Engine.

Usage from MATLAB:
  % Initialise once at simulation start
  py.importlib.import_module('snn_inference');
  py.snn_inference.initialize('../SNN/Derivative/Weights/best_model.pth');

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

_SNN_DIR = os.path.dirname(os.path.realpath(__file__))
if _SNN_DIR not in sys.path:
    sys.path.insert(0, _SNN_DIR)

from snn_controller import SpikingSatelliteController, _preprocess_frame

# =============================================================================
# CONFIGURATION — must match supervised_train.py
# =============================================================================
CONFIG = {
    'num_steps':           4,
    'hidden_gru':          128,
    'input_size':          256,       # must match supervised_train_NVIDIA.py
    'use_output_filter':   False,     # the new controller already has recurrent memory
    'alpha_min':           0.08,      # Suavizado fuerte para colas de frenado y reposo (elimina parpadeo)
    'alpha_max':           0.45,      # Reacción ágil para picos grandes (sin retraso)
    'alpha_t_ref':         0.25,      # Fracción del pico a partir de la cual se usa alpha_max
    'zero_thresh_ratio':   0.005,     # 0.5% del pico: corte microscópico para reposo estricto sin dientes de sierra
    # Normalizacion correspondiente a SNN/Derivative/Weights/best_model.pth (fuera de subcarpetas).
    'data_dir':            os.path.abspath(os.path.join(_SNN_DIR, '../../Data_D')),
    'stats_filename':      'dataset_stats.npz',
    'weights_dir':         os.path.join(_SNN_DIR, 'Weights'),
}
# =============================================================================

_model  = None
_state  = None
_stats  = None
_device = None
_last_torque = None


def initialize(model_path: str = None):
    """
    Load model and normalisation statistics.
    Call ONCE at the start of the simulation.
    """
    global _model, _state, _stats, _device

    _device = torch.device('cpu')

    # Resolver rutas relativas tanto si se ejecuta desde Simulation/, SNN/ o raíz
    data_dir = CONFIG['data_dir']
    if not os.path.exists(data_dir):
        for candidate in [os.path.normpath(os.path.join(_SNN_DIR, data_dir)),
                          os.path.join(_SNN_DIR, 'Weights', 'finetune_ideal'),
                          os.path.abspath(os.path.join(_SNN_DIR, '../../Data'))]:
            if os.path.exists(os.path.join(candidate, CONFIG['stats_filename'])):
                data_dir = candidate
                break

    stats_path = os.path.join(data_dir, CONFIG['stats_filename'])
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"No se encontró '{stats_path}'.\n"
            f"Selecciona las estadisticas correspondientes a los pesos cargados."
        )

    s      = np.load(stats_path)
    _stats = {k: torch.tensor(s[k], dtype=torch.float32) for k in s.files}
    print(f"[SNN] Stats cargadas desde {stats_path}")
    print(f"[SNN]   torque_mean = {_stats['torque_mean'].numpy()}")
    print(f"[SNN]   torque_peak = {_stats['torque_peak'].numpy()}")

    # Cargar pesos del modelo
    if model_path is None:
        weights_dir = CONFIG['weights_dir']
        if not os.path.exists(weights_dir):
            alt_w = os.path.normpath(os.path.join(_SNN_DIR, weights_dir))
            if os.path.exists(alt_w):
                weights_dir = alt_w
        model_path = os.path.join(weights_dir, 'best_model.pth')

    _model = SpikingSatelliteController(
        num_steps=CONFIG['num_steps'],
        hidden_gru=CONFIG['hidden_gru'],
        learn_beta=True,
    ).to(_device)

    try:
        state_dict = torch.load(model_path, map_location=_device)
        
        # Limpiamos el prefijo '_orig_mod.' si el modelo fue entrenado con torch.compile()
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                clean_state_dict[k.replace('_orig_mod.', '')] = v
            else:
                clean_state_dict[k] = v
                
        _model.load_state_dict(clean_state_dict)
        print(f"[SNN] Modelo cargado desde {model_path}")
    except FileNotFoundError:
        print(f"\n[WARNING] No se encontró el modelo en {model_path}.")
        print("[WARNING] Simulink se ha inicializado con pesos aleatorios. Ignora esto si estás generando datos (USE_SNN=0).")
    except RuntimeError as exc:
        raise RuntimeError(
            "[SNN] El checkpoint no coincide con la arquitectura temporal actual. "
            "Reentrena con supervised_train.py o supervised_train_NVIDIA.py para generar "
            "un nuevo best_model.pth."
        ) from exc

    _model.eval()

    reset()
    print("[SNN] Listo para inferencia.")


def reset():
    """Reset internal SNN state. Call at the start of each episode."""
    global _state, _last_torque
    if _model is None:
        raise RuntimeError("Llama a initialize() antes de reset().")
    _state = _model.init_state(batch_size=1, device=_device)
    _last_torque = None
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
    global _state, _last_torque

    if _model is None or _state is None:
        raise RuntimeError("Llama a initialize() antes de step().")

    try:
        arr = np.asarray(memoryview(raw_event_matrix), dtype=np.float32)
    except Exception:
        try:
            arr = np.array(raw_event_matrix, dtype=np.float32)
        except Exception:
            arr = np.array(list(raw_event_matrix), dtype=np.float32)

    # _preprocess_frame handles: squeeze, shape detection, H×W→square interpolation
    tensor_in = _preprocess_frame(arr, CONFIG['input_size'], _device)

    with torch.no_grad():
        torque_norm, _state, metrics = _model(tensor_in, _state)

    torque_real = (torque_norm.squeeze().cpu()
               * _stats['torque_peak']
               + _stats['torque_mean'])
    torque_real_np = torque_real.numpy().copy()

    if CONFIG.get('use_output_filter', False):
        # --- FILTRO EMA ADAPTATIVO (AMPLITUD-DEPENDIENTE) ---
        # Disponible para comparativas, pero desactivado por defecto porque
        # puede introducir retraso sobre una arquitectura ya recurrente.
        alpha_min = CONFIG.get('alpha_min', 0.08)
        alpha_max = CONFIG.get('alpha_max', 0.45)
        t_ref_ratio = CONFIG.get('alpha_t_ref', 0.25)
        zero_ratio = CONFIG.get('zero_thresh_ratio', 0.005)
        peak_np = _stats['torque_peak'].numpy()

        if _last_torque is None:
            _last_torque = torque_real_np.copy()

        for i in range(3):
            mag = abs(torque_real_np[i]) / peak_np[i]
            scale = min(1.0, mag / t_ref_ratio)
            alpha_i = alpha_min + (alpha_max - alpha_min) * scale

            _last_torque[i] = alpha_i * torque_real_np[i] + (1.0 - alpha_i) * _last_torque[i]

            if abs(_last_torque[i]) < zero_ratio * peak_np[i]:
                _last_torque[i] = 0.0
    else:
        _last_torque = torque_real_np

    return _last_torque.tolist() + [
        metrics.get('fr_layer1', 0.0), 
        metrics.get('fr_layer2', 0.0), 
        metrics.get('fr_layer3', 0.0), 
        metrics.get('gru_abs', 0.0)
    ]

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
