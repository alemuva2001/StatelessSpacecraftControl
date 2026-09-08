"""
frame2events.py — Conversor frame->eventos usando la librería V2E para Simulink.

Convierte frames de video en escala de grises (uint8) a un frame de eventos
con dos canales: ON (incremento de luz) y OFF (decremento de luz).
Utiliza la librería oficial V2E (v2ecore.emulator.EventEmulator).

Uso desde MATLAB / Simulink:
    % InitFcn callback:
    py.importlib.reload(py.importlib.import_module('frame2events'));
    py.frame2events.reset();

    % En cada paso:
    events = double(py.frame2events.process(frame));
    % events(:,:,1) = ON,  events(:,:,2) = OFF
"""

import os
import sys
import numpy as np
import array

# Configurar variable de entorno para evitar conflictos de OpenMP (crítico en macOS/MATLAB)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from v2ecore.emulator import EventEmulator

# =====================================================================
#                         CONFIGURACION V2E
# =====================================================================
POS_THRES = 0.1              # Umbral ON (log-luminancia) (0.05 para máxima sensibilidad)
NEG_THRES = 0.1              # Umbral OFF (log-luminancia) (0.05 para máxima sensibilidad)
SIGMA_THRES = 0.00            # Desviación estándar del umbral (mismatch)
CUTOFF_HZ = 0.0               # Frecuencia de corte paso bajo del fotorreceptor (0.0 = respuesta instantánea)
LEAK_RATE_HZ = 0.0            # Tasa de eventos de fuga por píxel en Hz (0.0 = sin fugas)
REFRACTORY_PERIOD_S = 0.0     # Período refractario del píxel en segundos (0.0 = sin periodo refractario)
SHOT_NOISE_RATE_HZ = 0.0      # Tasa de ruido térmico (shot noise) por píxel en Hz (0.0 = sin ruido)
DEVICE = "cpu"                # Dispositivo de cómputo ('cpu' o 'cuda')
DT = 0.01                     # Paso de tiempo de simulación (dt de Simulink, por defecto 0.1s)
# =====================================================================


def _to_numpy_2d(frame):
    """Convierte cualquier entrada de MATLAB a numpy 2D float64."""
    try:
        arr = np.array(frame, dtype=np.float64)
    except Exception:
        arr = np.array(list(frame), dtype=np.float64)

    if arr.ndim == 1:
        n = arr.size
        side = int(np.sqrt(n))
        if side * side == n:
            arr = arr.reshape(side, side)
        else:
            raise ValueError(f"frame2events: frame 1D con {n} elementos, no es cuadrado")

    if arr.ndim != 2:
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:,:,0]
        else:
            raise ValueError(f"frame2events: forma inesperada {arr.shape}")

    return arr


class Frame2EventConverter:
    """
    Conversor de frames a eventos utilizando la librería oficial V2E.
    Mantiene el estado interno persistente del emulador.
    """

    def __init__(self, pos_thres=POS_THRES, neg_thres=NEG_THRES, dt=DT):
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.dt = dt
        self.emulator = None
        self._call_count = 0

    def reset(self):
        """Reinicia el estado interno del emulador."""
        self.emulator = None
        self._call_count = 0
        print("[frame2events] Reset OK (V2E Emulator)", file=sys.stderr)

    def process(self, frame):
        """
        Procesa un frame uint8 [H,W] usando EventEmulator de V2E.
        Retorna la matriz de eventos aplanada en formato lista.
        """
        self._call_count += 1
        arr = _to_numpy_2d(frame)
        H, W = arr.shape

        if self.emulator is not None:
            if self.emulator.output_width != W or self.emulator.output_height != H:
                print(f"[frame2events] Cambio de resolucion detectado: "
                      f"{self.emulator.output_height}x{self.emulator.output_width} -> {H}x{W}. "
                      f"Reiniciando emulador...", file=sys.stderr)
                self.reset()
                self._call_count = 1

        if self.emulator is None:
            # Inicialización perezosa con las dimensiones dinámicas del frame
            self.emulator = EventEmulator(
                pos_thres=self.pos_thres,
                neg_thres=self.neg_thres,
                sigma_thres=SIGMA_THRES,
                cutoff_hz=CUTOFF_HZ,
                leak_rate_hz=LEAK_RATE_HZ,
                refractory_period_s=REFRACTORY_PERIOD_S,
                shot_noise_rate_hz=SHOT_NOISE_RATE_HZ,
                device=DEVICE,
                output_width=W,
                output_height=H
            )
            # En la primera llamada, V2E registra el baseline inicial (primer frame)
            # y no produce eventos. Retorna una matriz vacía para mantener la consistencia.
            t_frame = 0.0
            _ = self.emulator.generate_events(arr, t_frame)
            events_matrix = np.zeros((H, W, 2), dtype=np.float64)
            return array.array('d', events_matrix.flatten(order='F'))

        # Tiempo de simulación (t = (k - 1) * dt)
        t_frame = (self._call_count - 1) * self.dt

        # Ejecución del emulador oficial V2E
        events = self.emulator.generate_events(arr, t_frame)

        # Reconstrucción de la matriz [H, W, 2] esperada por Simulink
        events_matrix = np.zeros((H, W, 2), dtype=np.float64)

        if events is not None and len(events) > 0:
            xs = events[:, 1].astype(np.int32)
            ys = events[:, 2].astype(np.int32)
            pols = events[:, 3]

            # Filtrado por si algún índice queda fuera de rango
            valid_mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            on_mask = (pols > 0) & valid_mask
            off_mask = (pols < 0) & valid_mask

            # Suma de eventos por píxel de manera vectorizada y unbuffered
            np.add.at(events_matrix[:, :, 0], (ys[on_mask], xs[on_mask]), 1.0)
            np.add.at(events_matrix[:, :, 1], (ys[off_mask], xs[off_mask]), 1.0)

        # Aplanar en orden Fortran (column-major) y conversión a array.array para MATLAB
        return array.array('d', events_matrix.flatten(order='F'))


# =====================================================================
#   API para MATLAB/Simulink
# =====================================================================
_converter = Frame2EventConverter()


def reset():
    """Reinicia el conversor global. Llamar al inicio de cada simulacion."""
    _converter.reset()


def process(frame):
    """Procesa un frame uint8 [H,W] y devuelve eventos [H,W] usando V2E."""
    return _converter.process(frame)
