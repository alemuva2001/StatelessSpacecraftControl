import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import numpy as np

# Sensor resolution from Simulation.m
SENSOR_H = 1024/4
SENSOR_W = 1024/4

def _preprocess_frame(arr: np.ndarray, input_size: int, device) -> torch.Tensor:
    arr = np.squeeze(arr.astype(np.float32))          

    if arr.ndim == 3 and arr.shape[2] == 2:           
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 3 and arr.shape[0] == 2:         
        pass
    elif arr.ndim == 2:                                
        on  = (arr > 0).astype(np.float32)
        off = (arr < 0).astype(np.float32)
        arr = np.stack([on, off])
    else:
        raise ValueError(f"Unexpected event matrix shape: {arr.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device)

    _, _, H, W = tensor.shape
    if H != input_size or W != input_size:
        tensor = F.interpolate(
            tensor,
            size=(input_size, input_size),   
            mode='area'
        )
    return tensor


class SpikingSatelliteController(nn.Module):
    """
    Arquitectura 100% SNN (Deep Spiking Neural Network) mejorada.
    Incluye una vía recurrente principal (Maniobra rápida) y un subgrupo
    paralelo de 10 Neuronas Integradoras (Fine-Pointing / Banda muerta).
    """

    def __init__(self, num_steps: int = 4, hidden_gru: int = 128,
                 learn_beta: bool = True):
        super().__init__()
        self.num_steps  = num_steps
        self.hidden_size = hidden_gru
        self.num_int_neurons = 10  # <-- NUEVO: 10 Neuronas Integradoras puras
        
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # 1. RETINA VISUAL
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
        self.lif1  = snn.Leaky(beta=0.85, threshold=0.5, spike_grad=spike_grad, learn_beta=learn_beta)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lif2  = snn.Leaky(beta=0.7, threshold=0.5, spike_grad=spike_grad, learn_beta=learn_beta)

        self.pool    = nn.AdaptiveMaxPool2d((8, 8))
        self.flatten = nn.Flatten()

        # 2. PROYECCIÓN SNN
        self.fc_proj = nn.Linear(32 * 8 * 8, hidden_gru)
        self.lif_proj = snn.Leaky(beta=0.85, threshold=0.5, spike_grad=spike_grad, learn_beta=learn_beta)

        # 3A. VÍA 1: MEMORIA RECURRENTE SNN (Dinámica PD rápida)
        self.fc_in  = nn.Linear(hidden_gru + 3, hidden_gru) 
        self.fc_rec = nn.Linear(hidden_gru, hidden_gru)
        self.lif_rec = snn.Leaky(beta=0.95, threshold=0.5, spike_grad=spike_grad, learn_beta=learn_beta)

        # 3B. VÍA 2: SUBGRUPO INTEGRADOR PURE ('I' de PID)
        # Inspirado en el paper: beta=1.0 estricto para evitar olvido a bajas velocidades
        self.fc_int = nn.Linear(hidden_gru + 3, self.num_int_neurons)
        self.lif_int = snn.Leaky(beta=1.0, threshold=1.0, spike_grad=spike_grad, 
                                 learn_beta=False, reset_mechanism="subtract")

        # 4. SALIDA: Decodificación de Voltaje de Membrana
        # Recibe 128 dimensiones de la vía rápida + 10 dimensiones de la vía integradora
        self.fc_out = nn.Linear(hidden_gru + self.num_int_neurons, 3)
        self.lif_out = snn.Leaky(beta=0.98, reset_mechanism="none", learn_beta=learn_beta)

        nn.init.uniform_(self.fc_out.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc_out.bias)

    def init_state(self, batch_size: int = 1, device=None):
        device = device or next(self.parameters()).device
        return {
            'mem1':     self.lif1.init_leaky(),
            'mem2':     self.lif2.init_leaky(),
            'mem_proj': self.lif_proj.init_leaky(),
            'mem_rec':  self.lif_rec.init_leaky(),
            'spk_rec':  torch.zeros(batch_size, self.hidden_size, device=device),
            'mem_int':  self.lif_int.init_leaky(), # <-- NUEVO: Memoria de los integradores
            'mem_out':  self.lif_out.init_leaky(), 
            'prev_torque': torch.zeros(batch_size, 3, device=device)
        }

    def forward(self, x, state):
        mem1     = state['mem1']
        mem2     = state['mem2']
        mem_proj = state['mem_proj']
        mem_rec  = state['mem_rec']
        spk_rec  = state['spk_rec']
        mem_int  = state['mem_int']   # <-- NUEVO
        mem_out  = state['mem_out']
        prev_torque = state['prev_torque']

        x = x * 8.0
        torque_accum = 0.0

        for _ in range(self.num_steps):
            # Fase Visual
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)

            visual_flat = self.flatten(self.pool(spk2))
            spk_proj, mem_proj = self.lif_proj(self.fc_proj(visual_flat), mem_proj)

            # Entrada combinada (visión actual + inercia previa)
            rec_input = torch.cat([spk_proj, prev_torque], dim=-1)
            
            # --- VÍA 1: Cerebro Principal Recurrente ---
            current_rec = self.fc_in(rec_input) + self.fc_rec(spk_rec)
            spk_rec, mem_rec = self.lif_rec(current_rec, mem_rec)

            # --- VÍA 2: Las 10 Neuronas Integradoras Puras ---
            current_int = self.fc_int(rec_input)
            spk_int, mem_int = self.lif_int(current_int, mem_int)

            # --- FUSIÓN ---
            # Unimos los picos de las maniobras rápidas con los picos acumulados del integrador
            out_input = torch.cat([spk_rec, spk_int], dim=-1)
            
            _, mem_out = self.lif_out(self.fc_out(out_input), mem_out)
            torque_accum = torque_accum + mem_out

        torque = torque_accum / self.num_steps

        new_state = {
            'mem1':     mem1, 
            'mem2':     mem2, 
            'mem_proj': mem_proj,
            'mem_rec':  mem_rec,
            'spk_rec':  spk_rec,
            'mem_int':  mem_int,      # <-- NUEVO
            'mem_out':  mem_out,
            'prev_torque': torque.detach(),
        }

        return torque, new_state