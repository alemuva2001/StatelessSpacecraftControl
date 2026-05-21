"""
SpikingSatelliteController — Event-based attitude controller.

Architecture:
  Event frame [2, H, W]  (ON channel + OFF channel)
        ↓
  Conv2d(2→16, k=5, s=2) + BN + LIF (β learnable ≈0.9)
        ↓
  Conv2d(16→32, k=3, s=2) + BN + LIF (β learnable ≈0.9)
        ↓
  AdaptiveAvgPool2d(4×4)  ← avg = local firing rate
        ↓
  Linear(512→32) + ReLU
        ↓
  GRUCell(32→64)  ← long-term integrator
        ↓
  ┌─────┴────────┐
fc(64→3)      fc(64→3)
torque       error_est  ← auxiliary head (training only, ignored at inference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import numpy as np

# Sensor resolution from Simulation.m:  W = 1024/2 = 512,  H = 1024/2 = 256
SENSOR_H = 1024/4
SENSOR_W = 1024/4


def _preprocess_frame(arr: np.ndarray, input_size: int, device) -> torch.Tensor:
    """
    Normalise a raw event array coming from MATLAB/Simulink or the dataset.

    Accepted input shapes (all handled identically):
        [H, W, 2]  — Simulink sends [256, 512, 2]
        [2, H, W]  — already channel-first
        [H, W]     — signed 2-D frame (ON>0, OFF<0)
        Any of the above with leading size-1 batch dimensions added by MATLAB

    Returns:
        torch.Tensor  [1, 2, input_size, input_size]  on *device*
    """
    arr = np.squeeze(arr.astype(np.float32))          # drop MATLAB batch dims

    if arr.ndim == 3 and arr.shape[2] == 2:           # [H, W, 2] → [2, H, W]
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 3 and arr.shape[0] == 2:         # already [2, H, W]
        pass
    elif arr.ndim == 2:                                # signed 2-D → split ON/OFF
        on  = (arr > 0).astype(np.float32)
        off = (arr < 0).astype(np.float32)
        arr = np.stack([on, off])
    else:
        raise ValueError(f"Unexpected event matrix shape: {arr.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device)
    # tensor: [1, 2, H, W]

    _, _, H, W = tensor.shape
    if H != input_size or W != input_size:
        tensor = F.interpolate(
            tensor,
            size=(input_size, input_size),   # square target — consistent with training
            mode='area'
        )
    return tensor


class SpikingSatelliteController(nn.Module):
    """
    Hybrid SNN+GRU attitude controller for event-based star trackers.

    The SNN convolutional layers exploit spike sparsity for efficient
    visual feature extraction. The GRUCell provides long-term memory
    required to integrate attitude error over maneuver timescales (50-70s),
    which far exceeds the effective LIF memory (τ ≈ 0.95s at β=0.9, dt=0.1s).
    """

    def __init__(self, num_steps: int = 4, hidden_gru: int = 128,
                 learn_beta: bool = True):
        """
        Args:
            num_steps:  Internal rate-coding steps per DVS frame.
                        Approximates DVS stochasticity lost due to fixed timestep.
            hidden_gru: GRU hidden size. 64 adds ~10K params — sufficient for
                        integrating attitude error over 700-frame trajectories.
            learn_beta: Allow LIF time constants to be learned per layer.
                        Convolutional and decision layers have different dynamics.
        """
        super().__init__()
        self.num_steps  = num_steps
        self.hidden_gru = hidden_gru
        spike_grad      = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
        #self.bn1   = nn.BatchNorm2d(16)
        self.lif1  = snn.Leaky(beta=0.85, threshold=0.5, spike_grad=spike_grad,
                                learn_beta=learn_beta)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        #self.bn2   = nn.BatchNorm2d(32)
        self.lif2  = snn.Leaky(beta=0.7, threshold=0.5, spike_grad=spike_grad,
                                learn_beta=learn_beta)

        self.pool    = nn.AdaptiveMaxPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.gru = nn.GRUCell(input_size=128+3, hidden_size=hidden_gru)

        self.fc_torque    = nn.Linear(hidden_gru, 3)
        nn.init.uniform_(self.fc_torque.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc_torque.bias)

    def init_state(self, batch_size: int = 1, device=None):
        device = device or next(self.parameters()).device
        return {
            'mem1':  self.lif1.init_leaky(),
            'mem2':  self.lif2.init_leaky(),
            'h_gru': torch.zeros(batch_size, self.gru.hidden_size, device=device),
            'prev_torque': torch.zeros(batch_size, 3, device=device)
        }

    def forward(self, x, state):
        mem1  = state['mem1']
        mem2  = state['mem2']
        h_gru = state['h_gru']
        prev_torque = state["prev_torque"]

        x = x * 8.0

        # Acumulador para no perder el 75% de los spikes
        spk2_sum = 0.0 

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk2_sum = spk2_sum + spk2
        
        # 3. Pasamos la SUMA TOTAL de la actividad neuronal al GRU
        visual = self.proj(self.flatten(self.pool(spk2_sum)))
        gru_input = torch.cat([visual, prev_torque], dim=-1)
        h_gru  = self.gru(gru_input, h_gru)
        torque    = self.fc_torque(h_gru)

        new_state = {'mem1'       : mem1, 
                     'mem2'       : mem2, 
                     'h_gru'      : h_gru,
                     'prev_torque': torque.detach(),
                     }

        return torque, new_state


class SatelliteInferenceInterface:
    """
    Bridge between MATLAB/Simulink and the PyTorch model.
    Accepts any supported event-matrix format (see _preprocess_frame).
    """

    def __init__(self, model_path: str = None, num_steps: int = 4,
                 hidden_gru: int = 128, input_size: int = 64):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model = SpikingSatelliteController(
            num_steps=num_steps,
            hidden_gru=hidden_gru,
        ).to(self.device)

        if model_path is not None:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )

        self.model.eval()
        self.reset_memory()

    def reset_memory(self):
        self.state = self.model.init_state(batch_size=1, device=self.device)

    def calculate_torque(self, raw_event_matrix: np.ndarray) -> list:
        """
        Process a single DVS frame and return 3-axis torque command.

        Args:
            raw_event_matrix: array in any format accepted by _preprocess_frame
        Returns:
            [Tx, Ty, Tz] torque command as Python list (unnormalised)
        """
        tensor_in = _preprocess_frame(raw_event_matrix, self.input_size, self.device)

        with torch.no_grad():
            torque_out, self.state = self.model(tensor_in, self.state)

        return torque_out.squeeze().cpu().tolist()


# Global instance for MATLAB:
#   py.snn_controller.controller_interface.calculate_torque(event_matrix)
controller_interface = SatelliteInferenceInterface()