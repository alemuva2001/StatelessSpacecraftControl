"""
SpikingSatelliteController - Event-based attitude controller.

End-to-end supervised training via surrogate gradient BPTT.

Architecture (for 256x256 input):
  Event frame [2, 256, 256]  (ON + OFF channel)
        |
  Conv2d(2->8,  k=3, s=1)   -> [8,  256, 256]  <- no stride: preserves 3x3px stars
  LIF1 (beta=0.9)
        |
  Conv2d(8->16, k=3, s=2)   -> [16, 128, 128]  <- 1st downsampling
  LIF2 (beta=0.85)
        |
  Conv2d(16->32, k=3, s=2)  -> [32,  64,  64]  <- 2nd downsampling
  LIF3 (beta=0.8)
        |
  AdaptiveAvgPool2d(4x4)    -> [32,   4,   4]  <- mean firing rate per region
  Flatten                   -> [512]
        |
  Linear(512->128) + LayerNorm + ReLU
        |
  GRUCell(131->128)  <- temporal integrator (128 visual + 3 prev_torque)
        |
  fc(128->3) -> torque

Camera: 512x512. Dataset saved at 256x256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import numpy as np

SENSOR_H = 256
SENSOR_W = 256


def _preprocess_frame(arr: np.ndarray, input_size: int, device) -> torch.Tensor:
    """
    Normalise a raw event array from MATLAB/Simulink or the dataset.

    Accepted shapes:
        [H, W, 2]  -- Simulink format
        [2, H, W]  -- channel-first
        [H, W]     -- signed 2-D (ON>0, OFF<0)
        Any of the above with leading size-1 batch dims from MATLAB

    Returns:
        torch.Tensor [1, 2, input_size, input_size] on *device*
    """
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
        tensor = F.interpolate(tensor, size=(input_size, input_size), mode='area')
    return tensor


class SpikingSatelliteController(nn.Module):
    """
    Hybrid SNN+GRU attitude controller for event-based star trackers.

    Design rationale:
    - Conv1 has NO stride: preserves 3x3px star clusters at full resolution.
    - Three conv layers with gradual downsampling (x1, x2, x2).
    - AdaptiveAvgPool computes mean firing rate per spatial region.
    - GRUCell integrates firing-rate map over time (maneuver scale >> LIF tau).
    - ~200K parameters, balanced across all layers.
    """

    def __init__(self, num_steps: int = 4, hidden_gru: int = 128,
                 learn_beta: bool = True):
        super().__init__()
        self.num_steps  = num_steps
        self.hidden_gru = hidden_gru
        spike_grad      = surrogate.fast_sigmoid(slope=25)

        # Conv1
        self.conv1 = nn.Conv2d(2, 8, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(8)
        self.lif1  = snn.Leaky(beta=0.9, threshold=0.1, spike_grad=spike_grad, learn_beta=learn_beta)

        # Conv2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.lif2  = snn.Leaky(beta=0.85, threshold=0.1, spike_grad=spike_grad, learn_beta=learn_beta)

        # Conv3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(32)
        self.lif3  = snn.Leaky(beta=0.8, threshold=0.1, spike_grad=spike_grad, learn_beta=learn_beta)

        # AvgPool -> 32 * 8 * 8 = 2048 values
        self.pool    = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        
        # --- Output Head ---
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        # Initialize output layer
        nn.init.uniform_(self.fc[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.fc[-1].bias)

    def init_state(self, batch_size: int = 1, device=None):
        device = device or next(self.parameters()).device
        return {
            'mem1': self.lif1.init_leaky(),
            'mem2': self.lif2.init_leaky(),
            'mem3': self.lif3.init_leaky(),
        }

    def forward(self, x, state):
        mem1 = state['mem1']
        mem2 = state['mem2']
        mem3 = state['mem3']

        # Rate coding: accumulate spikes over num_steps (no x*8 saturation)
        spk1_sum = 0.0
        spk2_sum = 0.0
        spk3_sum = 0.0

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.bn1(self.conv1(x)), mem1)
            spk2, mem2 = self.lif2(self.bn2(self.conv2(spk1)), mem2)
            spk3, mem3 = self.lif3(self.bn3(self.conv3(spk2)), mem3)
            spk1_sum = spk1_sum + spk1
            spk2_sum = spk2_sum + spk2
            spk3_sum = spk3_sum + spk3

        visual = self.flatten(self.pool(spk3_sum))
        torque = self.fc(visual)

        new_state = {
            'mem1': mem1,
            'mem2': mem2,
            'mem3': mem3,
        }

        metrics = {
            'fr_layer1':     (spk1_sum.mean() / self.num_steps).item(),
            'fr_layer2':     (spk2_sum.mean() / self.num_steps).item(),
            'fr_layer3':     (spk3_sum.mean() / self.num_steps).item(),
            'active_layer1': (spk1_sum > 0).float().mean().item(),
            'active_layer2': (spk2_sum > 0).float().mean().item(),
            'active_layer3': (spk3_sum > 0).float().mean().item(),
        }

        return torque, new_state, metrics


class SatelliteInferenceInterface:
    """Bridge between MATLAB/Simulink and the PyTorch model."""

    def __init__(self, model_path: str = None, num_steps: int = 4,
                 hidden_gru: int = 128, input_size: int = 256,
                 ema_alpha: float = 0.2):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.ema_alpha  = ema_alpha
        self.last_torque = [0.0, 0.0, 0.0]
        
        self.model = SpikingSatelliteController(
            num_steps=num_steps,
            hidden_gru=hidden_gru,
        ).to(self.device)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            clean = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(clean)

        self.model.eval()
        self.reset_memory()

    def reset_memory(self):
        self.state = self.model.init_state(batch_size=1, device=self.device)
        self.last_torque = [0.0, 0.0, 0.0]

    def calculate_torque(self, raw_event_matrix: np.ndarray) -> list:
        tensor_in = _preprocess_frame(raw_event_matrix, self.input_size, self.device)
        with torch.no_grad():
            torque_out, self.state, _ = self.model(tensor_in, self.state)
        
        raw_t = torque_out.squeeze().cpu().tolist()
        
        # Apply EMA filter to smooth out high-frequency SNN noise
        smooth_t = [0.0, 0.0, 0.0]
        for i in range(3):
            smooth_t[i] = self.ema_alpha * raw_t[i] + (1 - self.ema_alpha) * self.last_torque[i]
        
        self.last_torque = smooth_t
        return smooth_t


# Global instance for MATLAB:
#   py.snn_controller.controller_interface.calculate_torque(event_matrix)
controller_interface = SatelliteInferenceInterface()
