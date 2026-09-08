"""
SpikingSatelliteController - event-camera derivative torque controller.

The network is trained to imitate the derivative part of the PD controller:
it predicts T_D directly from event frames. Since T_D is equivalent to an
angular-rate damping command, the model uses explicit recurrent memory:

  Event frame [2, H, W]
        |
  Conv/LIF encoder over a few internal spike steps
        |
  pooled spike-rate + membrane features
        |
  projection -> GRUCell temporal integrator
        |
  temporal torque head + small instantaneous torque head
        |
  T_D [Tx, Ty, Tz]

The instantaneous head gives the model a short path for sharp perturbation
responses, while the GRU carries the temporal information needed to infer
angular rate from event motion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

SENSOR_H = 256
SENSOR_W = 256


def _preprocess_frame(arr: np.ndarray, input_size: int, device) -> torch.Tensor:
    """
    Normalize a raw event array from MATLAB/Simulink or the dataset.

    Accepted shapes:
        [H, W, 2]  -- Simulink/dataset format
        [2, H, W]  -- channel-first
        [H, W]     -- signed 2-D frame (ON>0, OFF<0)
        Any of the above with leading size-1 batch dims from MATLAB

    Returns:
        torch.Tensor [1, 2, input_size, input_size] on *device*
    """
    arr = np.squeeze(arr.astype(np.float32))

    if arr.ndim == 3 and arr.shape[2] == 2:
        # Direct MATLAB input is [H, W, C], unlike reversed HDF5 axes.
        # Move channels first while preserving row/column coordinates.
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 3 and arr.shape[0] == 2:
        pass
    elif arr.ndim == 2:
        on = (arr > 0).astype(np.float32)
        off = (arr < 0).astype(np.float32)
        arr = np.stack([on, off])
    else:
        raise ValueError(f"Unexpected event matrix shape: {arr.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device)

    _, _, height, width = tensor.shape
    if height != input_size or width != input_size:
        tensor = F.interpolate(tensor, size=(input_size, input_size), mode="area")
    return tensor


def _detach_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    return value


class SpikingSatelliteController(nn.Module):
    """
    Hybrid SNN + GRU controller for direct T_D imitation.

    The convolutional LIF stack extracts sparse event-motion evidence from each
    control frame. The GRU integrates that evidence over the maneuver, which is
    the part the previous purely instantaneous heads could not represent well.
    """

    def __init__(
        self,
        num_steps: int = 4,
        hidden_gru: int = 128,
        learn_beta: bool = True,
        feature_dim: int = 128,
        instant_gain: float = 0.25,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_gru = hidden_gru
        self.instant_gain = instant_gain

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(2, 8, kernel_size=7, stride=1, padding=3, bias=False)
        self.lif1 = snn.Leaky(
            beta=0.9,
            threshold=0.1,
            spike_grad=spike_grad,
            learn_beta=learn_beta,
        )

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.lif2 = snn.Leaky(
            beta=0.85,
            threshold=0.1,
            spike_grad=spike_grad,
            learn_beta=learn_beta,
        )

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.lif3 = snn.Leaky(
            beta=0.8,
            threshold=0.1,
            spike_grad=spike_grad,
            learn_beta=learn_beta,
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()

        encoder_width = 64 * 8 * 8  # 32 spike-rate channels + 32 membrane channels
        self.proj = nn.Sequential(
            nn.Linear(encoder_width, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

        self.gru = nn.GRUCell(input_size=feature_dim, hidden_size=hidden_gru)
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(hidden_gru),
            nn.Linear(hidden_gru, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.instant_head = nn.Linear(feature_dim, 3)

        nn.init.uniform_(self.temporal_head[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.temporal_head[-1].bias)
        nn.init.uniform_(self.instant_head.weight, -0.005, 0.005)
        nn.init.zeros_(self.instant_head.bias)

    def init_state(self, batch_size: int = 1, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem1": self.lif1.init_leaky(),
            "mem2": self.lif2.init_leaky(),
            "mem3": self.lif3.init_leaky(),
            "h_gru": torch.zeros(batch_size, self.hidden_gru, device=device),
        }

    def detach_state(self, state: dict) -> dict:
        """Detach recurrent state for truncated backpropagation through time."""
        return _detach_tree(state)

    def forward(self, x, state):
        if state is None:
            state = self.init_state(batch_size=x.shape[0], device=x.device)

        mem1 = state["mem1"]
        mem2 = state["mem2"]
        mem3 = state["mem3"]
        h_gru = state["h_gru"]

        spk1_sum = 0.0
        spk2_sum = 0.0
        spk3_sum = 0.0

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk3, mem3 = self.lif3(self.conv3(spk2), mem3)
            spk1_sum = spk1_sum + spk1
            spk2_sum = spk2_sum + spk2
            spk3_sum = spk3_sum + spk3

        spk3_rate = spk3_sum / float(self.num_steps)
        feature_map = torch.cat([spk3_rate, mem3], dim=1)
        visual = self.proj(self.flatten(self.pool(feature_map)))

        h_gru = self.gru(visual, h_gru)
        torque_slow = self.temporal_head(h_gru)
        torque_fast = self.instant_head(visual)
        torque = torque_slow + self.instant_gain * torque_fast

        new_state = {
            "mem1": mem1,
            "mem2": mem2,
            "mem3": mem3,
            "h_gru": h_gru,
        }

        metrics = {
            "fr_layer1": (spk1_sum.mean() / self.num_steps).detach().item(),
            "fr_layer2": (spk2_sum.mean() / self.num_steps).detach().item(),
            "fr_layer3": (spk3_sum.mean() / self.num_steps).detach().item(),
            "active_layer1": (spk1_sum > 0).float().mean().detach().item(),
            "active_layer2": (spk2_sum > 0).float().mean().detach().item(),
            "active_layer3": (spk3_sum > 0).float().mean().detach().item(),
            "gru_abs": h_gru.abs().mean().detach().item(),
            "instant_abs": torque_fast.abs().mean().detach().item(),
        }

        return torque, new_state, metrics


class SatelliteInferenceInterface:
    """Bridge between MATLAB/Simulink and the PyTorch model."""

    def __init__(
        self,
        model_path: str | None = None,
        num_steps: int = 4,
        hidden_gru: int = 128,
        input_size: int = 256,
        ema_alpha: float = 1.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.ema_alpha = ema_alpha
        self.last_torque = [0.0, 0.0, 0.0]

        self.model = SpikingSatelliteController(
            num_steps=num_steps,
            hidden_gru=hidden_gru,
        ).to(self.device)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            clean = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
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

        smooth_t = [0.0, 0.0, 0.0]
        for i in range(3):
            smooth_t[i] = self.ema_alpha * raw_t[i] + (1 - self.ema_alpha) * self.last_torque[i]

        self.last_torque = smooth_t
        return smooth_t


# Global instance for MATLAB:
#   py.snn_controller.controller_interface.calculate_torque(event_matrix)
controller_interface = SatelliteInferenceInterface()
