"""
Default supervised training for the SNN T_D controller.

This script uses the shared training loop in training_common.py and keeps
moderate defaults for local machines. Use supervised_train_NVIDIA.py for the
same model with CUDA-oriented defaults.
"""

from training_common import train_from_config


CONFIG = {
    "batch_size": 1,
    "seq_length": 20,
    "input_size": 256,
    "epochs": 200,
    "lr": 2e-4,
    "grad_clip": 1.0,
    "patience": 30,
    "device": "auto",
    "resume": True,
}


if __name__ == "__main__":
    train_from_config(CONFIG)
