"""
CUDA-oriented supervised training for the SNN T_D controller.

Run from the SNN folder or project root:
    python supervised_train_NVIDIA.py

Useful options:
    --resume              resume from checkpoint_last.pth when present
    --fresh               ignore checkpoints and start from scratch
    --batch-size 1        reduce VRAM use
    --seq-length 10       reduce TBPTT memory
"""

from training_common import train_from_config


CONFIG = {
    "batch_size": 2,
    "seq_length": 20,
    "input_size": 256,
    "epochs": 200,
    "lr": 3e-4,
    "grad_clip": 1.0,
    "patience": 30,
    "device": "auto",
    "resume": True,
}


if __name__ == "__main__":
    train_from_config(CONFIG)
