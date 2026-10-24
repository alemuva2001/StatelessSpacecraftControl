"""
Overfit sanity check for the SNN T_D controller.

This trains and validates on one episode. It is useful to verify that the
architecture can reproduce a known T_D trajectory before running the full
dataset. It writes separate overfit weights/checkpoints by default.
"""

from training_common import train_from_config


CONFIG = {
    "overfit_episode": 1,
    "batch_size": 1,
    "seq_length": 40,
    "input_size": 256,
    "epochs": 100,
    "lr": 3e-4,
    "grad_clip": 1.0,
    "patience": 20,
    "best_model_name": "best_model_overfit.pth",
    "last_checkpoint_name": "checkpoint_last_overfit.pth",
    "best_checkpoint_name": "checkpoint_best_overfit.pth",
    "device": "auto",
    "resume": True,
}


if __name__ == "__main__":
    train_from_config(CONFIG)
