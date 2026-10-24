"""
Shared supervised training loop for the SNN derivative-torque controller.

The target is torque_pid from the dataset, which in this project represents the
derivative braking term T_D. The scripts in this folder only choose defaults:
this module owns dataset loading, TBPTT, checkpointing, and resume logic.
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from snn_controller import SpikingSatelliteController


SNN_DIR = Path(__file__).resolve().parent


BASE_CONFIG: dict[str, Any] = {
    "data_dir": str(SNN_DIR.parents[1] / "Data"),
    "weights_dir": str(SNN_DIR / "Weights"),
    "seq_length": 20,
    "batch_size": 2,
    "epochs": 200,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_split": 0.15,
    "grad_clip": 1.0,
    "patience": 30,
    "num_workers": 0,
    "num_steps": 4,
    "hidden_gru": 128,
    "feature_dim": 128,
    "instant_gain": 0.25,
    "input_size": 256,
    "peak_weight": 5.0,
    "smooth_weight": 0.0,
    "sign_weight": 0.05,
    "sign_threshold": 0.03,
    "target_shift": 0,
    "seed": 42,
    "device": "auto",
    "amp": True,
    "resume": True,
    "resume_path": None,
    "best_model_name": "best_model.pth",
    "last_checkpoint_name": "checkpoint_last.pth",
    "best_checkpoint_name": "checkpoint_best.pth",
    "overfit_episode": None,
}


def merged_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(BASE_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def resolve_path(path_like: str | os.PathLike) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (SNN_DIR / path).resolve()


def load_stats(data_dir: Path) -> dict[str, np.ndarray]:
    stats_path = data_dir / "dataset_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"No se encontro '{stats_path}'. Ejecuta primero compute_dataset_stats.py."
        )

    stats = np.load(stats_path)
    return {key: stats[key].astype(np.float32) for key in stats.files}


class EpisodeDataset(Dataset):
    def __init__(
        self,
        episode_pairs: list[tuple[Path, Path]],
        input_size: int,
        stats: dict[str, np.ndarray],
        target_shift: int = 0,
    ):
        self.pairs = episode_pairs
        self.input_size = input_size
        self.stats = stats
        self.target_shift = target_shift

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ev_path, st_path = self.pairs[idx]

        with h5py.File(ev_path, "r") as h5_file:
            raw = h5_file["events_matrix"][:]

        events = self._events_to_tensor(raw)
        total_steps, _, height, width = events.shape
        if height != self.input_size or width != self.input_size:
            events = F.interpolate(
                events.float(),
                size=(self.input_size, self.input_size),
                mode="area",
            )

        state_data = scipy.io.loadmat(st_path)
        torques = state_data["torque_pid"].astype(np.float32)

        if torques.shape[0] != total_steps:
            torques = self._resample_targets(torques, total_steps)

        torques = (torques - self.stats["torque_mean"]) / self.stats["torque_peak"]
        events, torques = self._apply_target_shift(events, torques)

        return events, torch.from_numpy(torques)

    @staticmethod
    def _events_to_tensor(raw: np.ndarray) -> torch.Tensor:
        if raw.ndim != 4:
            raise ValueError(f"Expected 4-D events_matrix, got shape {raw.shape}")

        if raw.shape[-1] == 2:
            # h5py reads MATLAB v7.3 arrays with the spatial axes swapped
            # relative to the convention used by Simulink at inference time.
            return torch.from_numpy(raw).permute(0, 3, 2, 1)
        if raw.shape[1] == 2:
            return torch.from_numpy(raw)

        raise ValueError(f"Cannot infer channel axis for events_matrix shape {raw.shape}")

    @staticmethod
    def _resample_targets(torques: np.ndarray, total_steps: int) -> np.ndarray:
        old_grid = np.linspace(0, 1, torques.shape[0])
        new_grid = np.linspace(0, 1, total_steps)
        out = np.zeros((total_steps, torques.shape[1]), dtype=np.float32)
        for axis in range(torques.shape[1]):
            out[:, axis] = np.interp(new_grid, old_grid, torques[:, axis])
        return out

    def _apply_target_shift(
        self,
        events: torch.Tensor,
        torques: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        shift = int(self.target_shift)
        if shift == 0:
            return events, torques
        if abs(shift) >= len(torques):
            raise ValueError("target_shift is larger than the episode length")

        if shift > 0:
            return events[:-shift], torques[shift:]
        return events[-shift:], torques[:shift]


def build_dataloaders(cfg: dict[str, Any], stats: dict[str, np.ndarray]):
    data_dir = Path(cfg["data_dir"])

    if cfg.get("overfit_episode") is not None:
        ep = int(cfg["overfit_episode"])
        events_path = data_dir / f"ep_{ep:04d}_events.mat"
        states_path = data_dir / f"ep_{ep:04d}_states.mat"
        if not events_path.exists() or not states_path.exists():
            raise FileNotFoundError(f"Episode {ep:04d} not found in {data_dir}")
        train_pairs = [(events_path, states_path)]
        val_pairs = [(events_path, states_path)]
        print(f"\n[OVERFIT MODE] Training and validating only on episode {ep:04d}")
    else:
        event_files = sorted(Path(path) for path in glob.glob(str(data_dir / "ep_*_events.mat")))
        if not event_files:
            raise FileNotFoundError(f"No episode files found in {data_dir}")

        pairs = []
        for events_path in event_files:
            states_path = Path(str(events_path).replace("_events.mat", "_states.mat"))
            if states_path.exists():
                pairs.append((events_path, states_path))

        if not pairs:
            raise FileNotFoundError(f"No matching _events/_states pairs found in {data_dir}")

        rng = np.random.default_rng(int(cfg["seed"]))
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * float(cfg["val_split"])))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]
        print(f"\nDataset split: {len(train_pairs)} train / {len(val_pairs)} val episodes")

    ds_train = EpisodeDataset(
        train_pairs,
        input_size=int(cfg["input_size"]),
        stats=stats,
        target_shift=int(cfg["target_shift"]),
    )
    ds_val = EpisodeDataset(
        val_pairs,
        input_size=int(cfg["input_size"]),
        stats=stats,
        target_shift=int(cfg["target_shift"]),
    )

    drop_train = len(train_pairs) >= int(cfg["batch_size"]) and cfg.get("overfit_episode") is None
    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg["batch_size"]),
        shuffle=cfg.get("overfit_episode") is None,
        drop_last=drop_train,
        num_workers=int(cfg["num_workers"]),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg["num_workers"]),
    )
    return dl_train, dl_val


def select_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def optimizer_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device: torch.device):
    checkpoint = torch.load(path, map_location=device)

    if "model_state" not in checkpoint:
        model.load_state_dict(clean_state_dict(checkpoint))
        print(f"[INFO] Loaded model weights from {path}")
        return 1, float("inf"), 0

    model.load_state_dict(clean_state_dict(checkpoint["model_state"]))

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to_device(optimizer, device)

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val = float(checkpoint.get("best_val", float("inf")))
    no_improve = int(checkpoint.get("no_improve", 0))
    print(f"[INFO] Resuming from {path} at epoch {start_epoch}")
    return start_epoch, best_val, no_improve


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val: float,
    no_improve: int,
    cfg: dict[str, Any],
) -> None:
    checkpoint = {
        "format_version": 1,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val": best_val,
        "no_improve": no_improve,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in cfg.items()},
    }
    torch.save(checkpoint, path)


def torque_loss(pred, target, previous_pred, criterion, cfg: dict[str, Any]):
    raw_loss = criterion(pred, target)
    peak_weight = 1.0 + float(cfg["peak_weight"]) * target.abs()
    mse_loss = torch.mean(raw_loss * peak_weight)

    sign_mask = (target.abs() > float(cfg["sign_threshold"])).float()
    wrong_sign = F.relu(-(pred * target))
    sign_den = sign_mask.sum().clamp_min(1.0)
    sign_loss = (wrong_sign * sign_mask).sum() / sign_den

    smooth_loss = pred.new_tensor(0.0)
    if previous_pred is not None:
        smooth_loss = torch.mean((pred - previous_pred.detach()) ** 2)

    loss = (
        mse_loss
        + float(cfg["sign_weight"]) * sign_loss
        + float(cfg["smooth_weight"]) * smooth_loss
    )
    return loss


def run_epoch(model, loader, optimizer, criterion, scaler, cfg, device, train: bool):
    model.train(train)
    phase = "Train" if train else "Val"
    total_loss = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc=f"  {phase:<5}", unit="batch", leave=False)

    with torch.set_grad_enabled(train):
        for batch_events, batch_torques in pbar:
            batch_size, total_steps, _, _, _ = batch_events.shape
            state = model.init_state(batch_size=batch_size, device=device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            chunk_losses = []
            previous_pred = None

            for start in range(0, total_steps, int(cfg["seq_length"])):
                end = min(start + int(cfg["seq_length"]), total_steps)
                if end - start < 2:
                    break

                state = model.detach_state(state)
                if previous_pred is not None:
                    previous_pred = previous_pred.detach()

                loss_chunk = torch.zeros((), device=device)
                n_steps = 0

                for step in range(start, end):
                    # Keep event episodes in their compact stored dtype on the CPU;
                    # cast only the current frame as it is transferred to the device.
                    x_t = batch_events[:, step].to(device=device, dtype=torch.float32)
                    target = batch_torques[:, step].to(device)

                    with torch.amp.autocast(
                        "cuda",
                        enabled=(device.type == "cuda" and bool(cfg["amp"])),
                    ):
                        pred, state, _ = model(x_t, state, collect_metrics=False)
                        loss_step = torque_loss(pred, target, previous_pred, criterion, cfg)

                    previous_pred = pred
                    loss_chunk = loss_chunk + loss_step
                    n_steps += 1

                loss_chunk = loss_chunk / max(n_steps, 1)

                if train:
                    if scaler is not None:
                        scaler.scale(loss_chunk).backward()
                    else:
                        loss_chunk.backward()

                chunk_losses.append(loss_chunk.detach().item())

            if not chunk_losses:
                continue

            if train:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                    optimizer.step()

            batch_loss = float(np.mean(chunk_losses))
            total_loss += batch_loss
            total_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss:.6f}"})

    return total_loss / max(total_batches, 1)


def parse_cli(cfg: dict[str, Any]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train the SNN derivative-torque controller.")
    parser.add_argument("--data-dir")
    parser.add_argument("--weights-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device")
    parser.add_argument("--resume", dest="resume", action="store_true", default=None)
    parser.add_argument("--fresh", dest="resume", action="store_false")
    parser.add_argument("--resume-path")
    parser.add_argument("--overfit-episode", type=int)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    args = parser.parse_args()

    mapping = {
        "data_dir": args.data_dir,
        "weights_dir": args.weights_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "input_size": args.input_size,
        "lr": args.lr,
        "device": args.device,
        "resume": args.resume,
        "resume_path": args.resume_path,
        "overfit_episode": args.overfit_episode,
        "amp": args.amp,
    }
    for key, value in mapping.items():
        if value is not None:
            cfg[key] = value
    return cfg


def train_from_config(overrides: dict[str, Any] | None = None, *, use_cli: bool = True):
    cfg = merged_config(overrides)
    if use_cli:
        cfg = parse_cli(cfg)

    cfg["data_dir"] = resolve_path(cfg["data_dir"])
    cfg["weights_dir"] = resolve_path(cfg["weights_dir"])
    Path(cfg["weights_dir"]).mkdir(parents=True, exist_ok=True)

    device = select_device(str(cfg["device"]))
    print(f"Device: {device}")
    print(f"Input size: {cfg['input_size']}x{cfg['input_size']}")
    print(f"Batch size: {cfg['batch_size']} | TBPTT: {cfg['seq_length']} steps")

    stats = load_stats(Path(cfg["data_dir"]))
    print("\nNormalisation stats loaded:")
    print(f"  torque_mean = {stats['torque_mean']}")
    print(f"  torque_peak = {stats['torque_peak']}")

    train_loader, val_loader = build_dataloaders(cfg, stats)

    model = SpikingSatelliteController(
        num_steps=int(cfg["num_steps"]),
        hidden_gru=int(cfg["hidden_gru"]),
        learn_beta=True,
        feature_dim=int(cfg["feature_dim"]),
        instant_gain=float(cfg["instant_gain"]),
    ).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"\nModel: {total_params:,} params ({trainable_params:,} trainable)")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    criterion = nn.MSELoss(reduction="none")

    use_amp = device.type == "cuda" and bool(cfg["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    weights_dir = Path(cfg["weights_dir"])
    best_model_path = weights_dir / str(cfg["best_model_name"])
    last_checkpoint_path = weights_dir / str(cfg["last_checkpoint_name"])
    best_checkpoint_path = weights_dir / str(cfg["best_checkpoint_name"])

    start_epoch = 1
    best_val = float("inf")
    no_improve = 0

    if bool(cfg["resume"]):
        resume_path = Path(cfg["resume_path"]) if cfg["resume_path"] else last_checkpoint_path
        if not resume_path.is_absolute():
            resume_path = (weights_dir / resume_path).resolve()
        if resume_path.exists():
            start_epoch, best_val, no_improve = load_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                scaler,
                device,
            )
        else:
            print(f"[INFO] No checkpoint found at {resume_path}. Starting fresh.")

    print(f"\nStarting training - epochs {start_epoch}..{cfg['epochs']}")
    print("=" * 60)

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, scaler, cfg, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, scaler, cfg, device, train=False)

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val,
                no_improve,
                cfg,
            )
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: train={train_loss}, val={val_loss}")

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val,
                no_improve,
                cfg,
            )
        else:
            no_improve += 1

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val,
            no_improve,
            cfg,
        )

        flag = " <- best" if improved else ""
        print(
            f"Epoch {epoch:03d}/{cfg['epochs']} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {lr_now:.2e}{flag}"
        )

        if no_improve >= int(cfg["patience"]):
            print(f"\nEarly stopping - no improvement for {cfg['patience']} epochs.")
            break

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    print(f"Best model weights: {best_model_path}")
    print(f"Resume checkpoint:  {last_checkpoint_path}")
