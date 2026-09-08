# IAC_version

Snapshot of the event-camera spacecraft controller for the IAC work.
The tested configuration combines a conventional proportional controller with
a neural derivative-torque estimate. The network is a shared convolutional SNN
encoder followed by a GRU and three torque outputs. It predicts the derivative
term, not the complete PD command.

The closed-loop examples discussed for this version show recovery with P plus
neural D. Standalone neural D has not demonstrated reliable braking in the
tested cases. This snapshot does not claim general stability or flight readiness.

## Included files

- `SNN/snn_controller.py`: architecture and direct-frame preprocessing.
- `SNN/snn_inference.py`: stateful Python interface for Simulink.
- `SNN/training_common.py` and `SNN/supervised_train*.py`: supervised training
  and end-of-epoch checkpoint resume.
- `SNN/compute_dataset_stats.py`: torque normalization statistics generation.
- `SNN/frame2events.py`: V2E bridge used by the model initialization callback.
- `SNN/Weights/best_model_BEST060926.pth`: selected inference weights.
- `Data/dataset_stats.npz`: normalization statistics currently used locally
  for inference. Training provenance must be checked before pairing these
  statistics with any other checkpoint.
- `Simulation/Model.slx`, MATLAB helpers, and `Simulation/dataset_generation.m`.
- Evaluation and plotting utilities in `SNN/` and `Simulation/`.

Dataset episodes, optimizer checkpoints, local archives, generated plots/videos,
Simulink caches, Python environments, and the external reference repository are
not part of this snapshot. Older commits retain their existing history.

## Running the snapshot

Commands below assume the working directory is `SNN_ControlSimulation`.

1. Configure a Python environment compatible with your MATLAB installation.
   Core Python imports are `torch`, `snntorch`, `numpy`, `scipy`, `h5py`,
   `matplotlib`, and `tqdm`. The current Simulink initialization also imports
   `frame2events`, which requires `v2ecore.emulator` from the V2E project.
   An environment and the V2E dependency are not vendored here.
2. Prepare the filename expected by the inference and training scripts:

   ```bash
   cp SNN/Weights/best_model_BEST060926.pth SNN/Weights/best_model.pth
   ```

   The generated `best_model.pth` is ignored. The selected dated checkpoint is
   kept separately so training cannot overwrite the published snapshot.
3. Configure MATLAB `pyenv` to use that environment. The `Model.slx` InitFcn
   callback currently contains a machine-specific `snn_dir`: set it to the
   absolute `SNN` directory in your checkout. `setup.sh` is a legacy local
   helper with absolute paths and must not be used unchanged on another machine.
4. Reload Python modules after code changes:

   ```matlab
   py.importlib.reload(py.importlib.import_module('snn_controller'));
   py.importlib.reload(py.importlib.import_module('snn_inference'));
   ```

5. Run the desired configuration from `Simulation/Simulation.m`. Check the
   active signal routing in `Model.slx`: for P plus neural D, replace the
   conventional derivative contribution rather than summing both. The network
   already predicts signed derivative torque; do not invert it again after
   inference. Reset neural state at the start of each simulation.

Direct MATLAB frames use `[H,W,C]` and are converted to `[C,H,W]` with
`transpose(2,0,1)`. The dataset loader uses `permute(0,3,2,1)` because HDF5
exposes the saved MATLAB dimensions in reverse order. These distinct
conversions intentionally preserve the same pixel coordinates.

## Training

The generator currently writes episodes to `Data2` and keeps initial angular
velocity zero. Attitude and the star field are sampled per episode.
Generate matching normalization statistics for the intended training directory
before training. `compute_dataset_stats.py` has its own directory configuration;
check it before running. Do not recompute the published inference statistics
and continue using the existing weights as though their normalization were unchanged.

For a new experiment with data and matching statistics in `Data2`:

```bash
python SNN/supervised_train_NVIDIA.py --fresh --data-dir Data2 --weights-dir output/training
```

Resume that experiment using the same dataset and options:

```bash
python SNN/supervised_train_NVIDIA.py --data-dir Data2 --weights-dir output/training
```

`--epochs` specifies the final total epoch number. Resume loads
`checkpoint_last.pth`, including optimizer and scheduler state, when present.
Only completed epochs are saved. The published inference weights do not include
the optimizer state or the original train/validation split manifest.

## Artifact identity and checks

SHA-256 of the selected weights:
`1d520d90a46b6e0675ebbb2813efcb93aea113b75de6e20bdf60ab0cd052af52`

SHA-256 of the included inference statistics:
`cd938f6809ce31345ef2aa8c5d5863fa79cb99ae445617bfb5c8283b66c76aad`

During snapshot preparation, the selected weights loaded successfully into the
current architecture. This packaging operation did not rerun MATLAB simulations
or training. Runtime paths and dependencies above must be configured on a fresh
checkout.
