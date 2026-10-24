"""Independent bridge: initialize(checkpoint), reset(), step(H,W,2) -> 3 Nm."""
from pathlib import Path
import numpy as np
import torch
from proportional_model import ProportionalSNN, preprocess, device_auto
from proportional_data import TARGET_KIND

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / 'outputs' / 'run_150ep_mps' / 'best.pt'


class Predictor:
    def __init__(self, checkpoint=None, device='auto'):
        self.device = device_auto() if device=='auto' else device
        ckpt_path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        if ckpt_path.is_file():
            ck=torch.load(ckpt_path,map_location='cpu',weights_only=False)
            if ck.get('target_kind') != TARGET_KIND:
                raise ValueError('Checkpoint must explicitly declare target_kind=p_plus_gyro')
            self.model=ProportionalSNN(**ck['config']).to(self.device).eval()
            try:
                self.model.load_state_dict(ck['model'],strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f'NeuralP checkpoint incompatible with current architecture: {ckpt_path}. '
                    'Use a checkpoint trained with the gated recurrent model; legacy first/ weights are not compatible.'
                ) from exc
            self.scale=torch.tensor(ck['scale'],device=self.device)
            self.dt=ck['dt']; self.sensor_shape=tuple(ck['sensor_shape'])
            print(f"[SNN_P] Modelo cargado desde {ckpt_path}")
        else:
            raise FileNotFoundError(f"No se encontró el checkpoint P en '{ckpt_path}'. Selecciona pesos compatibles con initialize(checkpoint).")
        self.reset()

    def reset(self):
        self.state=self.model.initial(device=self.device)
        self.first=True

    @torch.no_grad()
    def step(self, frame):
        try:
            a=np.asarray(memoryview(frame),dtype=np.float32)
        except Exception:
            try:
                a=np.asarray(frame,dtype=np.float32)
            except Exception:
                a=np.array(list(frame),dtype=np.float32)
        if a.shape != (*self.sensor_shape,2):
            raise ValueError(f'Expected {(*self.sensor_shape,2)}, got {a.shape}')
        x=preprocess(torch.from_numpy(a.transpose(2,0,1).copy())[None].to(self.device),self.model.config['size'])
        # Initial DVS transient is not motion. Same convention in exporter/train.
        if self.first:
            x.zero_(); self.first=False
        y,self.state=self.model(x,self.state)
        return (y[0]*self.scale).cpu().tolist()


_predictor=None

def initialize(checkpoint=None, device='cpu'):
    global _predictor
    _predictor=Predictor(checkpoint,str(device))

def reset():
    if _predictor is None: raise RuntimeError('Call initialize first')
    _predictor.reset()

def step(frame):
    if _predictor is None: raise RuntimeError('Call initialize first')
    return _predictor.step(frame)

