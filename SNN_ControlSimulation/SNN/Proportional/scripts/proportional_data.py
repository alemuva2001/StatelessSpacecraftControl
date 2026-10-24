"""Read the shared dataset_generation.m pairs. Never use torque_pid as P."""
from pathlib import Path
import h5py
import numpy as np
import scipy.io
import torch
from proportional_model import preprocess

TARGET_KIND = "p_plus_gyro"


def event_path(states_path):
    p=Path(states_path)
    if not p.name.endswith('_states.mat'):
        raise ValueError('Expected ep_XXXX_states.mat from dataset_generation.m')
    return p.with_name(p.name.replace('_states.mat','_events.mat'))


def sensor_shape(states_path):
    with h5py.File(event_path(states_path),'r') as f:
        # MATLAB C,H,W,T -> HDF5 T,W,H,C
        shape=f['events_matrix'].shape
        if len(shape)!=4 or shape[-1]!=2: raise ValueError('Expected MATLAB events_matrix [2,H,W,T]')
        return shape[2],shape[1]


def load_episode(path, size=64):
    states=scipy.io.loadmat(path,simplify_cells=True)
    meta=states.get('meta',{})
    if isinstance(meta,dict) and (not meta.get('completed_horizon',True) or not meta.get('training_ready',True)):
        raise ValueError(f'{path}: episode incomplete or not marked training_ready')
    if 'torque_p' not in states:
        raise ValueError(f'{path}: missing torque_p. Regenerate with dataset_generation.m; torque_pid is D.')
    with h5py.File(event_path(path),'r') as f:
        x=np.asarray(f['events_matrix']).transpose(0,3,2,1).copy()
        t=np.asarray(f['event_time']).reshape(-1)
    y=np.asarray(states['torque_p']).reshape(-1,3)
    control_time=np.asarray(states['time_control']).reshape(-1)
    kp=np.asarray(states['kp']).reshape(-1)
    if x.ndim!=4 or x.shape[1]!=2 or y.shape!=(len(x),3): raise ValueError(f'{path}: invalid dimensions')
    if len(t)!=len(x) or len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
        raise ValueError(f'{path}: invalid timestamps')
    if control_time.shape!=t.shape or not np.allclose(t,control_time,rtol=0,atol=1e-8):
        raise ValueError('Event and label timestamps differ')
    if not np.isfinite(y).all() or kp.shape!=(3,) or not np.isfinite(kp).all() or np.any(kp<=0):
        raise ValueError('Invalid labels/gains')
    qe=np.asarray(states['q_error_control']).reshape(-1,4)
    if qe.shape!=(len(t),4):
        raise ValueError('Invalid q_error_control shape')
    expected_pure_p = qe[:,1:]*kp
    if 'torque_gyro' not in states:
        raise ValueError('P+gyro target requires the separate torque_gyro diagnostic')
    gyro = np.asarray(states['torque_gyro']).reshape(-1,3)
    if gyro.shape != y.shape or not np.isfinite(gyro).all():
        raise ValueError('Invalid torque_gyro')
    if meta.get('target_kind',TARGET_KIND) != TARGET_KIND:
        raise ValueError('Dataset target_kind is not p_plus_gyro')
    if not np.allclose(y, expected_pure_p + gyro, atol=1e-10, rtol=0):
        raise ValueError('Target must be +Kp*q_error_vector + torque_gyro; pure P is not accepted')
    if not np.allclose(y[0],0,atol=1e-9): raise ValueError('Reset requires initial attitude at reference')
    frames=torch.cat([preprocess(torch.from_numpy(a),size) for a in np.array_split(x,max(1,(len(x)+63)//64))])
    frames[0].zero_() # startup flash ignored identically by inference
    return frames,torch.tensor(y,dtype=torch.float32),t,kp


def make_split(folder,seed=42):
    # Old D-only episodes remain usable for D, but are explicitly excluded from P.
    files=[]; skipped=0
    for p in sorted(Path(folder).glob('ep_*_states.mat')):
        names={n for n,_,_ in scipy.io.whosmat(p)}
        if 'torque_p' not in names:
            skipped+=1; continue
        if not event_path(p).exists(): raise FileNotFoundError(event_path(p))
        files.append(p.name)
    print(f'P episodes: {len(files)}; skipped legacy D-only episodes: {skipped}',flush=True)
    if len(files)<3: raise ValueError('Generate at least three complete P-labelled episodes with dataset_generation.m')
    np.random.default_rng(seed).shuffle(files)
    n=max(1,round(len(files)*0.15))
    return {'train':files[2*n:],'val':files[n:2*n],'test':files[:n]}


def verify_split(split):
    groups=[set(split[k]) for k in ('train','val','test')]
    if any(not g for g in groups) or any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Empty or overlapping episode splits')
