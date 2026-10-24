"""TBPTT in chronological order; state carries between chunks, not episodes."""
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
from reporting import plot_history
import numpy as np
import torch
from proportional_data import load_episode, make_split, verify_split, sensor_shape, TARGET_KIND
from proportional_model import ProportionalSNN, device_auto
from episode_store import EpisodeStore


@torch.no_grad()
def predict(model,x,device):
    x_dev=x.to(device)
    state=model.initial(device=device)
    values=[]
    for frame in x_dev:
        y,state=model(frame[None],state)
        values.append(y[0])
    return torch.stack(values).cpu()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'outputs'/'first'))
    p.add_argument('--epochs',type=int,default=15); p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--size',type=int,default=64); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto'); p.add_argument('--lr',type=float,default=0.002)
    p.add_argument('--lr-scheduler',choices=['plateau','none'],default='plateau')
    p.add_argument('--lr-patience',type=int,default=5,help='Tolerated epochs without improvement; reduction on the next one')
    p.add_argument('--lr-factor',type=float,default=0.5)
    p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--cache-mode',choices=['memory','stream'],default='memory',help='stream loads one episode at a time without a full dataset RAM cache')
    p.add_argument('--feature-grid',type=int,default=4,help='Spatial pooling grid before GRU; 4 preserves the baseline')
    p.add_argument('--overfit-episode',type=int,help='Diagnostic: train and replay one episode; no held-out validation')
    args=p.parse_args()
    if min(args.epochs,args.chunk,args.size)<1: p.error('epochs/chunk/size must be positive')
    if args.lr<=0 or not np.isfinite(args.lr): p.error('lr must be finite and positive')
    if args.lr_patience<0: p.error('lr-patience must be nonnegative')
    if not 0<args.lr_factor<1: p.error('lr-factor must be between 0 and 1')
    if not 0<=args.min_lr<=args.lr: p.error('min-lr must be between zero and lr')
    torch.set_num_threads(4); torch.manual_seed(args.seed); random.seed(args.seed)
    device=device_auto() if args.device=='auto' else args.device
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists(): raise FileExistsError('Use a fresh --out to preserve checkpoints')
    if args.overfit_episode is None:
        split=make_split(args.data,args.seed); verify_split(split)
    else:
        name=f'ep_{args.overfit_episode:04d}_states.mat'
        if not (Path(args.data)/name).is_file(): raise FileNotFoundError(name)
        split={'train':[name],'val':[],'test':[]}
        print(f'[OVERFIT MODE] Training and replaying {name}; no generalisation measurement',flush=True)
    replay_names=split['val'] if args.overfit_episode is None else split['train']
    score_label='Val' if args.overfit_episode is None else 'Replay'
    (out/'split.json').write_text(json.dumps(split,indent=2))
    print(f'Device: {device}',flush=True)
    print(f'Input size: {args.size}x{args.size}')
    print(f'Batch size: 1 | TBPTT: {args.chunk} steps')
    print(f"Dataset split: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test episodes")
    print('Target: P + gyroscopic compensation | Loss: normalised MSE',flush=True)
    cache=EpisodeStore(args.data,args.size,args.cache_mode)
    print(f'Data loading: {args.cache_mode}',flush=True)
    dt=None; reference_kp=None; reference_shape=None
    scale=torch.zeros(3)
    train_names=set(split['train'])
    for name in tqdm([n for names in split.values() for n in names],desc='  Check',unit='episode'):
        x,y,t,kp=cache[name]
        shape=sensor_shape(Path(args.data)/name)
        if dt is None:
            dt=float(np.diff(t)[0]); reference_kp=kp; reference_shape=shape
        if not np.allclose(np.diff(t),dt,atol=1e-8,rtol=1e-6):
            raise ValueError('All episodes must have identical fixed dt')
        if not np.allclose(kp,reference_kp): raise ValueError('Fixed Kp required')
        if shape!=reference_shape: raise ValueError('Fixed sensor shape required')
        if name in train_names: scale=torch.maximum(scale,y.abs().amax(0))
        del x,y
    scale=scale.clamp_min(1e-8)
    model=ProportionalSNN(size=args.size,feature_grid=args.feature_grid).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode='min',factor=args.lr_factor,patience=args.lr_patience,
        threshold=0.0,threshold_mode='abs',min_lr=args.min_lr)
        if args.lr_scheduler=='plateau' else None)
    print(f'LR scheduler: {args.lr_scheduler}; patience={args.lr_patience}; factor={args.lr_factor}; min_lr={args.min_lr:g}')
    best=float('inf'); history=[]
    print(f'\nNormalisation torque_peak [Nm]: {scale.tolist()}')
    print(f'Model: {sum(v.numel() for v in model.parameters()):,} params')
    print(f'\nStarting training - epochs 1..{args.epochs}')
    print('=' * 60,flush=True)
    for epoch in range(args.epochs):
        model.train(); names=split['train'].copy(); random.shuffle(names)
        total=0.; count=0
        progress=tqdm(names,desc='  Train',unit='episode',leave=False)
        for name in progress:
            x,y,_,_=cache[name]
            x_dev=x.to(device); y_dev=(y/scale).to(device)
            state=model.initial(device=device)
            opt.zero_grad()
            for start in range(0,len(x_dev),args.chunk):
                predictions=[]
                end=min(start+args.chunk,len(x_dev))
                for frame in x_dev[start:end]:
                    pred,state=model(frame[None],state); predictions.append(pred[0])
                pred=torch.stack(predictions)
                loss=(pred-y_dev[start:end]).square().mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                (loss*((end-start)/len(x_dev))).backward()
                state=model.detach(state)
                total+=loss.item()*(end-start); count+=end-start
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            progress.set_postfix(loss=f'{total/count:.6f}')
            del x,y,x_dev,y_dev,state,pred,predictions,loss
        model.eval()
        errors=[]
        for n in tqdm(replay_names,desc=f'  {score_label}',unit='episode',leave=False):
            x,y,_,_=cache[n]
            errors.append((predict(model,x,device)-y/scale).square())
            del x,y
        val=torch.cat(errors).mean().item()
        if not np.isfinite(val): raise RuntimeError('Nonfinite validation loss')
        lr_used=opt.param_groups[0]['lr']
        if scheduler is not None: scheduler.step(val)
        lr_next=opt.param_groups[0]['lr']
        row=dict(epoch=epoch+1,train_mse=total/count,val_mse=val,lr=lr_used,lr_next=lr_next); history.append(row)
        if lr_next<lr_used:
            print(f'[LR] {score_label} plateau: {lr_used:.2e} -> {lr_next:.2e} for next epoch',flush=True)
        improved=val<best
        flag=' <- best' if improved else ''
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Train: {total/count:.6f} | {score_label}: {val:.6f} | LR: {row['lr']:.2e}{flag}",flush=True)
        if improved:
            best=val
            torch.save(dict(mode='overfit' if args.overfit_episode is not None else 'held_out',target_kind=TARGET_KIND,model=model.state_dict(),config=model.config,scale=scale.tolist(),
                            dt=dt,sensor_shape=reference_shape,kp=reference_kp.tolist(),split=split,
                            seed=args.seed,epoch=epoch+1,args=vars(args),
                            scheduler_state=scheduler.state_dict() if scheduler is not None else None),out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        plot_history(history,out/'training_history.png',score_label)
    print(f'\nTraining complete. Best {score_label.lower()} loss: {best:.6f}')
    print(f"Best model weights: {out/'best.pt'}")
    print(f"Training curves: {out/'training_history.png'}")

if __name__=='__main__': main()
