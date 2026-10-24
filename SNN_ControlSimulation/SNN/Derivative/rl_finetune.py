"""Online Simulink rollouts and PPO fine-tuning of torque heads, optionally GRU.

The visual encoder (including LIF memory) stays frozen. For GRU training, its
cached visual outputs are replayed chronologically from zero GRU state on every
PPO epoch. Old GRU states are never used as actor inputs in that mode.
Each action is a Gaussian latent in training-normalized units,
then denormalized and clipped in Nm. Log probabilities refer to the sampled
latent (not the clipped actuator command).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from snn_controller import SpikingSatelliteController, _preprocess_frame


DEFAULTS = dict(
    seed=42, threads=2, action_std=0.03, torque_limit=1e-3,
    actor_lr=1e-5, critic_lr=3e-4, epochs=4, minibatch=256,
    clip_ratio=0.2, target_kl=0.01, gamma=0.99, gae_lambda=0.95,
    imitation_weight=0.05, theta_ref_deg=10.0, omega_ref=0.1,
    attitude_weight=1.0, rate_weight=0.2, effort_weight=0.01,
    failure_angle_deg=60.0, failure_penalty=5.0,
    recovery_deg=0.5, recovery_hold_s=2.0,
    train_gru=False,
)


def gae(reward, value, gamma, lam):
    """Finite-horizon episode: reward[k] follows action[k], terminal V = 0."""
    advantage = np.zeros_like(reward, dtype=np.float64)
    acc = 0.0
    for k in reversed(range(len(reward))):
        next_value = value[k + 1] if k + 1 < len(value) else 0.0
        delta = reward[k] + gamma * next_value - value[k]
        acc = delta + gamma * lam * acc
        advantage[k] = acc
    return advantage, advantage + value


def angle_error(q, reference):
    q = np.asarray(q, dtype=float)
    reference = np.asarray(reference, dtype=float).reshape(4)
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm < 1e-12) or np.linalg.norm(reference) < 1e-12:
        raise ValueError('Zero quaternion in rollout.')
    cosine = np.abs((q / norm[:, None]) @ (reference / np.linalg.norm(reference)))
    return 2 * np.arccos(np.clip(cosine, 0, 1))


class Trainer:
    def __init__(self, checkpoint, stats_path, output_dir, config=None):
        self.cfg = DEFAULTS | (config or {})
        c = self.cfg
        if c['action_std'] <= 0 or c['torque_limit'] <= 0:
            raise ValueError('Exploration scale and torque limit must be positive.')
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / 'manifest.json').exists():
            raise FileExistsError('Use a new output directory for each run.')
        torch.set_num_threads(int(c['threads']))
        torch.manual_seed(int(c['seed']))
        self.rng = np.random.default_rng(int(c['seed']))
        self.model = SpikingSatelliteController().cpu().eval()
        raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in raw.items()})
        with np.load(stats_path) as s:
            self.mean = torch.tensor(s['torque_mean'], dtype=torch.float32)
            self.peak = torch.tensor(s['torque_peak'], dtype=torch.float32)
        if not torch.all(self.peak > 0):
            raise ValueError('Invalid torque normalization.')
        # Keep evaluation reward scaling fixed when a checkpoint uses another
        # output normalization; only mean/peak denormalize that policy's actions.
        self.reward_peak = self.peak.clone()
        trainable = ('temporal_head.', 'instant_head.', 'gru.') if c['train_gru'] else ('temporal_head.', 'instant_head.')
        for name, p in self.model.named_parameters():
            p.requires_grad_(name.startswith(trainable))
        self.actor = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=c['actor_lr'])
        self.critic = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.value_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c['critic_lr'])
        self.model.proj.register_forward_hook(self._capture_visual)
        self.history = []
        self.pending = False
        self.evaluation_only = False
        self.policy_source = str(Path(checkpoint).resolve())
        self.best_validation_rank = None
        manifest = dict(config=c, source_checkpoint=str(Path(checkpoint).resolve()),
                        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                        stats_sha256=hashlib.sha256(Path(stats_path).read_bytes()).hexdigest(),
                        trainable=list(trainable),
                        actor_replay='full chronological sequence' if c['train_gru'] else 'cached fixed features',
                        horizon='finite; no bootstrap at episode end')
        (self.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    def _capture_visual(self, module, inputs, output):
        self.visual = output.detach()

    def begin(self, training=False, seed=0):
        if training and self.evaluation_only:
            raise RuntimeError('A checkpoint loaded for comparison cannot be trained in this session.')
        if self.pending:
            raise RuntimeError('Previous rollout has not been finished.')
        self.training = bool(training)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.state = self.model.init_state(batch_size=1, device='cpu')
        self.rows = []
        self.pending = True

    def step(self, events):
        if not self.pending:
            raise RuntimeError('Call begin_episode before starting Simulink.')
        with torch.no_grad():
            x = _preprocess_frame(np.asarray(events, dtype=np.float32), 256, 'cpu')
            mu, self.state, metrics = self.model(x, self.state)
            mu = mu.squeeze(0)
            if not torch.isfinite(mu).all():
                raise FloatingPointError('Non-finite neural torque; rollout aborted.')
            features = torch.cat((self.state['h_gru'], self.visual), dim=1).squeeze(0)
            latent = mu.clone()
            if self.training:
                latent += self.cfg['action_std'] * torch.randn(3, generator=self.generator)
            raw_torque = latent * self.peak + self.mean
            torque = raw_torque.clamp(-self.cfg['torque_limit'], self.cfg['torque_limit'])
            logp = Normal(mu, self.cfg['action_std']).log_prob(latent).sum()
            self.rows.append((features.numpy().copy(), latent.numpy().copy(),
                              float(logp), torque.numpy().copy(), mu.numpy().copy()))
        return torque.tolist() + [metrics[k] for k in ('fr_layer1', 'fr_layer2', 'fr_layer3', 'gru_abs')]

    def policy(self, features):
        return (self.model.temporal_head(features[:, :128]) +
                self.model.instant_gain * self.model.instant_head(features[:, 128:]))

    def recurrent_policy(self, visual):
        """Full BPTT over one episode; no stale hidden states or shuffled steps."""
        hidden = visual.new_zeros((1, self.model.hidden_gru))
        states = []
        for frame in visual:
            hidden = self.model.gru(frame.unsqueeze(0), hidden)
            states.append(hidden.squeeze(0))
        return self.policy(torch.cat((torch.stack(states), visual), dim=1))

    def finish(self, time, q, omega, target, applied, reference, last_disturbance=None, controller='NeuralD', filter_alpha=None):
        if controller not in ('NeuralD', 'PD') or (controller == 'PD' and self.training):
            raise ValueError('PD is evaluation-only; no training on PD trajectories.')
        if not self.pending or len(self.rows) < 2:
            raise RuntimeError('No complete rollout.')
        time = np.asarray(time, float).reshape(-1)
        n = len(self.rows)
        arrays = [np.asarray(a, float) for a in (q, omega, target, applied)]
        if len(time) != n or any(a.shape != (n, d) for a, d in zip(arrays, (4, 3, 3, 3))):
            raise ValueError('Inference calls and logged samples do not align; no update performed.')
        if not all(np.isfinite(a).all() for a in [time, *arrays]):
            raise ValueError('Non-finite simulation data; no update performed.')
        dt = np.diff(time)
        if not np.allclose(dt, 0.1, rtol=0, atol=1e-7):
            raise ValueError('This pilot requires synchronized dt = dt_control = 0.1 s.')
        q, omega, target, applied = arrays
        features, latent, old_logp, action, old_mu = map(np.asarray, zip(*self.rows))
        expected = target if controller == 'PD' else action
        if filter_alpha is not None:
            if self.training or controller != 'NeuralD' or not 0 <= filter_alpha < 1:
                raise ValueError('Filtering is supported only for deterministic NeuralD evaluation.')
            expected = np.empty_like(action)
            previous = np.zeros(3)
            for k, torque in enumerate(action):
                previous = filter_alpha * previous + (1 - filter_alpha) * torque
                expected[k] = previous
        if not np.allclose(expected, applied, rtol=1e-5, atol=1e-10):
            raise ValueError('The applied derivative torque differs from the sampled action. Check selector/logging.')
        neural_action = action.copy()
        action = applied
        theta = angle_error(q, reference)
        c = self.cfg
        # Action at t[k] is scored against the state reached at t[k+1].
        cost = (c['attitude_weight'] * (theta[1:] / np.deg2rad(c['theta_ref_deg']))**2 +
                c['rate_weight'] * np.sum((omega[1:] / c['omega_ref'])**2, axis=1) +
                c['effort_weight'] * np.sum((action[:-1] / self.reward_peak.numpy())**2, axis=1) +
                c['failure_penalty'] * (theta[1:] > np.deg2rad(c['failure_angle_deg'])))
        reward = -dt * cost
        x = torch.tensor(features[:-1], dtype=torch.float32)
        with torch.no_grad():
            value = self.critic(x).squeeze(1).numpy()
        advantage, returns = gae(reward, value, c['gamma'], c['gae_lambda'])
        index = len(self.history) + 1
        report = dict(episode=index, training=self.training, samples=n, controller=controller,
                      policy_source=self.policy_source if controller == 'NeuralD' else 'conventional_PD',
                      return_sum=float(reward.sum()), peak_deg=float(np.rad2deg(theta).max()),
                      final_deg=float(np.rad2deg(theta[-1])),
                      residual_deg=float(np.rad2deg(theta[time >= time[-1]-5]).mean()),
                      saturation_pct=float(100*np.mean(np.any(np.abs(action[:-1]) >= c['torque_limit'], axis=1))))
        if controller == 'PD':
            report['saturation_pct'] = None  # PD has no added torque clipping.
        if last_disturbance is not None:
            start = int(np.searchsorted(time, float(last_disturbance)))
            above = np.flatnonzero(np.rad2deg(theta[start:]) > c['recovery_deg'])
            first = start + int(above[-1]) + 1 if len(above) else start
            recovered = first < n and time[-1]-time[first] >= c['recovery_hold_s']-1e-8
            report['recovered'] = bool(recovered)
            report['recovery_s'] = float(time[first]-last_disturbance) if recovered else None
        np.savez_compressed(self.output / f'rollout_{index:04d}.npz',
                            time=time, q=q, omega=omega, reference=reference, target=target,
                            action=action, neural_action=neural_action, controller=controller,
                            latent=latent, old_logp=old_logp, old_mu=old_mu,
                            features=features, reward=reward, training=self.training)
        if self.training:
            report.update(self.update(x, latent[:-1], old_logp[:-1], old_mu[:-1],
                                      target[:-1], advantage, returns))
            # Original supervised checkpoint is never overwritten.
            torch.save(self.model.state_dict(), self.output / 'latest_model.pth')
            torch.save(dict(model=self.model.state_dict(), critic=self.critic.state_dict(),
                            actor_optimizer=self.actor.state_dict(),
                            value_optimizer=self.value_optimizer.state_dict(), config=c),
                       self.output / 'latest_training_state.pt')
        self.history.append(report)
        (self.output / 'history.json').write_text(json.dumps(self.history, indent=2))
        self.pending = False
        self.rows = []
        return report

    def save_validation(self, iteration, count=3):
        """Archive every evaluated policy; rank by recovery, then worst residual."""
        if self.pending:
            raise RuntimeError('Finish rollout before saving validation.')
        rows = self.history[-int(count):]
        if len(rows) != count or any(r['training'] or 'recovered' not in r for r in rows):
            raise ValueError('Expected a complete deterministic validation group.')
        folder = self.output / 'checkpoints'
        folder.mkdir(exist_ok=True)
        path = folder / f'iteration_{int(iteration):04d}.pth'
        if path.exists():
            raise FileExistsError(path)
        torch.save(self.model.state_dict(), path)
        report = dict(iteration=int(iteration), checkpoint=str(path),
                      recovered=sum(r['recovered'] for r in rows), episodes=count,
                      worst_residual_deg=max(r['residual_deg'] for r in rows),
                      mean_residual_deg=float(np.mean([r['residual_deg'] for r in rows])),
                      mean_return=float(np.mean([r['return_sum'] for r in rows])))
        rank = (-report['recovered'], report['worst_residual_deg'], report['mean_residual_deg'])
        if self.best_validation_rank is None or rank < self.best_validation_rank:
            self.best_validation_rank = rank
            torch.save(self.model.state_dict(), self.output/'best_validation_model.pth')
            (self.output/'best_validation.json').write_text(json.dumps(report, indent=2))
        path.with_suffix('.json').write_text(json.dumps(report, indent=2))
        return report

    def load_evaluation_checkpoint(self, path, stats_path=None):
        if self.pending:
            raise RuntimeError('Finish rollout before switching policy.')
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()})
        if stats_path is not None:
            with np.load(stats_path) as stats:
                mean = torch.tensor(stats['torque_mean'], dtype=torch.float32)
                peak = torch.tensor(stats['torque_peak'], dtype=torch.float32)
            if not torch.all(peak > 0):
                raise ValueError('Invalid torque normalization.')
            self.mean, self.peak = mean, peak
        self.model.eval()
        self.evaluation_only = True
        self.policy_source = str(Path(path).resolve())

    def update(self, x, latent, old_logp, old_mu, target, advantage, returns):
        if self.cfg['train_gru']:
            return self.update_recurrent(x, latent, old_logp, old_mu, target, advantage, returns)
        c = self.cfg
        z, lp, old_mean, td, adv, ret = [torch.tensor(a, dtype=torch.float32) for a in
                                      (latent, old_logp, old_mu, target, advantage, returns)]
        td = (td - self.mean) / self.peak
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        with torch.no_grad():
            if not torch.allclose(self.policy(x), old_mean, rtol=1e-4, atol=1e-6):
                raise ValueError('Policy changed during rollout or cached features are inconsistent.')
        stopped = False
        updates = 0
        for _ in range(int(c['epochs'])):
            order = self.rng.permutation(len(x))
            for start in range(0, len(x), int(c['minibatch'])):
                ix = order[start:start+int(c['minibatch'])]
                mean = self.policy(x[ix])
                logp = Normal(mean, c['action_std']).log_prob(z[ix]).sum(1)
                ratio = torch.exp(logp-lp[ix])
                surrogate = torch.minimum(ratio*adv[ix], ratio.clamp(1-c['clip_ratio'], 1+c['clip_ratio'])*adv[ix])
                imitation = (mean-td[ix]).square().mean()
                loss = -surrogate.mean() + c['imitation_weight']*imitation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite PPO loss.')
                self.actor.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 0.5)
                self.actor.step(); updates += 1
                value_loss = (self.critic(x[ix]).squeeze(1)-ret[ix]).square().mean()
                self.value_optimizer.zero_grad(); value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.value_optimizer.step()
                with torch.no_grad():
                    kl = ((self.policy(x)-old_mean).square()/(2*c['action_std']**2)).sum(1).mean().item()
                if kl > c['target_kl']:
                    stopped = True
                    break
            if stopped:
                break
        return dict(kl=float(kl), updates=updates, early_stop_kl=stopped)

    def update_recurrent(self, x, latent, old_logp, old_mu, target, advantage, returns):
        c = self.cfg
        z, lp, old_mean, td, adv, ret = [torch.tensor(a, dtype=torch.float32) for a in
                                      (latent, old_logp, old_mu, target, advantage, returns)]
        visual = x[:, 128:].detach()
        td = (td-self.mean)/self.peak
        adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
        with torch.no_grad():
            replay = self.recurrent_policy(visual)
            if not torch.allclose(replay,old_mean,rtol=1e-4,atol=1e-6):
                raise ValueError('Chronological GRU replay does not reproduce the rollout policy.')
            expected_lp = Normal(replay,c['action_std']).log_prob(z).sum(1)
            if not torch.allclose(expected_lp,lp,rtol=1e-4,atol=1e-4):
                raise ValueError('Rollout action probabilities do not match recurrent replay.')
        stopped = False
        for epoch in range(int(c['epochs'])):
            # One optimizer step per complete sequence. Recompute all hidden
            # states with the current weights before each gradient calculation.
            mean = self.recurrent_policy(visual)
            logp = Normal(mean,c['action_std']).log_prob(z).sum(1)
            ratio = torch.exp(logp-lp)
            surrogate = torch.minimum(ratio*adv,ratio.clamp(1-c['clip_ratio'],1+c['clip_ratio'])*adv)
            loss = -surrogate.mean()+c['imitation_weight']*(mean-td).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite recurrent PPO loss.')
            self.actor.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],0.5)
            self.actor.step()
            # Critic is separate. Fixed rollout features provide its baseline;
            # gradients must never flow from the critic into the actor GRU.
            value_loss = (self.critic(x.detach()).squeeze(1)-ret).square().mean()
            self.value_optimizer.zero_grad(); value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(),1.0)
            self.value_optimizer.step()
            with torch.no_grad():
                kl = ((self.recurrent_policy(visual)-old_mean).square()/(2*c['action_std']**2)).sum(1).mean().item()
            if kl > c['target_kl']:
                stopped = True
                break
        return dict(kl=float(kl),updates=epoch+1,early_stop_kl=stopped,train_gru=True)


_trainer = None


def configure(checkpoint, stats_path, output_dir, config_json='{}'):
    global _trainer
    _trainer = Trainer(checkpoint, stats_path, output_dir, json.loads(config_json))


def begin_episode(training=False, seed=0):
    _trainer.begin(training, seed)


def step(events):
    return _trainer.step(events)


def end_matlab_evaluation():
    """Return raw sampled torques and release buffers; MATLAB computes metrics."""
    if _trainer.training or not _trainer.pending:
        raise RuntimeError('Expected a pending evaluation, not training.')
    actions = np.asarray([row[3] for row in _trainer.rows], dtype=float)
    _trainer.rows = []
    _trainer.pending = False
    return actions


def finish_episode(time, q, omega, target, applied, reference, last_disturbance=None):
    return json.dumps(_trainer.finish(time, q, omega, target, applied, reference, last_disturbance))


def finish_pd_episode(time, q, omega, target, applied, reference, last_disturbance=None):
    return json.dumps(_trainer.finish(time, q, omega, target, applied, reference, last_disturbance, 'PD'))


def finish_filtered_episode(time, q, omega, target, applied, reference, last_disturbance, alpha):
    return json.dumps(_trainer.finish(time, q, omega, target, applied, reference,
                                     last_disturbance, filter_alpha=float(alpha)))


def save_validation(iteration, count=3):
    return json.dumps(_trainer.save_validation(int(iteration), int(count)))


def load_evaluation_checkpoint(path, stats_path=None):
    _trainer.load_evaluation_checkpoint(path, stats_path)


def comparison_config(source_run, checkpoint, stats_path):
    """Refuse a different supervised baseline or normalization for a paired test."""
    manifest = json.loads((Path(source_run)/'manifest.json').read_text())
    for path, key in [(checkpoint,'checkpoint_sha256'), (stats_path,'stats_sha256')]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f'{key} differs from the training run; comparison aborted.')
    return json.dumps(manifest['config'])


def supervised_comparison_config(source_run, checkpoint):
    """Use a prior paired scenario set while allowing model-specific output scales."""
    manifest = json.loads((Path(source_run)/'manifest.json').read_text())
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
        raise ValueError('The baseline checkpoint differs from the one used by the source run.')
    config = dict(manifest['config'])
    config['recovery_deg'] = 5.0
    config['recovery_hold_s'] = 2.0
    return json.dumps(config)
