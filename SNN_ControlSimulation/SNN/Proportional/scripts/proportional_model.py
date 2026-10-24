"""Small hybrid Conv-LIF/GRU observer; explicit non-leaky error memory."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import snntorch as snn
from snntorch import surrogate


def device_auto():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def preprocess(x, size=64):
    """Channel-first binary ON/OFF; area pooling identical in training/inference."""
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError('Expected [B,2,H,W], ON then OFF')
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError('Events must be finite and nonnegative')
    return F.interpolate((x > 0).float(), (size, size), mode='area')


class ProportionalSNN(nn.Module):
    def __init__(self, size=64, hidden=64, feature_grid=4):
        super().__init__()
        if feature_grid < 1 or feature_grid > ((size+1)//2+1)//2:
            raise ValueError('feature_grid must fit the second convolution output')
        self.config = dict(size=size, hidden=hidden, feature_grid=feature_grid)
        self.conv1 = nn.Conv2d(2, 8, 5, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False)
        kw = dict(beta=0.9, threshold=0.05, spike_grad=surrogate.fast_sigmoid())
        self.lif1, self.lif2 = snn.Leaky(**kw), snn.Leaky(**kw)
        self.gru = nn.GRUCell(32 * feature_grid * feature_grid + 3, hidden)
        self.gate = nn.Linear(hidden + 3, 3)
        self.delta = nn.Linear(hidden + 3, 3)
        nn.init.normal_(self.delta.weight, std=0.01)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def initial(self, batch=1, device=None):
        device = device or next(self.parameters()).device
        n = self.config['size']; a = (n + 1)//2; b = (a + 1)//2
        return (torch.zeros(batch,8,a,a,device=device),
                torch.zeros(batch,16,b,b,device=device),
                torch.zeros(batch,self.config['hidden'],device=device),
                torch.zeros(batch,3,device=device))

    @staticmethod
    def detach(state):
        return tuple(t.detach() for t in state)

    def forward(self, x, state=None):
        # x already area-pooled. No hidden module state is used across episodes.
        if state is None:
            state = self.initial(len(x), x.device)
        m1,m2,h,error = state
        active = (x.flatten(1).sum(1,keepdim=True) > 0)
        s1,n1 = self.lif1(self.conv1(x),m1)
        s2,n2 = self.lif2(self.conv2(s1),m2)
        features = F.adaptive_avg_pool2d(torch.cat((s2,n2),1),(self.config['feature_grid'],self.config['feature_grid'])).flatten(1)
        nh = self.gru(torch.cat((features,error),1),h)
        he = torch.cat((nh,error),1)
        g = torch.sigmoid(self.gate(he))
        # Persistent normalized P-error estimate in closed loop. NEVER reset by silence.
        ne = (1.0 - g) * error + self.delta(he)
        result = (torch.where(active[:,:,None,None],n1,m1),
                  torch.where(active[:,:,None,None],n2,m2),
                  torch.where(active,nh,h), torch.where(active,ne,error))
        return result[-1], result
