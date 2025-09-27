import torch
import numpy as np

class EulerMaruyamaPredictor:
    def __init__(self, sde, score_fn, probability_flow=False):
        self.sde = sde
        self.score_fn = score_fn
        self.probability_flow = probability_flow
        self.rsde = self.sde.reverse_prit(self.probability_flow)

    def update_fn(self, x, t, score):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        t = torch.ones(x.shape, device=x.device) * t
        drift, diffusion = self.rsde.sde(x, t, score)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        return x, x_mean
