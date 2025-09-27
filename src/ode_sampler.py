import torch
from scipy import integrate
from .utils import from_flattened_numpy, to_flattened_numpy, get_data_inverse_scaler

class ODESampler:
    def __init__(
            self,
            sde_model,
            device,
            eps=1e-3,
            probability_flow=True,
            denoise=False,
            rtol=1e-5,
            atol=1e-5,
            method='RK45',
    ):
        self.sde = sde_model
        self.probability_flow = probability_flow

        self.rsde = self.sde.create_reverse_sde(self.probability_flow)
        self.denoise = denoise
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.eps = eps
        self.device = device

    def update_fn(self, x, t, score):
        with torch.no_grad():
            def ode_func(t_s, x):
                shape_0 = x.shape[0]
                x = from_flattened_numpy(x, x.shape).to(
                    self.device).type(torch.float32)
                vec_t = torch.ones(shape_0, device=x.device) * t_s
                drift = self.rsde.sde(x, vec_t, score)[0]
                return to_flattened_numpy(drift)

            solution = integrate.solve_ivp(ode_func, (self.sde.T, self.eps), to_flattened_numpy(x),
                                           rtol=self.rtol, atol=self.atol, 
                                           method=self.method)
            
            nfe = solution.nfev
            x = torch.tensor(
                solution.y[:, -1]).reshape(x.shape).to(self.device).type(torch.float32)
            inverse_scaler = get_data_inverse_scaler(data_centered=True)
            x = inverse_scaler(x)
            return x, nfe
