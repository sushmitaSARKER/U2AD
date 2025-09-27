import numpy as np
import torch

class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        # super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

        # Create discrete beta schedule
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1.

    def sde(self, x, t):
        """Compute the drift and diffusion coefficients of the forward SDE.
        Args:
            x: Input tensor.
            t: Time tensor.
        Returns:
            tuple: (drift, diffusion) coefficients.
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Compute the mean and standard deviation of the marginal distribution p_t(x).
        Args:
            x: Input tensor.
            t: Time tensor.
        Returns:
            tuple: (mean, std) of the marginal distribution.
        """
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        """Compute the log-probability of the prior distribution.
        Args:
            z: Input tensor.
        Returns:
            torch.Tensor: Log-probabilities.
        """
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - \
            torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        """DDPM discretization.
        Args:
            x: Input tensor.
            t: Time tensor.
        Returns:
            tuple: (drift, diffusion) for discrete time steps.
        """
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None] * x - x
        G = sqrt_beta
        return f, G

    def create_reverse_sde(self, probability_flow=True):
        """Create the reverse-time SDE/ODE.
        Args:
            probability_flow: If True, create the reverse-time ODE used for probability flow sampling.
        Returns:
            RSDE: Reverse SDE class instance.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, score):
                """Create the drift and diffusion functions for the reverse SDE/ODE.
                Args:
                    x: Input tensor.
                    t: Time tensor.
                    score: Score function output.
                Returns:
                    tuple: (drift, diffusion) for reverse process.
                """
                drift, diffusion = sde_fn(x, t)
                score_flat = torch.flatten(score)
                drift = drift - diffusion ** 2 * score_flat * (0.5 if self.probability_flow else 1.0)
                diffusion = 0.0 if self.probability_flow else diffusion
                
                return drift, diffusion

            def discretize(self, x, t, score):
                """
                Create discretized iteration rules for the reverse diffusion sampler.
                Args:
                    x: Input tensor.
                    t: Time tensor.
                    score: Score function output.
                Returns:
                    tuple: (reverse_drift, reverse_diffusion)
                """
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.0)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()