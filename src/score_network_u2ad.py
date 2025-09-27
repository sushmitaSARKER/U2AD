import torch
import torch.nn as nn
import torch.nn.functional as F
from .dual_pathway_transformer import DualPathwayTransformer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)
        
        
        self.projection_dim = projection_dim
        self.embedding_dim = embedding_dim


    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step.long()]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1)
                               * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat(
            [torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class ScoreNetwork(nn.Module):
    def __init__(
        self,
        marginal_prob_std,
        win_size,
        enc_in,
        c_out,
        num_steps,
        e_layers,
        embedding_dim=128,
        projection_dim=None
    ):
        super(ScoreNetwork, self).__init__()
        self.marginal_prob_std = marginal_prob_std
        self.win_size = win_size
        self.enc_in = enc_in
        self.c_out = c_out
        self.model = DualPathwayTransformer(
            win_size=self.win_size,
            enc_in=self.enc_in,
            c_out=self.c_out,
            e_layers=e_layers,
            marginal_prob_std=self.marginal_prob_std)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps,
            embedding_dim,
            projection_dim
        )

        self.diffusion_projection = nn.Linear(embedding_dim, self.win_size)

    def forward(self, x, diffusion_step):

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)

        x = x + diffusion_emb

        score, global_context, local_context, _ = self.model(x, diffusion_step)
        return score, global_context, local_context, _
