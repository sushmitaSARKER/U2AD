
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_coherence import SimilarityBlock
from .embedding import DataEmbedding

class TriangularCausalMask():
    def __init__(self, B, L, device='cpu'):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(
                    mask_shape, dtype=torch.bool
                ), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self.mask

class CosineSimilarity(nn.Module):
    def __init__(
        self,
        win_size,
        d_model,
        cosine_n_heads=8,
        cosine_dropout=0.1, 
        mask_flag=True,
        scale=None,
        attention_dropout=0.0,
        output_attention=False
    ):
        super(CosineSimilarity, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.win_size = win_size
        self.similarity = SimilarityBlock(
                        win_size,
                        d_model,
                        cosine_n_heads, 
                        cosine_dropout)

    def forward(
        self,
        queries,
        keys,
        values,
        sigma,
        attn_mask
    ):
        x = queries
        B, L, H, E = queries.shape  
        _, S, _, D = values.shape  
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        
        local_context = self.similarity(sigma)
        local_context = local_context.unsqueeze(-1).repeat(1, 1, 1, self.win_size)

        global_context = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", global_context, values)
        if self.output_attention:
            return (V.contiguous(), global_context, local_context, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None
    ):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries,
        keys,
        values,
        attn_mask
    ):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)
        

        out, global_context, local_context, sigma = self.inner_attention(
            queries,+
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), global_context, local_context, sigma
    

class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation='relu'
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or d_model * 4
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(
        self,
        x,
        attn_mask=None
    ):
        new_x, attn, mask, sigma = self.attention(
            x, x, x, attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(
        self,
        attn_layers,
        norm_layer=None
    ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(
        self,
        x,
        attn_mask=None
    ):
        global_context_list = []
        local_context_list = []
        sigma_list = []

        for attn_layer in self.attn_layers:
            x, global_context, local_context, sigma = attn_layer(x, attn_mask=attn_mask)
            global_context_list.append(global_context)
            local_context_list.append(local_context)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, global_context_list, local_context_list, sigma_list


class DualPathwayTransformer(nn.Module):
    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        marginal_prob_std,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation='gelu',
        output_attention=True
    ):
        super(DualPathwayTransformer, self).__init__()
        self.output_attention = output_attention
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        CosineSimilarity(
                            win_size=win_size,
                            d_model=d_model,
                            cosine_n_heads=n_heads//2,
                            mask_flag=False,
                            scale=None,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        enc_out = self.embedding(x)
        enc_out, global_context, local_context, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        enc_out = - enc_out / self.marginal_prob_std(x, t)[1][:, None, None]

        return enc_out, global_context, local_context, sigmas
        