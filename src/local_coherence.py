import torch
import torch.nn as nn

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, 
        nhead=heads, 
        dim_feedforward=64, 
        activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class get_cosine_similarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, x):
        return self.cos(x[:,None,:,:], x[:,:,None,:])

class SimilarityBlock(nn.Module):
    def __init__(
        self,
        win_size,
        d_model,
        n_heads,
        dropout=0.1
    ):
        super(SimilarityBlock, self).__init__()

        self.cosine_similarity = get_cosine_similarity()
        self.d_model = d_model
        self.cosine_attn = get_torch_trans(
                        heads=n_heads, 
                        layers=1, 
                        channels=win_size)
        
        self.linear1 = nn.Linear(win_size, n_heads * 2)
        self.norm = nn.LayerNorm(n_heads * 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        _, channel, _ = x.shape
        y = self.cosine_similarity(x)
        y = self.cosine_attn(y)
        y = self.dropout(torch.softmax(y, dim=1))
        y = self.linear1(y)
        x = self.norm(x + y)
        x = x.permute(0, 2, 1)
        return x