import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class linear_attention(nn.Module):
    def __init__(self,dim, heads=4, dim_head=32, conditiondim=None):
        super().__init__()
        if conditiondim is None:
            conditiondim = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv1d(conditiondim, hidden_dim * 2, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                   )

    def forward(self, q, kv=None, mask=None):
        # b, c, h, w = x.shape
        if kv is None:
            kv = q
        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, "b (h c) t -> b h c t", h=self.heads), (q, k, v)
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c t -> b (h c) t", h=self.heads, )
        return self.to_out(out)

