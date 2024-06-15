import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AttentionPool(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim, hidden_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(hidden_dim, dim, ),
                                    )
        self.positional_embedding = nn.Parameter(torch.randn(1, dim) / dim ** 0.5)

    def forward(self, q):
        # b, c, h, w = x.shape

        q, = map(
            lambda t: rearrange(t, "b c t -> b t c", ), (q,)
        )
        class_token = q.mean(dim=1, keepdim=True) + self.positional_embedding
        q = torch.cat([class_token, q], dim=1)
        kv = q

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=2)

        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.heads), (q, k, v)
        )

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, "b h t c -> b t (h c) ", h=self.heads, )
        return self.to_out(out)[:, 0, :]


class linear_attention_pool(nn.Module):
    def __init__(self,dim, heads=4, dim_head=32, ):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv1d(dim, hidden_dim * 2, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                   )
        self.positional_embedding = nn.Parameter(torch.randn(1, dim) / dim ** 0.5)

    def forward(self, q):
        # b, c, h, w = x.shape

        class_token = q.mean(dim=2, keepdim=True) + self.positional_embedding
        q = torch.cat([class_token, q], dim=2)



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
        return self.to_out(out)[:, :, 0]