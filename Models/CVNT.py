import torch
import torch.nn as nn

from module_lib.attention.base_attention import Attention
from module_lib.conv.base_conv import conform_conv


class conform_ffn(nn.Module):
    def __init__(self, dim, DropoutL1: float = 0.1, DropoutL2: float = 0.1):
        super().__init__()
        self.ln1 = nn.Linear(dim, dim * 4)
        self.ln2 = nn.Linear(dim * 4, dim)
        self.drop1 = nn.Dropout(DropoutL1) if DropoutL1 > 0. else nn.Identity()
        self.drop2 = nn.Dropout(DropoutL2) if DropoutL2 > 0. else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.ln2(x)
        return self.drop2(x)



class conform_blocke_full(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, conv_drop: float = 0.1, ffn_latent_drop: float = 0.1,
                 ffn_out_drop: float = 0.1, attention_drop: float = 0.1, attention_heads: int = 4,
                 attention_heads_dim: int = 64):
        super().__init__()
        self.ffn1 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)
        self.ffn2 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)
        self.att = Attention(dim, heads=attention_heads, dim_head=attention_heads_dim)
        self.attdrop = nn.Dropout(attention_drop) if attention_drop > 0. else nn.Identity()
        self.conv = conform_conv(dim, kernel_size=kernel_size,

                                 DropoutL=conv_drop, )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        # self.norm5 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, ):
        x = self.ffn1(self.norm1(x)) * 0.5 + x

        x = self.attdrop(self.att(self.norm2(x), mask=mask)) + x
        # x = self.norm1(self.ffn1(x) + x)
        x = self.conv(self.norm3(x)) + x
        x = self.ffn2(self.norm4(x)) * 0.5 + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        return x

class CVNT(nn.Module):
    def __init__(self, config, output_size=2):
        super().__init__()
        self.config = config
        model_arg = config['model_arg']
        self.in_k_size = model_arg.get('in_k_size', 1)
        self.inlinear = nn.Linear(config['spec_win'],
                                  model_arg['encoder_conform_dim']) if self.in_k_size == 1 else nn.Conv1d(
            config['spec_win'], model_arg['encoder_conform_dim'], kernel_size=self.in_k_size,
            padding=self.in_k_size // 2)

        norm_type = model_arg.get('norm_type', 'pre_n')
        if norm_type == 'FULL_CF':
            self.enc = nn.ModuleList([conform_blocke_full(
                dim=model_arg['encoder_conform_dim'],
                kernel_size=model_arg['encoder_conform_kernel_size'],
                ffn_latent_drop=model_arg['encoder_conform_ffn_latent_drop'],
                ffn_out_drop=model_arg['encoder_conform_ffn_out_drop'],

                attention_drop=model_arg['encoder_conform_attention_drop'],
                attention_heads=model_arg['encoder_conform_attention_heads'],
                attention_heads_dim=model_arg['encoder_conform_attention_heads_dim']

            ) for _ in range(model_arg['num_layers'])])
        self.outlinear = nn.Linear(model_arg['encoder_conform_dim'], output_size)
        self.use_final_norm = model_arg.get('use_final_norm', False)
        self.mel_scal = model_arg.get('mel_scal', 1)
        if self.use_final_norm:
            self.final_norm = nn.LayerNorm(model_arg['encoder_conform_dim'])

    def forward(self, x, mask=None, ):
        # x=torch.transpose(x,1,2)
        x = x / self.mel_scal
        if self.in_k_size == 1:
            x = self.inlinear(x)
        else:
            x = self.inlinear(x.transpose(1, 2)).transpose(1, 2)
        for i in self.enc:
            x = i(x, mask=mask)
        if self.use_final_norm:
            x = self.final_norm(x)
        x = self.outlinear(x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = torch.transpose(x, 1, 2)

        return x
