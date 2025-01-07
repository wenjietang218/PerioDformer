__all__ = ['PerioDformer_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from layers.PerioDformer_layers import *
from layers.PerioDformer_layers import LinearHeads1, LinearHeads2, LinearHeads3, LinearHeads4, LinearHeads5
from layers.RevIN import RevIN


# Cell
class PerioDformer_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True,
                 revin=True, affine=True,
                 subtract_last=False,
                 w=1, C=24, mlp_num=1, d_model2=16, d_ff2=128,
                 verbose: bool = False, **kwargs):

        super().__init__()

        self.w = w  # the number of copies in data augmentation technique
        self.C = C  # the length of period
        self.period_num = int(context_window / self.C)  # the number of periods of input series

        self.add_flag = 0 # add mean in the end of input series
        if int(self.period_num*self.C) != context_window:
            self.period_num += 1
            self.add_flag = 1
            print("use the mean")

        self.mlp_num = mlp_num  # the number of layers in accurate predictors

        self.num = int((target_window + self.C - 1) / self.C)  # the number of periods of output series
        self.target_window = target_window
        self.contex_window = context_window

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # input tokens
        self.intra_token_len = int(self.w * self.period_num)
        self.intra_token_num = C

        self.inter_token_len = C
        self.inter_token_num = self.period_num

        # Encoder Backbone
        self.intra_encoder = TSTiEncoder(c_in, token_num=self.intra_token_num, token_len=self.intra_token_len,
                                         max_seq_len=max_seq_len,
                                         n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                                         d_ff=d_ff,
                                         attn_dropout=attn_dropout, dropout=dropout, act=act,
                                         key_padding_mask=key_padding_mask, padding_var=padding_var,
                                         attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                         store_attn=store_attn,
                                         pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        self.inter_encoder = TSTiEncoder(c_in, token_num=self.inter_token_num, token_len=self.inter_token_len,
                                         max_seq_len=max_seq_len,
                                         n_layers=n_layers, d_model=d_model2, n_heads=n_heads, d_k=d_k, d_v=d_v,
                                         d_ff=d_ff2,
                                         attn_dropout=attn_dropout, dropout=dropout, act=act,
                                         key_padding_mask=key_padding_mask, padding_var=padding_var,
                                         attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                         store_attn=store_attn,
                                         pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # accurate predictors
        if self.mlp_num == 1:
            self.linears = nn.ModuleList(
                [LinearHeads1(int(d_model+d_model2), 1) for i in range(self.num)])
        elif self.mlp_num == 2:
            self.linears = nn.ModuleList(
                [LinearHeads2(int(d_model+d_model2), 1) for i in range(self.num)])
        elif self.mlp_num == 3:
            self.linears = nn.ModuleList(
                [LinearHeads3(int(d_model+d_model2), 1) for i in range(self.num)])
        elif self.mlp_num == 4:
            self.linears = nn.ModuleList(
                [LinearHeads4(int(d_model+d_model2), 1) for i in range(self.num)])
        elif self.mlp_num == 5:
            self.linears = nn.ModuleList(
                [LinearHeads5(int(d_model+d_model2), 1) for i in range(self.num)])

    def periodic_division(self, z):  # z: [bs x nvars x seq_len]
        if self.add_flag == 1: # use the mean of the previous periodic sequences to supplement the missing time steps at the end
            add_num = int(self.period_num * self.C) - self.contex_window
            for i in range(add_num):
                seq = (self.contex_window + i) % self.C
                sum = z[:, :, 0]
                for j in range(self.period_num - 1):
                    t = z[:, :, int(seq * j)]
                    t = torch.reshape(t, (t.shape[0], t.shape[1], 1))
                    if j == 0:
                        sum = t
                    else:
                        sum += t
                mean = torch.div(sum, self.period_num - 1)
                z = torch.cat((z, mean), dim=-1)

        ar1 = []
        for i in range(self.period_num):
            c = z[:, :, int(i * self.C): int((i + 1) * self.C)]
            c = torch.reshape(c, (c.shape[0], c.shape[1], c.shape[2], 1))
            ar1.append(c)  # ar1[i]:[bs x nvars x C x 1]

        z1 = torch.cat(ar1, dim=-1)  # z1:[bs x nvars x C x period_num]

        for i in range(len(ar1)):
            c = ar1[i]
            for j in range(self.w - 1):
                c = torch.cat((c, ar1[i]), dim=-1)
            ar1[i] = c  # ar[i]:[bs x nvars x C x w]

        z2 = torch.cat(ar1, dim=-1)  # z2:[bs x nvars x C x w*period_num]
        return z2, z1

    def copy_z2(self, z):
        zr = []
        for i in range(self.C):
            zr.append(z)
        z = torch.cat(zr, dim=-1)  # z:[bs x nvars x period_num*dimension x C]
        return z

    def linear_head(self, z1, z2):
        zr = []
        for i in range(self.num):
            z2i = z2[:, :, i]
            z2i = torch.reshape(z2i, (z2i.shape[0], z2i.shape[1], z2i.shape[2], 1))  # z2i:[bs x nvars x d_model2 x 1]
            z2i=self.copy_z2(z2i)
            z2i = z2i.permute(0, 1, 3, 2)
            zi = torch.cat((z1, z2i), dim=-1)  # zi:[bs x nvars x C x (d_model+d_model2)]
            t = self.linears[i](zi)  # t:[bs x nvars x C x 1]
            t = torch.reshape(t, (t.shape[0], t.shape[1], t.shape[2]))
            zr.append(t)

        z = torch.cat(zr, dim=-1)  # z:[bs x nvars x C*num]
        z = z[:, :, :self.target_window]  # z:[bs x nvars x target_window]
        return z

    def encoder1(self, z):
        z = z.permute(0, 1, 3, 2)  # z1:[bs x nvars x (w*period_num) x C]
        z = self.intra_encoder(z)
        z = z.permute(0, 1, 3, 2)  # z1:[bs x nvars x C x d_model]
        return z

    def encoder2(self, z2):
        z2 = self.inter_encoder(z2)  # z2:[bs x nvars x d_model2 x period_num]
        if self.num <= self.period_num:  # z2:[bs x nvars x d_model2 x self.num]
            z2 = z2[:, :, :, :self.num]
        else:
            while z2.shape[3] < self.num:
                add_num = self.num - z2.shape[3]
                add_z = z2[:, :, :, :add_num]
                z2 = torch.cat((z2, add_z), dim=-1)
        z2 = z2.permute(0, 1, 3, 2)  # z2:[bs x nvars x self.num x d_model2]
        return z2

    def norm(self, z):
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        return z

    def denorm(self, z):
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        z = self.norm(z)

        # periodic_divison
        z1, z2 = self.periodic_division(z)  # z1:[bs x nvars x C x w*period_num]  z2:[bs x nvars x C x period_num]

        # intra_encoder
        z1 = self.encoder1(z1)  # z1:[bs x nvars x C x d_model]

        # inter_encoder
        z2 = self.encoder2(z2)  # z2:[bs x nvars x self.num x d_model2]

        # accurate predictors
        z = self.linear_head(z1, z2)  # z:[bs x nvars x target_window]

        # denorm
        z = self.denorm(z)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, token_num, token_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.token_num = token_num
        self.token_len = token_len

        # Input encoding
        q_len = token_num
        self.W_P = nn.Linear(token_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x token_len x token_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x token_num x token_len]
        x = self.W_P(x)  # x: [bs x nvars x token_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x token_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x token_num x d_model]
        # 进行位置编码
        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x token_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x token_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x token_num]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:  # res_attention为True要返回attention矩阵 否则只返回attention的最终结果
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                     attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

