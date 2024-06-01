__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PerioDformer_backbone1 import PerioDformer_backbone as PerioDformer_backbone1
from layers.PerioDformer_backbone2 import PerioDformer_backbone as PerioDformer_backbone2

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True,
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False,
                 **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        w = configs.w
        C = configs.C
        mlp_num = configs.mlp_num
        d_model2 = configs.d_model2
        d_ff2 = configs.d_ff2
        dimension = configs.dimension
        experiment = configs.experiment

        if experiment == 1:
            self.model = PerioDformer_backbone1(c_in=c_in, context_window=context_window, target_window=target_window,
                                            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                            attn_dropout=attn_dropout,
                                            dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                            padding_var=padding_var,
                                            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                            store_attn=store_attn,
                                            pe=pe, learn_pe=learn_pe,
                                            pretrain_head=pretrain_head, head_type=head_type,
                                            revin=revin, affine=affine,
                                            w=w, C=C,
                                            mlp_num=mlp_num,
                                            d_model2=d_model2, d_ff2=d_ff2,
                                            subtract_last=subtract_last, verbose=verbose, **kwargs)

        elif experiment == 2:
            self.model = PerioDformer_backbone2(c_in=c_in, context_window=context_window, target_window=target_window,
                                            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                            attn_dropout=attn_dropout,
                                            dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                            padding_var=padding_var,
                                            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                            store_attn=store_attn,
                                            pe=pe, learn_pe=learn_pe,
                                            pretrain_head=pretrain_head, head_type=head_type,
                                            revin=revin, affine=affine,
                                            w=w, C=C,
                                            mlp_num=mlp_num,
                                            d_model2=d_model2, d_ff2=d_ff2, dimension=dimension,
                                            subtract_last=subtract_last, verbose=verbose, **kwargs)


    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x