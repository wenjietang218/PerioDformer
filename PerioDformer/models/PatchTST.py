__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone1 import PatchTST_backbone as PatchTST_backbone1
from layers.PatchTST_backbone2 import PatchTST_backbone as PatchTST_backbone2
from layers.PatchTST_backbone3 import PatchTST_backbone as PatchTST_backbone3
from layers.PatchTST_backbone4 import PatchTST_backbone as PatchTST_backbone4
from layers.PatchTST_backbone5 import PatchTST_backbone as PatchTST_backbone5
from layers.PatchTST_backbone6 import PatchTST_backbone as PatchTST_backbone6
from layers.PatchTST_backbone10 import PatchTST_backbone as PatchTST_backbone10
from layers.PatchTST_backbone11 import PatchTST_backbone as PatchTST_backbone11
from layers.PatchTST_backbone12 import PatchTST_backbone as PatchTST_backbone12
from layers.PatchTST_layers import series_decomp


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
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual    #没有
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin            #没有
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        weight_1 = configs.weight_1
        weight_2 = configs.weight_2
        patch_num = configs.patch_num
        week = configs.week
        mlp_num = configs.mlp_num
        d_model2 = configs.d_model2
        d_ff2 = configs.d_ff2
        cross = configs.cross
        experiment = configs.experiment

        percent = configs.percent

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            # self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
            #                       max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
            #                       n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
            #                       dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
            #                       attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            #                       pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
            #                       pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
            #                       weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
            #                       subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone1(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_trend = nn.Linear(context_window, target_window)
            # self.model_res = nn.Linear(context_window, target_window)
        else:
            print("no decomposition")
            if experiment == 1:
                self.model = PatchTST_backbone1(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week, mlp_num=mlp_num,
                                      d_model2=d_model2, d_ff2=d_ff2,
                                      subtract_last=subtract_last, verbose=verbose, **kwargs)
            elif experiment == 2:
                self.model = PatchTST_backbone2(c_in=c_in, context_window=context_window, target_window=target_window,
                                                patch_len=patch_len, stride=stride,
                                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                attn_dropout=attn_dropout,
                                                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                padding_var=padding_var,
                                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                store_attn=store_attn,
                                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                head_dropout=head_dropout, padding_patch=padding_patch,
                                                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                                revin=revin, affine=affine,
                                                weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                                mlp_num=mlp_num,
                                                d_model2=d_model2, d_ff2=d_ff2,
                                                subtract_last=subtract_last, verbose=verbose, **kwargs)
            elif experiment == 3:
                self.model = PatchTST_backbone3(c_in=c_in, context_window=context_window, target_window=target_window,
                                                patch_len=patch_len, stride=stride,
                                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                attn_dropout=attn_dropout,
                                                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                padding_var=padding_var,
                                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                store_attn=store_attn,
                                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                head_dropout=head_dropout, padding_patch=padding_patch,
                                                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                                revin=revin, affine=affine,
                                                weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                                mlp_num=mlp_num,
                                                d_model2=d_model2, d_ff2=d_ff2,
                                                subtract_last=subtract_last, verbose=verbose, **kwargs)
            elif experiment == 4:
                self.model = PatchTST_backbone4(c_in=c_in, context_window=context_window, target_window=target_window,
                                                patch_len=patch_len, stride=stride,
                                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                attn_dropout=attn_dropout,
                                                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                padding_var=padding_var,
                                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                store_attn=store_attn,
                                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                head_dropout=head_dropout, padding_patch=padding_patch,
                                                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                                revin=revin, affine=affine,
                                                weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                                mlp_num=mlp_num,
                                                d_model2=d_model2, d_ff2=d_ff2,
                                                subtract_last=subtract_last, verbose=verbose, **kwargs)
            elif experiment == 5:
                self.model = PatchTST_backbone5(c_in=c_in, context_window=context_window, target_window=target_window,
                                                patch_len=patch_len, stride=stride,
                                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                attn_dropout=attn_dropout,
                                                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                padding_var=padding_var,
                                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                store_attn=store_attn,
                                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                head_dropout=head_dropout, padding_patch=padding_patch,
                                                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                                revin=revin, affine=affine,
                                                weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                                mlp_num=mlp_num,
                                                d_model2=d_model2, d_ff2=d_ff2,
                                                subtract_last=subtract_last, verbose=verbose, **kwargs)
            elif experiment == 6:
                self.model = PatchTST_backbone6(c_in=c_in, context_window=context_window, target_window=target_window,
                                                patch_len=patch_len, stride=stride,
                                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                attn_dropout=attn_dropout,
                                                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                padding_var=padding_var,
                                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                store_attn=store_attn,
                                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                head_dropout=head_dropout, padding_patch=padding_patch,
                                                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                                revin=revin, affine=affine,
                                                weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                                mlp_num=mlp_num,
                                                d_model2=d_model2, d_ff2=d_ff2,
                                                subtract_last=subtract_last, verbose=verbose, **kwargs)
            elif experiment == 10:
                self.model = PatchTST_backbone10(c_in=c_in, context_window=context_window, target_window=target_window,
                                                patch_len=patch_len, stride=stride,
                                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                attn_dropout=attn_dropout,
                                                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                padding_var=padding_var,
                                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                store_attn=store_attn,
                                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                head_dropout=head_dropout, padding_patch=padding_patch,
                                                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                                revin=revin, affine=affine,
                                                weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                                mlp_num=mlp_num,
                                                d_model2=d_model2, d_ff2=d_ff2, cross=cross,
                                                subtract_last=subtract_last, verbose=verbose, **kwargs)
            elif experiment == 11:
                self.model = PatchTST_backbone11(c_in=c_in, context_window=context_window, target_window=target_window,
                                                patch_len=patch_len, stride=stride,
                                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                attn_dropout=attn_dropout,
                                                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                padding_var=padding_var,
                                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                store_attn=store_attn,
                                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                head_dropout=head_dropout, padding_patch=padding_patch,
                                                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                                revin=revin, affine=affine,
                                                weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                                mlp_num=mlp_num,
                                                d_model2=d_model2, d_ff2=d_ff2, cross=cross,
                                                subtract_last=subtract_last, verbose=verbose, **kwargs)
            elif experiment == 12:
                self.model = PatchTST_backbone12(c_in=c_in, context_window=context_window, target_window=target_window,
                                                 patch_len=patch_len, stride=stride,
                                                 max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                 n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=attn_dropout,
                                                 dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                 padding_var=padding_var,
                                                 attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                 store_attn=store_attn,
                                                 pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                 head_dropout=head_dropout, padding_patch=padding_patch,
                                                 pretrain_head=pretrain_head, head_type=head_type,
                                                 individual=individual,
                                                 revin=revin, affine=affine,
                                                 weight_1=weight_1, weight_2=weight_2, patch_num=patch_num, week=week,
                                                 mlp_num=mlp_num,
                                                 d_model2=d_model2, d_ff2=d_ff2, cross=cross,
                                                 subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x