import copy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple


class SAEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, attn_mask, key_padding_mask, pos_embed):
        return self.encoder(src, mask=attn_mask, src_key_padding_mask=key_padding_mask, pos=pos_embed)




class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        output_list = []
        self_attn_wei_list = []
        for layer in self.layers:
            output, self_attn_wei = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            output_list.append(output)
            self_attn_wei_list.append(self_attn_wei)
        if self.norm is not None:
            output = self.norm(output)
        # pdb.set_trace()
        return output, output_list, self_attn_wei_list


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # pdb.set_trace()
        q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2, self_attn_wei = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # pdb.set_trace()

        return src, self_attn_wei

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        # src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2, self_attn_wei = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)                
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, self_attn_wei

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")




# Cross attention Transformer Encoder.
class CXEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = CXTransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = CXTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def forward(self, query, key, mask, pos_embed):
    #     return self.encoder(query, key, src_key_padding_mask=mask, pos=pos_embed)
    def forward(self, query, key, attn_mask, key_padding_mask, q_pos_embed, k_pos_embed):
        return self.encoder(query, key, mask=attn_mask, src_key_padding_mask=key_padding_mask, q_pos=q_pos_embed, k_pos=k_pos_embed)



class CXTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                q_pos: Optional[Tensor] = None,
                k_pos: Optional[Tensor] = None):
        output = query

        output_list = []
        cx_attn_wei_list = []
        for layer in self.layers:
            output, cx_attn_wei = layer(output, key, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, q_pos=q_pos, k_pos=k_pos)
            output_list.append(output)
            cx_attn_wei_list.append(cx_attn_wei)

        # pdb.set_trace()
        if self.norm is not None:
            output = self.norm(output)
        # pdb.set_trace()
        return output, output_list, cx_attn_wei_list


class CXTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.cx_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     query, key,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     q_pos: Optional[Tensor] = None,
                     k_pos: Optional[Tensor] = None):
        src = query
        # pdb.set_trace()
        q = self.with_pos_embed(query, q_pos)
        k = self.with_pos_embed(key, k_pos)
        # src2 = self.cx_attn(q, k, value=k, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2, cx_attn_weights = self.cx_attn(q, k, value=k, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        # pdb.set_trace()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # pdb.set_trace()

        return src, cx_attn_weights

    def forward_pre(self, query, key,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src = query
        q = self.norm1(query)
        k = self.norm1(key)
        q = self.with_pos_embed(q, pos)
        k = self.with_pos_embed(k, pos)
        # src2 = self.cx_attn(q, k, value=k, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2, cx_attn_weights = self.cx_attn(q, k, value=k, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        # pdb.set_trace()
        return src, cx_attn_weights

    def forward(self, query, key,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                q_pos: Optional[Tensor] = None,
                k_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(query, key, src_mask, src_key_padding_mask, q_pos, k_pos)
        return self.forward_post(query, key, src_mask, src_key_padding_mask, q_pos, k_pos)