"""Deformable Transformer module for DINO.

This module implements the Multi-Scale Deformable Transformer architecture,
including native PyTorch implementations of deformable attention.
"""

import math
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

def inverse_sigmoid(x, eps=1e-5):
    """Calculates the inverse sigmoid to project coordinates back to logit space."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def get_sine_pos_embed(pos_tensor, num_pos_feats=128, temperature=10000):
    """Generate sine positional embeddings for 2D maps."""
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos_tensor[:, :, :, 1] / dim_t
    pos_y = pos_tensor[:, :, :, 0] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3)
    return pos


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention module (Native PyTorch)."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = torch.nn.functional.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.')

        # Native PyTorch Grid Sample implementation of MSDA
        output = torch.zeros((N, Len_q, self.n_heads, self.d_model // self.n_heads), device=query.device, dtype=query.dtype)
        for level, (H_, W_) in enumerate(input_spatial_shapes):
            value_l_ = value[:, input_level_start_index[level]:input_level_start_index[level] + H_ * W_].view(N, H_, W_, self.n_heads, -1)
            value_l_ = value_l_.permute(0, 3, 4, 1, 2).flatten(0, 1) # (N*H, D, H_, W_)
            
            sampling_locations_l_ = sampling_locations[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1) # (N*H, Lq, P, 2)
            sampling_locations_l_ = sampling_locations_l_ * 2 - 1 # grid_sample expects [-1, 1]

            sampled_value_l_ = torch.nn.functional.grid_sample(
                value_l_, 
                sampling_locations_l_, 
                mode='bilinear', padding_mode='zeros', align_corners=False
            ) # (N*H, D, Lq, P)

            sampled_value_l_ = sampled_value_l_.permute(0, 2, 3, 1).view(N, self.n_heads, Len_q, self.n_points, -1)
            attention_weights_l_ = attention_weights[:, :, :, level].permute(0, 2, 1, 3).unsqueeze(-1)
            
            output += (sampled_value_l_ * attention_weights_l_).sum(dim=3).permute(0, 2, 1, 3)

        output = output.flatten(2)
        return self.output_proj(output)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = getattr(torch.nn.functional, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = getattr(torch.nn.functional, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(tgt + query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DeformableTransformer(nn.Module):
    """The full Multi-Scale Deformable Transformer."""

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=True,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = False # Essential for standard DINO base
        
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )
        self.encoder = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points
        )
        self.decoder = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def forward(self, srcs, masks, pos_embeds, query_embed=None, bbox_embed=None):
        # Flatten and align all multi-scale features into a single sequence
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # Encoder
        memory = src_flatten
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.encoder:
            memory = layer(memory, lvl_pos_embed_flatten, reference_points, spatial_shapes, level_start_index, mask_flatten)

        # Decoder
        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        
        intermediate = []
        intermediate_reference_points = []
        
        for layer in self.decoder:
            # Pass the current layer's reference points into the decoder layer
            tgt = layer(tgt, query_embed, reference_points.unsqueeze(2), memory, spatial_shapes, level_start_index, mask_flatten)
            
            intermediate.append(tgt)
            intermediate_reference_points.append(reference_points)

            # Iterative Bounding Box Refinement
            if bbox_embed is not None:
                # Detach for gradient stability (Standard Deformable DETR practice)
                reference_points_detached = reference_points.detach()
                
                # Predict the offset using the current layer's features
                tmp = bbox_embed(tgt)
                
                # Add the offset to the inverse-sigmoided reference point
                if reference_points_detached.shape[-1] == 2:
                    tmp[..., :2] += inverse_sigmoid(reference_points_detached)
                else:
                    tmp += inverse_sigmoid(reference_points_detached)
                    
                # Sigmoid to get the new normalized [cx, cy, w, h] for the next layer
                reference_points = tmp.sigmoid()

        # DINO requires initial reference points to calculate layer 0 loss offsets properly.
        # We return intermediate outputs, the initial anchors, and the stack of all refined anchors.
        #return torch.stack(intermediate), intermediate_reference_points[0], torch.stack(intermediate_reference_points)
        return torch.stack(intermediate), intermediate_reference_points[0], intermediate_reference_points
        

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points