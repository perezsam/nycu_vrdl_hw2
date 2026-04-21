"""DINO / Deformable DETR core architecture.

This module wraps the multi-scale backbone and deformable transformer, 
handling feature projection, positional embeddings, and bounding box decoding.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from models.backbone import build_backbone
from models.deformable_transformer import DeformableTransformer


def inverse_sigmoid(x, eps=1e-5):
    """Calculates the inverse sigmoid to project coordinates back to logit space."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class PositionEmbeddingSine(nn.Module):
    """2D Sine Positional Embedding for multi-scale feature maps."""
    
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, tensor_list: torch.Tensor):
        x = tensor_list
        b, c, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MLP(nn.Module):
    """Multi-layer perceptron for prediction heads."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DINO(nn.Module):
    """The DINO (Deformable DETR variant) model."""

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=True):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.aux_loss = aux_loss
        hidden_dim = transformer.d_model
        
        # 1. Prediction Heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # 2. Query Embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        
        # 3. Multi-Scale Feature Projection
        num_backbone_outs = len(backbone.num_channels)
        input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = backbone.num_channels[i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        self.input_proj = nn.ModuleList(input_proj_list)
        
        self.backbone = backbone
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # 4. CRITICAL INITIALIZATION FIX
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Initialize class embedding to predict background heavily at the start (Prior Prob = 0.01)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes + 1) * bias_value
        
        # Initialize BBox embed to strictly 0.0 so reference points do not explode in Epoch 1
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, samples: torch.Tensor):
        features = self.backbone(samples)
        
        srcs = []
        masks = []
        pos_embeds = []
        
        for i, (feat_name, feat) in enumerate(features.items()):
            srcs.append(self.input_proj[i](feat))
            mask = torch.zeros((feat.shape[0], feat.shape[2], feat.shape[3]), 
                               dtype=torch.bool, device=feat.device)
            masks.append(mask)
            pos_embeds.append(self.pos_embed(feat))

        hs, init_reference, inter_references = self.transformer(
            srcs, masks, pos_embeds, self.query_embed.weight, self.bbox_embed
        )

        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(hs.shape[0]):
            # CRITICAL DESYNC FIX: Use the exact reference points passed into this specific layer
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference)
            
            outputs_class = self.class_embed(hs[lvl])
            
            tmp = self.bbox_embed(hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                
            outputs_coord = tmp.sigmoid()
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {
            "pred_logits": outputs_class[-1], 
            "pred_boxes": outputs_coord[-1]
        }
        
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
            
        return out


def build_dino(args):
    """Factory function to build the DINO model."""
    backbone = build_backbone(args)
    
    transformer = DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=1024, # DINO standard
        num_feature_levels=3, # C3, C4, C5
    )
    
    model = DINO(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    return model