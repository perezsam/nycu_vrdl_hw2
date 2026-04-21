"""Backbone modules for feature extraction.

This module implements the ResNet backbone with multi-scale feature
extraction required for advanced DETR variants like DINO and Deformable DETR.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed.

    This prevents the running means and variances from shifting during training
    when using small batch sizes or heavily augmented datasets.
    """

    def __init__(self, n, eps=1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class MultiScaleBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm returning multiple feature scales."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool,
                 dilation: bool):
        """Initializes the multi-scale backbone.

        Args:
            name (str): Name of the torchvision ResNet model (e.g., 'resnet50').
            train_backbone (bool): Whether to fine-tune the backbone.
            return_interm_layers (bool): If True, returns C3, C4, and C5.
                If False, returns only C5.
            dilation (bool): Whether to replace stride with dilation in the
                last convolutional block.
        """
        super().__init__()
        
        # Load pre-trained ResNet, replacing standard BN with FrozenBN
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1,
            replace_stride_with_dilation=[False, False, dilation],
            norm_layer=FrozenBatchNorm2d
        )
        
        # Determine which layers to extract for DINO/Deformable DETR
        if return_interm_layers:
            # layer2 = C3 (1/8 resolution)
            # layer3 = C4 (1/16 resolution)
            # layer4 = C5 (1/32 resolution)
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        # Freeze parameters if not training the backbone
        if not train_backbone:
            for name, parameter in self.body.named_parameters():
                parameter.requires_grad_(False)
                
        self.num_channels = [512, 1024, 2048] if return_interm_layers else [2048]

    def forward(self, tensor_list: torch.Tensor) -> dict:
        """Executes the forward pass.

        Args:
            tensor_list (torch.Tensor): Batched images of shape [B, C, H, W].

        Returns:
            dict: A dictionary of tensors mapping layer indices to feature maps.
        """
        out = self.body(tensor_list)
        return out


def build_backbone(args) -> MultiScaleBackbone:
    """Builds the multi-scale backbone based on parsed arguments."""
    train_backbone = args.lr_backbone > 0
    # DINO requires intermediate layers (C3, C4) in addition to C5
    return_interm_layers = getattr(args, 'masks', False) or getattr(args, 'dino', True)
    
    backbone = MultiScaleBackbone(
        args.backbone, 
        train_backbone, 
        return_interm_layers, 
        args.dilation
    )
    return backbone