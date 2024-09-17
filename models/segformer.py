import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import SegFormerHead, mit_b2

class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b2', pretrained=False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]

        self.backbone   = {
             'b2': mit_b2,
        }[phi](pretrained)

        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]

        self.decode_head = SegFormerHead(self.in_channels, self.embedding_dim)
    
    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone(inputs)
        x = self.decode_head(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x