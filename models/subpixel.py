import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .glow import Glow
from .conditional_glow import ConditionalGlow
from .utils import squeeze


class SubPixelFlow(nn.Module):
    def __init__(self, image_channels, num_channels, num_levels, num_steps,
                       size, scale=3):
        super(SubPixelFlow, self).__init__()

        self.masks = get_masks(size, scale)
        self.image_channels = image_channels
        self.base_size = size // 2**(scale-1)
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.flows = [Glow(image_channels=image_channels,
                           num_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)]
        for i in range(1,scale):
            self.flows.append(ConditionalGlow(
                              image_channels=image_channels*3,
                              cond_channels = image_channels,
                              num_channels=num_channels,
                              num_levels=num_levels,
                              num_steps=num_steps))
        self.flows = nn.ModuleList(self.flows)

    def forward(self, x, reverse=False):
        out = torch.zeros(x.shape, device=x.device)
        sldj = torch.zeros(x.shape[0], device=x.device)
        prev = torch.zeros(self.masks[0].shape).byte()
        for i, (flow, mask) in enumerate(zip(self.flows, self.masks)):
            x_res = x[:,:,mask]
            if i == 0:
                x_res = x_res.view(x_res.shape[0], self.image_channels, 
                                   self.base_size, self.base_size)
                z_i, ldj = flow(x_res, reverse)
            else:
                x_cond = out[:,:,prev] if reverse else x[:,:,prev]
                size = int(math.sqrt(x_cond.shape[-1]))
                x_cond = x_cond.view(x.shape[0], self.image_channels,
                                     size, size)
                x_res = x_res.view(x_res.shape[0], self.image_channels * 3, 
                                   size, size)
                z_i, ldj = flow(x_res, x_cond, reverse)
            prev ^= mask
            out[:,:,mask] = z_i.view(z_i.shape[0], self.image_channels, -1)
            sldj += ldj

        return out, sldj

def get_masks(size, scale):

    masks = []
    for i in range(scale):
        stride = int(2**(scale-1-i))
        mask = torch.zeros((size, size))
        for j in range(0,size,stride):
            for k in range(0,size,stride):
                mask[j,k] = 1.
        for prev in masks:
            mask = mask - prev
        masks.append(mask)
    masks = [m.byte() for m in masks]
    return masks

def _pre_process(x):
    """Dequantize the input image `x` and convert to logits.
    See Also:
        - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
        - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
    Args:
        x (torch.Tensor): Input image.
    Returns:
        y (torch.Tensor): Dequantized logits of `x`.
    """
    y = (x * 255. + torch.rand_like(x)) / 256.
    y = (2 * y - 1) * 0.9
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()

    return y
