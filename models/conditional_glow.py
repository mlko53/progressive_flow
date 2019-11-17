import torch
import torch.nn as nn
import torch.nn.functional as F

from .act_norm import ActNorm
from .coupling import ConditionalCoupling
from .inv_conv import InvConv
from .utils import squeeze


class ConditionalGlow(nn.Module):
    def __init__(self, image_channels, num_channels, num_levels, num_steps):
        super(ConditionalGlow, self).__init__()

        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.flows = _CGlow(in_channels=4 * image_channels,  # RGB image after squeeze
                           mid_channels=num_channels,
                           cond_channels=image_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)

    def forward(self, x, x2, reverse=False):
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'
                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)

        x = squeeze(x)
        x, sldj = self.flows(x, x2, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj

    def _pre_process(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj


class _CGlow(nn.Module):
    def __init__(self, in_channels, mid_channels, cond_channels, num_levels, num_steps):
        super(_CGlow, self).__init__()
        self.steps = nn.ModuleList([_CFlowStep(in_channels=in_channels,
                                              mid_channels=mid_channels,
                                              cond_channels=cond_channels)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _CGlow(in_channels=2 * in_channels,
                              mid_channels=mid_channels,
                              cond_channels=4 * cond_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, x2, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, x2, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x2 = squeeze(x2)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, x2, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)
            x2 = squeeze(x2, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, x2, sldj, reverse)

        return x, sldj


class _CFlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels, cond_channels):
        super(_CFlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = ConditionalCoupling(in_channels // 2, mid_channels, cond_channels)

    def forward(self, x, x2, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, x2, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, x2, sldj, reverse)

        return x, sldj
