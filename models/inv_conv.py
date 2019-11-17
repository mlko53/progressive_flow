import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.
    Args:
        num_channels (int): Number of channels in the input and output.
    """
    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)

        return z, sldj
        

class ConditionalInvConv(nn.Module):
    def __init__(self, num_channels, mid_channels):
        super(ConditionalInvConv, self).__init__()
        self.num_channels = num_channels

        self.NN = NN(num_channels // 2, mid_channels, num_channels**2)

    def forward(self, x, x2, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        weight = self.NN(x2)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)

        return z, sldj

class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, mid_channels, in_dim, out_dim,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels,
                                 kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv = nn.Conv2d(mid_channels, mid_channels,
                                  kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, mid_channels,
                                  kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = self.fc(x)

        return x
