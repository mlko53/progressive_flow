import torch
import torch.nn as nn
import torch.nn.functional as F


class ActNorm(nn.Module):
    """Activation normalization for 2D inputs.
    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_features, scale=1., return_ldj=False):
        super(ActNorm, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6
        self.return_ldj = return_ldj

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            bias = -mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()

        if sldj is not None:
            ldj = logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj

        return x, sldj

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)

        if self.return_ldj:
            return x, ldj

        return x

class ConditionalActNorm(nn.Module):
    def __init__(self, num_features, mid_channels, cond_channels, return_ldj=False):
        super(ConditionalActNorm, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        
        self.nn = NN(cond_channels, mid_channels, num_features)

        self.num_features = num_features
        self.return_ldj = return_ldj

    def _center(self, x, bias, reverse=False):
        if reverse:
            return x - bias
        else:
            return x + bias

    def _scale(self, x, sldj, logs, reverse=False):
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()

        if sldj is not None:
            ldj = logs.mean(0).sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj

        return x, sldj

    def forward(self, x, x2, ldj=None, reverse=False):
        nn_output = self.nn(x2)
        bias, logs = nn_output.chunk(2, dim=1)

        if reverse:
            x, ldj = self._scale(x, ldj, logs, reverse)
            x = self._center(x, bias, reverse)
        else:
            x = self._center(x, bias, reverse)
            x, ldj = self._scale(x, ldj, logs, reverse)

        if self.return_ldj:
            return x, ldj

        return x


def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.
    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.
    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor

class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, mid_channels, out_channels,
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
        self.out_conv = nn.Conv2d(mid_channels, 2 * out_channels,
                                  kernel_size=3, padding=1, bias=True)
        nn.init.normal_(self.out_conv.weight, 0., 0.05)

        self.pool = nn.AdaptiveAvgPool2d(1)
        
        #self.nn = nn.Linear(8 * out_channels, 2 * out_channels)
        #self.out_channels = out_channels


    def forward(self, x):
        #x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        #x = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_conv(x)

        #x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = self.pool(x)
        #x = x.view(-1, 8*self.out_channels)
        #x = self.nn(x)
        #x = x.view(-1, 2*self.out_channels, 1, 1)

        return x
