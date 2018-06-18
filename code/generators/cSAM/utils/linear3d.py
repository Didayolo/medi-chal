"""Layer that takes as input as 2D vector, with 3D params."""
import math
import torch as th
from torch.nn import Parameter


def functional_linear3d(input, weight, bias=None, normalize=False):
    r"""
    Apply a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if normalize:
        output = input.t().matmul(weight/weight.norm(p=2, dim=1, keepdim=True).expand_as(weight))
        if bias is not None:
            output += bias.unsqueeze(1)
    else:
        output = input.t().matmul(weight)
        if bias is not None:
            output += bias.unsqueeze(1)
    return output.t()


class Linear3D(th.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(3, 20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, channels, in_features, out_features, batch_size=-1, bias=True, noise=False, normalize=False):
        super(Linear3D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        if noise:
            self.in_features += 1
        self.weight = Parameter(th.Tensor(channels, self.in_features, out_features))
        if bias:
            self.bias = Parameter(th.Tensor(channels, out_features))
        else:
            self.register_parameter('bias', None)
        if noise:
            self.register_buffer("noise", th.Tensor(batch_size, channels, 1))
        self.normalize = normalize
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_matrix=None):
        if input.dim() == 2:
            if hasattr(self, "noise"):
                input_ = input.unsqueeze(1).expand([input.shape[0], self.channels, self.in_features - 1 ])
            else:
                input_ = input.unsqueeze(1).expand([input.shape[0], self.channels, self.in_features])
            
        else:
            input_ = input

        if adj_matrix is None:
            return functional_linear3d(input_, self.weight, self.bias, normalize=self.normalize)
        else:
            if hasattr(self, "noise"):
                self.noise.normal_()
                # print(th.cat([input_ * adj_matrix.t().unsqueeze(1), self.noise], 2).shape, self.weight.shape)
                return functional_linear3d(th.cat([input_ * adj_matrix.t().unsqueeze(0), self.noise], 2),
                                           self.weight, self.bias, normalize=self.normalize)

            return functional_linear3d(input_ * adj_matrix.t().unsqueeze(0), self.weight, self.bias, normalize=self.normalize)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
