import torch
import torch.nn as nn

__all__ = ("SimAM", "C3_SimAM", "C2f_SimAM_KAN", "KANLayer", "KANBottleneckWithSimAM")

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )
        return x * self.activation(y)

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SimAM_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.sam = SimAM(e_lambda=1e-4)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return (
            x + self.sam(self.cv2(self.cv1(x)))
            if self.add
            else self.sam(self.cv2(self.cv1(x)))
        )

class C3_SimAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(SimAM_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_basis=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.basis_coeff = nn.Parameter(torch.randn(in_dim, out_dim, num_basis))

    def forward(self, x):
        b, c, h, w = x.size()
        powers = x.unsqueeze(-1) ** torch.arange(0, self.num_basis, device=x.device)
        out = torch.einsum('bchwk,cok->bohw', powers, self.basis_coeff)
        return out

class KANBottleneckWithSimAM(nn.Module):
    def __init__(self, c1, c2, shortcut=True, num_basis=3):
        super().__init__()
        self.kan = KANLayer(c1, c1, num_basis)
        self.conv = Conv(c1, c2, 3, 1, None, 1, 1, act=False)
        self.simam = SimAM(e_lambda=1e-4)
        self.shortcut = shortcut and (c1 == c2)

    def forward(self, x):
        residual = x
        x = self.kan(x)
        x = self.conv(x)
        x = self.simam(x)
        return x + residual if self.shortcut else x

class C2f_SimAM_KAN(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, num_basis=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            KANBottleneckWithSimAM(self.c, self.c, shortcut, num_basis) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))