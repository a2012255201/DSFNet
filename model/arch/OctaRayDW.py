import torch
import torch.nn as nn
import torch.nn.functional as F

class CALayer(nn.Module):
    def __init__(self, channel, reduction=1, bias=False):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        reduction = 1
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DiagDWConv(nn.Module):

    def __init__(self, dim, kernel_size, dilation=1, anti_diag=False, bias=True):
        super().__init__()
        assert kernel_size % 2 == 1, "对角卷积核大小必须是奇数，才能保证中心对齐"

        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.anti_diag = anti_diag

        self.weight = nn.Parameter(torch.zeros(dim, kernel_size))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.kernel_size
        r = k // 2
        d = self.dilation
        pad = r * d

        x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

        out = x.new_zeros(b, c, h, w)

        for t in range(k):
            offset = (t - r) * d

            row_start = pad + offset

            if self.anti_diag:
                col_start = pad - offset
            else:
                col_start = pad + offset

            sampled = x_pad[
                :,
                :,
                row_start:row_start + h,
                col_start:col_start + w
            ]

            out = out + sampled * self.weight[:, t].view(1, c, 1, 1)

        if self.bias is not None:
            out = out + self.bias.view(1, c, 1, 1)

        return out

class OctaRayDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel

        self.dw_h = nn.Conv2d(
            dim, dim,
            kernel_size=(h, 1),
            padding='same',
            groups=dim,
            dilation=dilation
        )
        self.dw_w = nn.Conv2d(
            dim, dim,
            kernel_size=(1, w),
            padding='same',
            groups=dim,
            dilation=dilation
        )

        k = max(h, w)
        if k % 2 == 0:
            k += 1

        self.dw_d1 = DiagDWConv(
            dim=dim,
            kernel_size=k,
            dilation=dilation,
            anti_diag=False,
            bias=True
        )
        self.dw_d2 = DiagDWConv(
            dim=dim,
            kernel_size=k,
            dilation=dilation,
            anti_diag=True,
            bias=True
        )

        self.CAB = CALayer(dim)

        nn.init.zeros_(self.dw_d1.weight)
        nn.init.zeros_(self.dw_d2.weight)
        if self.dw_d1.bias is not None:
            nn.init.zeros_(self.dw_d1.bias)
        if self.dw_d2.bias is not None:
            nn.init.zeros_(self.dw_d2.bias)

    def forward(self, x):
        out = x + self.dw_h(x) + self.dw_w(x) + self.dw_d1(x) + self.dw_d2(x)
        out = self.CAB(out)
        return out

class EncoderBlock(nn.Module):

    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.dw = OctaRayDW(in_c, mixer_kernel = (7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):

    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = OctaRayDW(out_c, mixer_kernel = (7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x

class BottleNeckBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        gc = dim//4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = OctaRayDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw2 = OctaRayDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.dw3 = OctaRayDW(gc, mixer_kernel = (3, 3), dilation = 3)

        self.bn = nn.BatchNorm2d(4*gc)
        self.pw2 = nn.Conv2d(4*gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x

class DSFNet(nn.Module):
    def __init__(self):
        super().__init__()

        """Encoder"""
        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(16, 64)

        self.e3 = EncoderBlock(64, 256)

        self.e5 = EncoderBlock(256, 512)

        """Bottle Neck"""
        self.b5 = BottleNeckBlock(512)

        """Decoder"""
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 64)

        self.d2 = DecoderBlock(64, 16)

        self.conv_out = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):

        x = self.conv_in(x)
        x, skip1 = self.e1(x)

        x, skip3 = self.e3(x)

        x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip3)

        x = self.d2(x, skip1)

        x = self.conv_out(x)
        return x
