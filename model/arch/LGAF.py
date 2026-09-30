import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

class AxialDW(nn.Module):

    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel

        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same',
                              groups=dim, dilation=dilation)

        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same',
                              groups=dim, dilation=dilation)

        self.CAB = CALayer(dim)

    def forward(self, x):

        x = self.CAB(x + self.dw_h(x) + self.dw_w(x))
        return x

class LGAF(nn.Module):

    def __init__(self, dim, num_heads=4, bias=False):
        super(LGAF, self).__init__()
        self.num_heads = num_heads
        self.factor = num_heads

        self.qkvl = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4, bias=bias)
        )

        self.local_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )

        self.temp_local = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temp_global = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.fuse_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def histogram_attn(self, q, k, v, ifBox, temp):

        b, c, n = q.shape

        pad_len = 0 if n % self.factor == 0 else (n // self.factor + 1) * self.factor - n
        if pad_len > 0:
            q = F.pad(q, (0, pad_len))
            k = F.pad(k, (0, pad_len))
            v = F.pad(v, (0, pad_len))

        curr_n = q.shape[-1]
        hw = curr_n // self.factor

        shape_ori = "b (head c) (f hw)" if ifBox else "b (head c) (hw f)"
        q = rearrange(q, f"{shape_ori} -> b head (c f) hw", head=self.num_heads, f=self.factor, hw=hw)
        k = rearrange(k, f"{shape_ori} -> b head (c f) hw", head=self.num_heads, f=self.factor, hw=hw)
        v = rearrange(v, f"{shape_ori} -> b head (c f) hw", head=self.num_heads, f=self.factor, hw=hw)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * temp
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = rearrange(out, f"b head (c f) hw -> {shape_ori}", head=self.num_heads, f=self.factor, hw=hw)
        return out[..., :n]

    def forward(self, x):
        b, c, h, w = x.shape

        x_in = x.clone()
        x_sort, idx_h = x_in[:, :c//2].sort(dim=-2)
        x_sort, idx_w = x_sort.sort(dim=-1)
        x_in[:, :c//2] = x_sort

        feats = self.qkvl(x_in)
        q, k, v, l = feats.chunk(4, dim=1)

        out_local = self.local_mixer(l)

        v_flat, idx = v.view(b, c, -1).sort(dim=-1)
        q_flat = torch.gather(q.view(b, c, -1), dim=2, index=idx)
        k_flat = torch.gather(k.view(b, c, -1), dim=2, index=idx)

        out1 = self.histogram_attn(q_flat, k_flat, v_flat, True, self.temp_local)
        out2 = self.histogram_attn(q_flat, k_flat, v_flat, False, self.temp_global)

        out_hist = (out1 * out2)
        out_hist = torch.scatter(out_hist, 2, idx, out_hist).view(b, c, h, w)

        gate = self.fuse_gate(torch.cat([out_local, out_hist], dim=1))
        out = out_local * gate + out_hist * (1 - gate)

        out_replace = out[:, :c//2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c//2] = out_replace

        return self.project_out(out) + x

class EncoderBlock(nn.Module):

    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.GELU()

    def forward(self, x):

        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):

    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):

        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x

class BottleNeckBlock(nn.Module):

    def __init__(self, dim, num_heads=4):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)

        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)

        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

        self.LGAF = LGAF(dim=dim, num_heads=num_heads)

    def forward(self, x):

        identity = x

        x1 = self.pw1(x)

        x = torch.cat([x1, self.dw1(x1), self.dw2(x1), self.dw3(x1)], dim=1)
        x = self.act(self.pw2(self.bn(x)))

        x = self.LGAF(x)

        x = x + identity
        return x

class DSFNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')

        self.e1 = EncoderBlock(16, 64)

        self.e3 = EncoderBlock(64, 256)

        self.e5 = EncoderBlock(256, 512)

        self.b5 = BottleNeckBlock(512, num_heads=4)

        self.d5 = DecoderBlock(512, 256)

        self.d4 = DecoderBlock(256, 64)

        self.d2 = DecoderBlock(64, 16)

        self.conv_out = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):

        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip3 = self.e3(x)
        x, skip5 = self.e5(x)

        x = self.b5(x)

        x = self.d5(x, skip5)
        x = self.d4(x, skip3)
        x = self.d2(x, skip1)
        x = self.conv_out(x)
        return x
