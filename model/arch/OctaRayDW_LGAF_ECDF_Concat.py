import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ExplicitFrequencyModel(nn.Module):
    """
    显式频域建模：对幅度与相位分别进行可学习调制，再回投影到空间域
    """
    def __init__(self, dim):
        super().__init__()
        self.amp_net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        self.phase_net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        # 用空间特征生成幅度门控，增强跨域交互稳定性
        self.amp_gate = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x, cond=None):
        b, c, h, w = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        amp = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        if cond is None:
            cond = x
        cond_freq = F.interpolate(cond, size=(h, amp.size(-1)), mode='bilinear', align_corners=False)

        amp_delta = self.amp_net(amp)
        phase_delta = self.phase_net(phase)
        gate = torch.sigmoid(self.amp_gate(cond_freq))

        amp = amp + amp_delta * gate
        phase = phase + phase_delta

        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        x_recon = torch.complex(real, imag)
        x_freq_spatial = torch.fft.irfft2(x_recon, s=(h, w), norm='backward')
        return x_freq_spatial, amp, phase


class CrossDomainAttention(nn.Module):
    """
    双输入跨域注意力：q来自source，k/v来自target
    """
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, source, target):
        b, c, h, w = source.shape
        assert source.shape == target.shape, 'CrossDomainAttention: source和target形状必须一致'
        assert c % self.num_heads == 0, 'CrossDomainAttention: 通道数必须能被head数整除'

        q = self.q_dw(self.q(source))
        kv = self.kv_dw(self.kv(target))
        k, v = kv.chunk(2, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class ExplicitCrossDomainFusion(nn.Module):
    """
    Comment 6 拼接对照：通道拼接后用 1x1 卷积投影，保留相同的细化模块
    """
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.project = nn.Conv2d(2 * dim, dim, 1, 1, 0)
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        )

    def forward(self, spatial_feat, freq_feat):
        fused = self.project(torch.cat([spatial_feat, freq_feat], dim=1))
        return fused + self.refine(fused)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=1, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
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
    """
    真对角Depthwise卷积：
    只存储每个通道的k个对角权重，而不是完整k×k权重。
    理论参数量: C × k
    理论MACs: H × W × C × k
    """
    def __init__(self, dim, kernel_size, dilation=1, anti_diag=False, bias=True):
        super().__init__()
        assert kernel_size % 2 == 1, "对角卷积核大小必须是奇数，才能保证中心对齐"

        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.anti_diag = anti_diag

        # 只保存对角线上的k个权重
        self.weight = nn.Parameter(torch.zeros(dim, kernel_size))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.kernel_size
        r = k // 2
        d = self.dilation
        pad = r * d

        # F.pad格式: left, right, top, bottom
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

        # 保留原始轴向能力
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

        # 新增“米字形”中的两条对角分支
        # 对角线卷积必须用方形核，这里取max(h, w)
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

        # 关键：新增分支零初始化
        # 这样初始时输出与原OctaRayDW完全一致，不破坏原能力
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
    
class LGAF(nn.Module):
    """
    LGAF: Local-Global Adaptive Fusion 
    融合了 CAFM 的局部卷积增强与 DHSA 的直方图自注意力机制。
    """
    def __init__(self, dim, num_heads=4, bias=False):
        super(LGAF, self).__init__()
        self.num_heads = num_heads
        self.factor = num_heads
        
        # 1. 共享特征预处理 (合并 DHSA 和 CAFM 的预处理步)
        # 输出 4 个通道部分：Q, K, V 和用于局部卷积的特征 L
        self.qkvl = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4, bias=bias)
        )

        # 2. 局部增强分支 (吸收 CAFM 思想)
        # 模仿 CAFM 的局部细节提取，但使用更轻量的 2D 形式
        self.local_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )

        # 3. 直方图注意力分支 (吸收 DHSA 思想)
        self.temp_local = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temp_global = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # 4. 融合阶段
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def histogram_attn(self, q, k, v, ifBox, temp):
        """ 核心直方图重排逻辑 """
        b, c, n = q.shape
        # 填充以满足分块要求
        pad_len = 0 if n % self.factor == 0 else (n // self.factor + 1) * self.factor - n
        if pad_len > 0:
            q = F.pad(q, (0, pad_len))
            k = F.pad(k, (0, pad_len))
            v = F.pad(v, (0, pad_len))
        
        curr_n = q.shape[-1]
        hw = curr_n // self.factor
        
        # 根据 ifBox 选择重排策略 (Local Box vs Global Interleaved)
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
        
        # --- 步骤 1: 特征生成 ---
        # 模拟 DHSA 的空间排序预处理（仅对前一半通道）
        x_in = x.clone()
        x_sort, idx_h = x_in[:, :c//2].sort(dim=-2)
        x_sort, idx_w = x_sort.sort(dim=-1)
        x_in[:, :c//2] = x_sort
        
        feats = self.qkvl(x_in)
        q, k, v, l = feats.chunk(4, dim=1)

        # --- 步骤 2: 局部卷积分支 (CAFM) ---
        out_local = self.local_mixer(l)

        # --- 步骤 3: 直方图注意力分支 (DHSA) ---
        # 基于 Value 的值进行排序索引（DHSA 核心）
        v_flat, idx = v.view(b, c, -1).sort(dim=-1)
        q_flat = torch.gather(q.view(b, c, -1), dim=2, index=idx)
        k_flat = torch.gather(k.view(b, c, -1), dim=2, index=idx)

        # 计算双路注意力
        out1 = self.histogram_attn(q_flat, k_flat, v_flat, True, self.temp_local)
        out2 = self.histogram_attn(q_flat, k_flat, v_flat, False, self.temp_global)
        
        # 恢复原始顺序并合并
        out_hist = (out1 * out2)
        out_hist = torch.scatter(out_hist, 2, idx, out_hist).view(b, c, h, w)

        # --- 步骤 4: 自适应融合 ---
        # 门控机制融合局部与全局特征
        gate = self.fuse_gate(torch.cat([out_local, out_hist], dim=1))
        out = out_local * gate + out_hist * (1 - gate)
        
        # 空间反排序（还原位置）
        out_replace = out[:, :c//2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c//2] = out_replace
        
        return self.project_out(out) + x # 残差连接


class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.dw = OctaRayDW(in_c, mixer_kernel = (7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.freq_model = ExplicitFrequencyModel(in_c)
        self.cross_domain_fuse = ExplicitCrossDomainFusion(in_c, num_heads=4)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()

    def forward(self, x):
        spatial_feat = self.bn(self.dw(x))
        freq_feat, _, _ = self.freq_model(spatial_feat, spatial_feat)
        skip = self.cross_domain_fuse(spatial_feat, freq_feat)
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = OctaRayDW(out_c, mixer_kernel = (7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.freq_model = ExplicitFrequencyModel(out_c)
        self.cross_domain_fuse = ExplicitCrossDomainFusion(out_c, num_heads=4)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        spatial_feat = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        freq_feat, _, _ = self.freq_model(spatial_feat, spatial_feat)
        x = self.cross_domain_fuse(spatial_feat, freq_feat)
        return x
    
class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""
    def __init__(self, dim, num_heads=4):
        super().__init__()

        gc = dim//4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = OctaRayDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw2 = OctaRayDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.dw3 = OctaRayDW(gc, mixer_kernel = (3, 3), dilation = 3)

        self.bn = nn.BatchNorm2d(4*gc)
        self.pw2 = nn.Conv2d(4*gc, dim, kernel_size=1)
        self.act = nn.GELU()

        # LGAF全局-局部特征融合
        self.LGAF = LGAF(dim=dim, num_heads=num_heads)

    def forward(self, x):
        identity = x
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        x = self.LGAF(x)
        x = x + identity
        return x

class FluxNet(nn.Module):
    """
    主网络：LU2Net
    结构：输入卷积 → 编码器 → 瓶颈(LGGF) → 解码器 → 输出卷积
    任务：图像复原/增强类任务（输入3通道，输出3通道）
    Input:  [B, 3, H, W]
    Output: [B, 3, H, W]
    """
    def __init__(self):
        super().__init__()
        # 输入卷积：3→16
        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        # 编码器：16→64（下采样×2）
        self.e1 = EncoderBlock(16, 64)
        # 编码器：64→256（下采样×2）
        self.e3 = EncoderBlock(64, 256)
        # 编码器：256→512（下采样×2）
        self.e5 = EncoderBlock(256, 512)

        # 瓶颈块：内含LGGF全局-局部融合
        self.b5 = BottleNeckBlock(512, num_heads=4)
        
        # 解码器：512→256（上采样×2）
        self.d5 = DecoderBlock(512, 256)
        # 解码器：256→64（上采样×2）
        self.d4 = DecoderBlock(256, 64)
        # 解码器：64→16（上采样×2）
        self.d2 = DecoderBlock(64, 16)
        # 输出卷积：16→3
        self.conv_out = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        # 输入: [B, 3, H, W]
        x = self.conv_in(x)                # [B, 16, H, W]
        x, skip1 = self.e1(x)              # x:[B,64,H/2,W/2], skip1:[B,16,H,W]
        x, skip3 = self.e3(x)              # x:[B,256,H/4,W/4], skip3:[B,64,H/2,W/2]
        x, skip5 = self.e5(x)              # x:[B,512,H/8,W/8], skip5:[B,256,H/4,W/4]

        x = self.b5(x)                     # [B, 512, H/8, W/8] (LGGF增强)

        x = self.d5(x, skip5)              # [B, 256, H/4, W/4]
        x = self.d4(x, skip3)              # [B, 64, H/2, W/2]
        x = self.d2(x, skip1)              # [B, 16, H, W]
        x = self.conv_out(x)               # [B, 3, H, W]
        return x