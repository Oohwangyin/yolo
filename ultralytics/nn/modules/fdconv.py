# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Frequency Dynamic Convolution modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, autopad

__all__ = ("FDConv",)


class StarReLU(nn.Module):
    """StarReLU activation used by the FDConv attention branch."""

    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x).pow(2) + self.bias


def get_fft2freq(d1: int, d2: int, use_rfft: bool = False):
    """Return frequency indices sorted by distance to the DC component."""
    freq_h = torch.fft.fftfreq(d1)
    freq_w = torch.fft.rfftfreq(d2) if use_rfft else torch.fft.fftfreq(d2)
    try:
        freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing="ij"), dim=-1)
    except TypeError:  # torch<1.10 compatibility
        freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    _, indices = torch.sort(dist.reshape(-1))
    d2_eff = d2 // 2 + 1 if use_rfft else d2
    sorted_coords = torch.stack((indices // d2_eff, indices % d2_eff), dim=-1)
    return sorted_coords.permute(1, 0), freq_hw


class KernelSpatialModulationGlobal(nn.Module):
    """Global branch that predicts FDConv channel, filter, spatial and kernel attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int,
        groups: int = 1,
        reduction: float = 0.0625,
        kernel_num: int = 4,
        min_channel: int = 16,
        temp: float = 1.0,
        kernel_temp: float = 1.0,
        att_multi: float = 2.0,
        ksm_only_kernel_att: bool = False,
        act_type: str = "sigmoid",
    ):
        super().__init__()
        c_ = max(int(c1 * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = k
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.att_multi = att_multi
        self.ksm_only_kernel_att = ksm_only_kernel_att

        self.fc = nn.Conv2d(c1, c_, 1, bias=False)
        self.norm = nn.GroupNorm(1, c_)
        self.relu = StarReLU()

        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            self.channel_fc = nn.Conv2d(c_, c1, 1, bias=True)
            self.func_channel = self.get_channel_attention

        if (c1 == groups and c1 == c2) or ksm_only_kernel_att:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(c_, c2, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if k == 1 or ksm_only_kernel_att:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(c_, k * k, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(c_, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name in ("channel_fc", "filter_fc", "spatial_fc", "kernel_fc"):
            m = getattr(self, name, None)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=1e-6)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def skip(_):
        return 1.0

    def _spatial_act(self, x):
        if self.act_type == "sigmoid":
            return torch.sigmoid(x / self.temperature) * self.att_multi
        if self.act_type == "tanh":
            return 1 + torch.tanh(x / self.temperature)
        raise NotImplementedError(f"Unsupported KSM activation: {self.act_type}")

    def get_channel_attention(self, x):
        return self._spatial_act(self.channel_fc(x)).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1))

    def get_filter_attention(self, x):
        return self._spatial_act(self.filter_fc(x)).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1))

    def get_spatial_attention(self, x):
        x = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        return self._spatial_act(x)

    def get_kernel_attention(self, x):
        x = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == "softmax":
            return F.softmax(x / self.kernel_temp, dim=1)
        if self.act_type == "sigmoid":
            return torch.sigmoid(x / self.kernel_temp) * 2 / x.size(1)
        if self.act_type == "tanh":
            return (1 + torch.tanh(x / self.kernel_temp)) / x.size(1)
        raise NotImplementedError(f"Unsupported kernel activation: {self.act_type}")

    def forward(self, x):
        x = self.relu(self.norm(self.fc(x)))
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class KernelSpatialModulationLocal(nn.Module):
    """Local branch that predicts element-wise kernel modulation."""

    def __init__(self, c1: int, kernel_num: int = 1, out_n: int = 1, k_size: int = 3, use_global: bool = False):
        super().__init__()
        self.kernel_num = kernel_num
        self.out_n = out_n
        self.c1 = c1
        k_size = round((math.log2(c1) / 2) + 0.5) // 2 * 2 + 1
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)
        self.use_global = use_global
        if use_global:
            self.complex_weight = nn.Parameter(torch.randn(1, c1 // 2 + 1, 2, dtype=torch.float32) * 1e-6)
        self.norm = nn.LayerNorm(c1)

    def forward(self, x):
        x = x.squeeze(-1).transpose(-1, -2)
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack((x_real, x_imag), dim=-1)), dim=-1)
        x = self.norm(x)
        x = self.conv(x).reshape(x.size(0), self.kernel_num, self.out_n, self.c1)
        return x.permute(0, 1, 3, 2)


class FrequencyBandModulation(nn.Module):
    """Spatially modulate feature frequency bands before FDConv."""

    def __init__(
        self,
        c1: int,
        k_list=(2, 4, 8),
        lowfreq_att: bool = False,
        act: str = "sigmoid",
        spatial_group: int = 1,
        spatial_kernel: int = 3,
        init: str = "zero",
        max_size=(128, 128),
        **kwargs,
    ):
        super().__init__()
        self.k_list = tuple(k_list)
        self.lowfreq_att = lowfreq_att
        self.act = act
        self.spatial_group = c1 if spatial_group > 64 else spatial_group
        n = len(self.k_list) + int(lowfreq_att)
        self.freq_weight_conv_list = nn.ModuleList(
            nn.Conv2d(c1, self.spatial_group, spatial_kernel, padding=spatial_kernel // 2, groups=self.spatial_group)
            for _ in range(n)
        )
        if init == "zero":
            for m in self.freq_weight_conv_list:
                nn.init.normal_(m.weight, std=1e-6)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.register_buffer("cached_masks", self._precompute_masks(max_size), persistent=False)

    def _precompute_masks(self, max_size):
        max_h, max_w = max_size
        _, freq_indices = get_fft2freq(max_h, max_w, use_rfft=True)
        freq_indices = freq_indices.abs().max(dim=-1)[0]
        masks = [freq_indices < 0.5 / freq + 1e-8 for freq in self.k_list]
        return torch.stack(masks, dim=0).unsqueeze(1)

    def sp_act(self, x):
        if self.act == "sigmoid":
            return x.sigmoid() * 2
        if self.act == "tanh":
            return 1 + x.tanh()
        if self.act == "softmax":
            return x.softmax(dim=1) * x.shape[1]
        raise NotImplementedError(f"Unsupported FBM activation: {self.act}")

    def forward(self, x, att_feat=None):
        att_feat = x if att_feat is None else att_feat
        x = x.float()
        pre_x = x.clone()
        b, _, h, w = x.shape
        x_fft = torch.fft.rfft2(x, norm="ortho")
        current_masks = F.interpolate(self.cached_masks.float(), size=(h, w // 2 + 1), mode="nearest").to(x.device)

        x_list = []
        for idx, _ in enumerate(self.k_list):
            low_part = torch.fft.irfft2(x_fft * current_masks[idx], s=(h, w), norm="ortho")
            high_part = pre_x - low_part
            pre_x = low_part
            freq_weight = self.sp_act(self.freq_weight_conv_list[idx](att_feat.float()))
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(
                b, self.spatial_group, -1, h, w
            )
            x_list.append(tmp.reshape(b, -1, h, w))

        if self.lowfreq_att:
            freq_weight = self.sp_act(self.freq_weight_conv_list[len(self.k_list)](att_feat.float()))
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(
                b, self.spatial_group, -1, h, w
            )
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)
        return sum(x_list)


class FrequencyDynamicConv2d(nn.Conv2d):
    """Conv2d with frequency-diverse dynamic weights."""

    def __init__(
        self,
        *args,
        reduction: float = 0.0625,
        kernel_num: int | None = 4,
        use_fdconv_if_c_gt: int = 16,
        use_fdconv_if_k_in=(1, 3),
        use_fbm_if_k_in=(3,),
        use_fbm_for_stride: bool = False,
        kernel_temp: float = 1.0,
        temp: float | None = None,
        att_multi: float = 2.0,
        param_ratio: int = 1,
        param_reduction: float = 1.0,
        ksm_only_kernel_att: bool = False,
        use_ksm_local: bool = True,
        ksm_local_act: str = "sigmoid",
        ksm_global_act: str = "sigmoid",
        convert_param: bool = False,
        fbm_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = tuple(use_fdconv_if_k_in)
        self.use_fbm_if_k_in = tuple(use_fbm_if_k_in)
        self.kernel_num = kernel_num or max(self.out_channels // 2, 1)
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.ksm_local_act = ksm_local_act
        assert ksm_local_act in {"sigmoid", "tanh"}
        assert ksm_global_act in {"softmax", "sigmoid", "tanh"}

        self.active = min(self.in_channels, self.out_channels) > use_fdconv_if_c_gt and self.kernel_size[0] in self.use_fdconv_if_k_in
        if not self.active:
            return

        kernel_temp = math.sqrt(self.kernel_num * self.param_ratio) if kernel_num is None else kernel_temp
        temp = kernel_temp if temp is None else temp
        self.alpha = min(self.out_channels, self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction
        self.KSM_Global = KernelSpatialModulationGlobal(
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            groups=self.groups,
            temp=temp,
            kernel_temp=kernel_temp,
            reduction=reduction,
            kernel_num=self.kernel_num * self.param_ratio,
            att_multi=att_multi,
            ksm_only_kernel_att=ksm_only_kernel_att,
            act_type=ksm_global_act,
        )
        if self.kernel_size[0] in self.use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            self.FBM = FrequencyBandModulation(self.in_channels, **(fbm_cfg or {}))
        if use_ksm_local:
            out_n = int(self.out_channels * self.kernel_size[0] * self.kernel_size[1])
            self.KSM_Local = KernelSpatialModulationLocal(self.in_channels, kernel_num=1, out_n=out_n)
        if convert_param:
            self.convert2dftweight()
        else:
            self._register_frequency_indices()

    def _get_frequency_indices(self):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)
        usable = (freq_indices.size(1) // self.kernel_num) * self.kernel_num
        return freq_indices[:, :usable]

    def _register_frequency_indices(self):
        freq_indices = self._get_frequency_indices()
        indices = [freq_indices.reshape(2, self.kernel_num, -1) for _ in range(self.param_ratio)]
        self.register_buffer("indices", torch.stack(indices, dim=0), persistent=False)

    def convert2dftweight(self):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices = self._get_frequency_indices()
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight.float(), dim=(0, 1))
        weight_rfft = torch.stack((weight_rfft.real, weight_rfft.imag), dim=-1)
        if self.param_reduction < 1:
            keep = int(freq_indices.size(1) * self.param_reduction)
            freq_indices = freq_indices[:, :keep]
            weight_rfft = weight_rfft[freq_indices[0], freq_indices[1]].reshape(-1, 2)[None]
        else:
            weight_rfft = weight_rfft[None]
        self.dft_weight = nn.Parameter(weight_rfft.repeat(self.param_ratio, *([1] * (weight_rfft.dim() - 1))))
        del self.weight
        self._register_frequency_indices()

    def get_FDW(self):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight.float(), dim=(0, 1))
        weight_rfft = torch.stack((weight_rfft.real, weight_rfft.imag), dim=-1)[None]
        return weight_rfft.repeat(self.param_ratio, 1, 1, 1) / max(min(self.out_channels, self.in_channels) // 2, 1)

    def forward(self, x):
        if not self.active:
            return super().forward(x)

        out_dtype = x.dtype
        x = x.float()
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        if self.use_ksm_local:
            hr_att = self.KSM_Local(global_x).reshape(
                x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1]
            )
            hr_att = hr_att.permute(0, 1, 3, 2, 4, 5)
            hr_att = hr_att.sigmoid() * self.att_multi if self.ksm_local_act == "sigmoid" else 1 + hr_att.tanh()
        else:
            hr_att = 1

        b, in_planes, height, width = x.shape
        dft_map = x.new_zeros((b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2))
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        dft_weight = self.dft_weight.float() if hasattr(self, "dft_weight") else self.get_FDW()

        for i in range(self.param_ratio):
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
            else:
                w = dft_weight[i][indices[0], indices[1]][None] * self.alpha
            dft_map[:, indices[0], indices[1]] += torch.stack(
                (w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]), dim=-1
            )

        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(dft_map), dim=(1, 2))
        adaptive_weights = adaptive_weights.reshape(
            b, 1, self.out_channels, self.kernel_size[0], self.in_channels, self.kernel_size[1]
        ).permute(0, 1, 2, 4, 3, 5)

        if hasattr(self, "FBM"):
            x = self.FBM(x)

        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (
            in_planes + self.out_channels
        ) * height * width:
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            aggregate_weight = aggregate_weight.sum(dim=1)
        else:
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = aggregate_weight.sum(dim=1)
            if not isinstance(channel_attention, float):
                x = x * channel_attention.view(b, -1, 1, 1)

        aggregate_weight = aggregate_weight.reshape(-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1])
        x = x.reshape(1, -1, height, width)
        output = F.conv2d(x, aggregate_weight, None, self.stride, self.padding, self.dilation, self.groups * b)
        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] >= (
            in_planes + self.out_channels
        ) * height * width and not isinstance(filter_attention, float):
            output = output * filter_attention.view(b, -1, 1, 1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output.to(out_dtype)


class FDConv(nn.Module):
    """YOLO-style FDConv block: frequency dynamic conv + BN + activation."""

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
        kernel_num=4,
        use_fbm=True,
        convert_param=False,
    ):
        super().__init__()
        self.conv = FrequencyDynamicConv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
            kernel_num=kernel_num,
            use_fbm_if_k_in=(3,) if use_fbm else (),
            convert_param=convert_param,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.forward(x)
