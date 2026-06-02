"""Feature-routed PromptIR-NAF/MSFN hybrid for rain/snow restoration.

This model keeps the PromptIR idea of learned restoration prompts, but uses a
feature-aware sparse prompt router instead of copying the canonical
PromptGenBlock concat path. The restoration backbone is NAFNet-style and the
model intentionally has no degradation classifier, task loss, mask head, or
expert routing. Optional MSFN adapters add mixed 3x3/5x5 local modelling for
rain streaks without changing the PromptIR prompt path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
        ctx.eps = eps
        _, channels, _, _ = x.size()
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, channels, 1, 1) * y + bias.view(1, channels, 1, 1)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        eps = ctx.eps
        _, channels, _, _ = grad_output.size()
        y, var, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, channels, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_y = (grad * y).mean(dim=1, keepdim=True)
        grad_x = torch.rsqrt(var + eps) * (grad - y * mean_grad_y - mean_grad)
        grad_weight = (grad_output * y).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        if dw_channels % 2 != 0 or ffn_channels % 2 != 0:
            raise ValueError("Expanded NAF channels must be divisible by two.")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        y = self.dropout2(y)
        return x + y * self.gamma


class NAFStage(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        dw_expand: int,
        ffn_expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *[
                NAFBlock(
                    channels=channels,
                    dw_expand=dw_expand,
                    ffn_expand=ffn_expand,
                    dropout=dropout,
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class MixedScaleFeedForward(nn.Module):
    """DRSformer-inspired mixed-scale feed-forward block."""

    def __init__(self, channels: int, expansion: float = 2.0, bias: bool = False) -> None:
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv3 = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.dwconv5 = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=5,
            padding=2,
            groups=hidden * 2,
            bias=bias,
        )
        self.mix3 = nn.Conv2d(
            hidden * 2,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=bias,
        )
        self.mix5 = nn.Conv2d(
            hidden * 2,
            hidden,
            kernel_size=5,
            padding=2,
            groups=hidden,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden * 2, channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x3_a, x3_b = F.relu(self.dwconv3(x), inplace=False).chunk(2, dim=1)
        x5_a, x5_b = F.relu(self.dwconv5(x), inplace=False).chunk(2, dim=1)
        branch_a = F.relu(self.mix3(torch.cat([x3_a, x5_a], dim=1)), inplace=False)
        branch_b = F.relu(self.mix5(torch.cat([x3_b, x5_b], dim=1)), inplace=False)
        return self.project_out(torch.cat([branch_a, branch_b], dim=1))


class MSFNBlock(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0, bias: bool = False) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.ffn = MixedScaleFeedForward(channels, expansion=expansion, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.gamma * self.ffn(self.norm(x))


class MSFNStage(nn.Module):
    def __init__(self, channels: int, num_blocks: int, expansion: float, bias: bool) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *[MSFNBlock(channels, expansion=expansion, bias=bias) for _ in range(int(num_blocks))]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


def make_hybrid_stage(
    name: str,
    channels: int,
    naf_blocks: int,
    dw_expand: int,
    ffn_expand: int,
    dropout: float,
    msfn_stages: set[str],
    msfn_blocks: int,
    msfn_expansion: float,
    bias: bool,
) -> nn.Module:
    naf_stage = NAFStage(channels, int(naf_blocks), dw_expand, ffn_expand, dropout)
    if name not in msfn_stages or msfn_blocks <= 0:
        return naf_stage
    return nn.Sequential(
        naf_stage,
        MSFNStage(channels, int(msfn_blocks), msfn_expansion, bias),
    )


class Downsample(nn.Module):
    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class FeaturePromptRouter(nn.Module):
    """Sparse PromptIR-style prompt routing from the current decoder feature."""

    def __init__(
        self,
        channels: int,
        prompt_dim: int,
        prompt_len: int,
        prompt_size: int,
        top_k: int = 2,
        strength: float = 0.10,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.top_k = max(1, min(int(top_k), self.prompt_len))
        hidden = max(channels // 2, 32)
        self.prompt_bank = nn.Parameter(
            torch.randn(self.prompt_len, prompt_dim, prompt_size, prompt_size) * 0.02
        )
        self.router = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.prompt_len),
        )
        self.prompt_proj = (
            nn.Identity()
            if prompt_dim == channels
            else nn.Conv2d(prompt_dim, channels, kernel_size=1, bias=bias)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), float(strength)))

    def _sparse_weights(self, logits: Tensor) -> Tensor:
        if self.top_k >= self.prompt_len:
            return logits.softmax(dim=1)
        values, indices = torch.topk(logits, self.top_k, dim=1)
        masked = logits.new_full(logits.shape, float("-inf"))
        masked.scatter_(1, indices, values)
        return masked.softmax(dim=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        logits = self.router(x.mean(dim=(-2, -1)))
        weights = self._sparse_weights(logits)
        prompt = torch.einsum("bp,pchw->bchw", weights, self.prompt_bank)
        prompt = F.interpolate(prompt, size=x.shape[-2:], mode="bilinear", align_corners=False)
        prompt = self.prompt_proj(prompt)
        update = self.fuse(torch.cat([x, prompt], dim=1))
        return x + self.scale * update, {
            "prompt_logits": logits,
            "prompt_weights": weights,
            "prompt_bank": self.prompt_bank,
        }


class PromptIRNAFNetClean(nn.Module):
    """PromptIR U-Net topology with NAFNet blocks and feature-routed prompts."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
        decoder_prompt: bool = True,
        prompt_dims: Sequence[int] = (64, 128, 256),
        prompt_len: int = 5,
        prompt_sizes: Sequence[int] = (48, 24, 12),
        prompt_top_k: int = 2,
        prompt_strength: float = 0.10,
        msfn_stages: Sequence[str] = (),
        msfn_blocks: int = 0,
        msfn_expansion: float = 2.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4:
            raise ValueError("num_blocks must contain four values: [level1, level2, level3, latent].")
        if len(prompt_dims) != 3 or len(prompt_sizes) != 3:
            raise ValueError("prompt_dims and prompt_sizes must contain three values.")
        self.decoder_prompt = bool(decoder_prompt)

        c1, c2, c3, c4 = dim, dim * 2, dim * 4, dim * 8
        pd1, pd2, pd3 = [int(value) for value in prompt_dims]
        ps1, ps2, ps3 = [int(value) for value in prompt_sizes]
        msfn_stage_set = {str(stage) for stage in msfn_stages}

        self.patch_embed = nn.Conv2d(inp_channels, c1, kernel_size=3, padding=1, bias=bias)
        self.encoder_level1 = make_hybrid_stage(
            "encoder_level1", c1, int(num_blocks[0]), dw_expand, ffn_expand, dropout,
            msfn_stage_set, msfn_blocks, msfn_expansion, bias,
        )
        self.down1_2 = Downsample(c1, bias=bias)
        self.encoder_level2 = make_hybrid_stage(
            "encoder_level2", c2, int(num_blocks[1]), dw_expand, ffn_expand, dropout,
            msfn_stage_set, msfn_blocks, msfn_expansion, bias,
        )
        self.down2_3 = Downsample(c2, bias=bias)
        self.encoder_level3 = make_hybrid_stage(
            "encoder_level3", c3, int(num_blocks[2]), dw_expand, ffn_expand, dropout,
            msfn_stage_set, msfn_blocks, msfn_expansion, bias,
        )
        self.down3_4 = Downsample(c3, bias=bias)
        self.latent = make_hybrid_stage(
            "latent", c4, int(num_blocks[3]), dw_expand, ffn_expand, dropout,
            msfn_stage_set, msfn_blocks, msfn_expansion, bias,
        )

        self.up4_3 = Upsample(c4, bias=bias)
        self.reduce_chan_level3 = nn.Conv2d(c3 * 2, c3, kernel_size=1, bias=bias)
        self.prompt3 = FeaturePromptRouter(c3, pd3, prompt_len, ps3, prompt_top_k, prompt_strength, bias)
        self.decoder_level3 = make_hybrid_stage(
            "decoder_level3", c3, int(num_blocks[2]), dw_expand, ffn_expand, dropout,
            msfn_stage_set, msfn_blocks, msfn_expansion, bias,
        )

        self.up3_2 = Upsample(c3, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(c2 * 2, c2, kernel_size=1, bias=bias)
        self.prompt2 = FeaturePromptRouter(c2, pd2, prompt_len, ps2, prompt_top_k, prompt_strength, bias)
        self.decoder_level2 = make_hybrid_stage(
            "decoder_level2", c2, int(num_blocks[1]), dw_expand, ffn_expand, dropout,
            msfn_stage_set, msfn_blocks, msfn_expansion, bias,
        )

        self.up2_1 = Upsample(c2, bias=bias)
        self.reduce_chan_level1 = nn.Conv2d(c1 * 2, c1, kernel_size=1, bias=bias)
        self.prompt1 = FeaturePromptRouter(c1, pd1, prompt_len, ps1, prompt_top_k, prompt_strength, bias)
        self.decoder_level1 = make_hybrid_stage(
            "decoder_level1", c1, int(num_blocks[0]), dw_expand, ffn_expand, dropout,
            msfn_stage_set, msfn_blocks, msfn_expansion, bias,
        )

        self.refinement = make_hybrid_stage(
            "refinement", c1, int(num_refinement_blocks), dw_expand, ffn_expand, dropout,
            msfn_stage_set, msfn_blocks, msfn_expansion, bias,
        )
        self.output = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1, bias=bias)

    def _maybe_prompt(
        self,
        x: Tensor,
        prompt: FeaturePromptRouter,
        aux_list: List[Dict[str, Tensor]],
    ) -> Tensor:
        if not self.decoder_prompt:
            return x
        x, aux = prompt(x)
        aux_list.append(aux)
        return x

    def forward(
        self,
        inp_img: Tensor,
        return_aux: bool = False,
        task_labels: Tensor | None = None,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        del task_labels
        prompt_aux: List[Dict[str, Tensor]] = []

        enc1 = self.encoder_level1(self.patch_embed(inp_img))
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        latent = self.latent(self.down3_4(enc3))

        dec3 = self.up4_3(latent)
        dec3 = self.reduce_chan_level3(torch.cat([dec3, enc3], dim=1))
        dec3 = self._maybe_prompt(dec3, self.prompt3, prompt_aux)
        dec3 = self.decoder_level3(dec3)

        dec2 = self.up3_2(dec3)
        dec2 = self.reduce_chan_level2(torch.cat([dec2, enc2], dim=1))
        dec2 = self._maybe_prompt(dec2, self.prompt2, prompt_aux)
        dec2 = self.decoder_level2(dec2)

        dec1 = self.up2_1(dec2)
        dec1 = self.reduce_chan_level1(torch.cat([dec1, enc1], dim=1))
        dec1 = self._maybe_prompt(dec1, self.prompt1, prompt_aux)
        dec1 = self.decoder_level1(dec1)
        dec1 = self.refinement(dec1)
        out = self.output(dec1) + inp_img

        if not return_aux:
            return out
        return out, {"prompt_aux": prompt_aux}

    def prompt_parameters(self) -> List[nn.Parameter]:
        if not self.decoder_prompt:
            return []
        return [
            self.prompt1.prompt_bank,
            self.prompt2.prompt_bank,
            self.prompt3.prompt_bank,
        ]


def build_model(cfg: Dict[str, Any]) -> PromptIRNAFNetClean:
    prompt_len = int(_get(cfg, "prompt_len", _get(cfg, "num_prompts", 5)))
    return PromptIRNAFNetClean(
        inp_channels=int(_get(cfg, "inp_channels", 3)),
        out_channels=int(_get(cfg, "out_channels", 3)),
        dim=int(_get(cfg, "dim", 48)),
        num_blocks=tuple(_get(cfg, "num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=int(_get(cfg, "num_refinement_blocks", 4)),
        dw_expand=int(_get(cfg, "dw_expand", 2)),
        ffn_expand=int(_get(cfg, "ffn_expand", 2)),
        dropout=float(_get(cfg, "dropout", _get(cfg, "drop_path", 0.0))),
        decoder_prompt=bool(_get(cfg, "decoder_prompt", True)),
        prompt_dims=tuple(_get(cfg, "prompt_dims", [64, 128, 256])),
        prompt_len=prompt_len,
        prompt_sizes=tuple(_get(cfg, "prompt_sizes", [48, 24, 12])),
        prompt_top_k=int(_get(cfg, "prompt_top_k", _get(cfg, "top_k", 2))),
        prompt_strength=float(_get(cfg, "prompt_strength", 0.10)),
        msfn_stages=tuple(_get(cfg, "msfn_stages", [])),
        msfn_blocks=int(_get(cfg, "msfn_blocks", 0)),
        msfn_expansion=float(_get(cfg, "msfn_expansion", 2.0)),
        bias=bool(_get(cfg, "bias", False)),
    )
