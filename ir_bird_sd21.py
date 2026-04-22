#!/usr/bin/env python3
"""
IR Blind Enhancement via BIRD + SD 2.1 (Path A: No LoRA, pure SD prior)

方法:
    - Prior: 冻结的 SD 2.1
    - 优化: BIRD noise-space optimization (优化 z_T 和退化参数 θ)
    - 跨模态适配: RGB -> grayscale projection 做 loss, 不动 SD 权重
    - 退化: blur + gamma + FPN 的复合退化族, 全部可微
    - 支持任意宽高比输入

依赖:
    pip install torch diffusers transformers accelerate pillow tqdm numpy pyyaml

用法:
    python ir_bird_sd21.py --input ir.png --output restored.png
    python ir_bird_sd21.py --config my_config.yaml --input ir.png --output restored.png
    python ir_bird_sd21.py --input ir.png --output restored.png \\
        --override optimization.num_steps=300 \\
        --override model.prompt="grayscale photo"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


# =====================================================================
# Config loading
# =====================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list) -> dict:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        key_path, value = ov.split("=", 1)
        keys = key_path.split(".")

        d = config
        for k in keys[:-1]:
            if k not in d:
                raise KeyError(f"Key '{k}' not found at {key_path}")
            d = d[k]

        leaf_key = keys[-1]
        if leaf_key not in d:
            raise KeyError(f"Key '{leaf_key}' not found at {key_path}")

        old_val = d[leaf_key]
        try:
            if isinstance(old_val, bool):
                new_val = value.lower() in ("true", "1", "yes")
            elif isinstance(old_val, int):
                new_val = int(value)
            elif isinstance(old_val, float):
                new_val = float(value)
            else:
                new_val = value
        except ValueError:
            new_val = value

        d[leaf_key] = new_val
        print(f"[override] {key_path}: {old_val} -> {new_val}")
    return config


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }[dtype_str.lower()]


# =====================================================================
# Aspect-ratio-aware resolution computation
# =====================================================================

def compute_processing_size(
    orig_h: int, orig_w: int, long_edge: int, max_long_edge: int,
) -> tuple:
    """
    计算处理分辨率: 保持宽高比, 长边 ≈ long_edge, 两边 round 到 8 的倍数.

    Returns:
        (proc_h, proc_w): 两个 int, 均为 8 的倍数
    """
    long_edge = min(long_edge, max_long_edge)

    # 按宽高比 scale
    if orig_h >= orig_w:
        scale = long_edge / orig_h
        new_h = long_edge
        new_w = int(round(orig_w * scale))
    else:
        scale = long_edge / orig_w
        new_w = long_edge
        new_h = int(round(orig_h * scale))

    # Round 到 8 的倍数 (VAE 要求). 优先向下取整避免过大.
    new_h = max(8, (new_h // 8) * 8)
    new_w = max(8, (new_w // 8) * 8)

    return new_h, new_w


# =====================================================================
# SD 2.1 Prior
# =====================================================================

class SD21Prior(nn.Module):
    """冻结的 SD 2.1 做可微 DDIM 采样."""

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.dtype = get_torch_dtype(model_cfg["dtype"])
        self.device = model_cfg["device"]
        model_id = model_cfg["model_id"]
        prompt = model_cfg["prompt"]

        print(f"[SD21Prior] Loading {model_id} ...")
        # VAE 强制 fp32: SD 2.1 的 VAE 在 fp16 下对 OOD latent 容易出 NaN/黑块,
        # 放大 z_T 漂移带来的伪影.
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=self.dtype
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        for m in [self.vae, self.unet, self.text_encoder]:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

        self.unet.enable_gradient_checkpointing()

        self.text_emb = self._encode_prompt(prompt)
        print(f"[SD21Prior] Prompt: '{prompt}' (emb shape: {self.text_emb.shape})")

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0]

    def ddim_reverse(self, z_T: torch.Tensor, num_steps: int) -> torch.Tensor:
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        z = z_T
        for i, t in enumerate(timesteps):
            z_in = z.to(self.dtype)
            noise_pred = self.unet(
                z_in, t, encoder_hidden_states=self.text_emb
            ).sample.float()

            alpha_t = alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_prev = alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=self.device, dtype=torch.float32)

            x_0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            dir_t = (1 - alpha_prev).sqrt() * noise_pred
            z = alpha_prev.sqrt() * x_0_pred + dir_t
        return z

    def decode(self, z_0: torch.Tensor) -> torch.Tensor:
        z = z_0 / self.vae.config.scale_factor
        # VAE 是 fp32, 这里不再 cast 到 self.dtype
        return self.vae.decode(z.float()).sample.float()


# =====================================================================
# Composite Degradation (支持非方形输入)
# =====================================================================

class CompositeDegradation(nn.Module):
    """Blur + gamma + FPN. 支持任意 H x W."""

    def __init__(self, deg_cfg: dict, reg_cfg: dict, height: int, width: int):
        super().__init__()
        self.kernel_size = deg_cfg["kernel_size"]
        self.enable_fpn = deg_cfg["enable_fpn"]
        self.enable_gamma = deg_cfg["enable_gamma"]
        self.fpn_rank = deg_cfg["fpn_rank"]
        self.height = height
        self.width = width

        self.reg_kernel_l1 = reg_cfg["kernel_l1"]
        self.reg_gamma_dev = reg_cfg["gamma_deviation"]
        self.reg_fpn_l2 = reg_cfg["fpn_l2"]

        # Kernel 近似 delta 初始化 (方形, 与输入宽高比无关)
        init_k = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        init_k[0, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        init_k += 0.001 * torch.randn_like(init_k)
        self.blur_kernel_raw = nn.Parameter(init_k)

        if self.enable_gamma:
            self.gamma = nn.Parameter(torch.tensor(1.0))

        # FPN 低秩分解: row (H, r), col (r, W) -> 外积是 (H, W)
        if self.enable_fpn:
            self.fpn_row = nn.Parameter(0.01 * torch.randn(height, self.fpn_rank))
            self.fpn_col = nn.Parameter(0.01 * torch.randn(self.fpn_rank, width))

    def normalized_kernel(self) -> torch.Tensor:
        k = F.softplus(self.blur_kernel_raw)
        return k / (k.sum() + 1e-8)

    def fpn_pattern(self) -> torch.Tensor:
        """Returns (H, W)."""
        return self.fpn_row @ self.fpn_col

    def forward(self, x_gray: torch.Tensor) -> torch.Tensor:
        """x_gray in [-1, 1] (SD VAE 原生空间)."""
        k = self.normalized_kernel()
        x = F.conv2d(x_gray, k, padding=self.kernel_size // 2)
        if self.enable_gamma:
            g = torch.clamp(self.gamma, 0.3, 3.0)
            # gamma 在负数上没定义, 先 map 到 [0,1] 做 gamma 再 map 回 [-1,1].
            # 用 STE 保留上下界外的梯度.
            x01 = (x + 1.0) / 2.0
            x01_safe = x01 + (x01.clamp(1e-6, 1.0) - x01).detach()
            x = x01_safe.pow(g) * 2.0 - 1.0
        if self.enable_fpn:
            fpn = self.fpn_pattern().unsqueeze(0).unsqueeze(0)
            x = x + fpn
        return x

    def regularizer(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=self.blur_kernel_raw.device)
        k = self.normalized_kernel()
        reg = reg + self.reg_kernel_l1 * k.abs().sum()
        if self.enable_gamma:
            reg = reg + self.reg_gamma_dev * (self.gamma - 1.0).pow(2)
        if self.enable_fpn:
            reg = reg + self.reg_fpn_l2 * (
                self.fpn_row.pow(2).sum() + self.fpn_col.pow(2).sum()
            )
        return reg


# =====================================================================
# BIRD Optimization
# =====================================================================

def rgb_to_gray(x_rgb: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], device=x_rgb.device, dtype=x_rgb.dtype)
    return (x_rgb * w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def restore_ir_image(y: torch.Tensor, prior: SD21Prior, config: dict) -> dict:
    """BIRD 联合优化. y 是任意宽高比的 (1, 1, H, W)."""
    device = prior.device
    opt_cfg = config["optimization"]
    deg_cfg = config["degradation"]
    reg_cfg = config["regularization"]

    H, W = y.shape[-2:]
    assert H % 8 == 0 and W % 8 == 0, f"H, W must be divisible by 8, got {H}x{W}"
    latent_h, latent_w = H // 8, W // 8
    print(f"[restore] Processing at {H}x{W} (latent {latent_h}x{latent_w})")

    z_T = torch.randn(
        1, 4, latent_h, latent_w,
        device=device, dtype=torch.float32, requires_grad=True
    )
    degradation = CompositeDegradation(deg_cfg, reg_cfg, H, W).to(device).float()

    # ---- Warm start ----
    if opt_cfg["warm_start_steps"] > 0:
        print(f"[warm_start] {opt_cfg['warm_start_steps']} steps ...")
        with torch.no_grad():
            z_warm = torch.randn_like(z_T)
            z_0_warm = prior.ddim_reverse(z_warm, opt_cfg["num_ddim_steps"])
            # 全链路 [-1, 1] 对齐 SD VAE 原生空间
            x_rgb_warm = prior.decode(z_0_warm)
            x_gray_warm = rgb_to_gray(x_rgb_warm).clamp(-1, 1)

        theta_opt = torch.optim.Adam(degradation.parameters(), lr=opt_cfg["lr_theta"])
        pbar = tqdm(range(opt_cfg["warm_start_steps"]), desc="warm_start")
        for step in pbar:
            theta_opt.zero_grad()
            loss = F.mse_loss(degradation(x_gray_warm), y) + degradation.regularizer()
            loss.backward()
            theta_opt.step()
            if step % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # ---- Joint optimization ----
    optimizer = torch.optim.Adam([
        {"params": [z_T], "lr": opt_cfg["lr_z"]},
        {"params": degradation.parameters(), "lr": opt_cfg["lr_theta"]},
    ])

    log = {
        "loss": [], "data_loss": [], "reg_loss": [], "z_reg": [],
        "z_mean": [], "z_std": [], "z_absmax": [],
    }

    # 中间 RGB 快照, 用于诊断 SD 输出何时崩坏.
    snapshot_interval = max(1, opt_cfg["num_steps"] // 5)
    rgb_snapshots = []

    print(f"[joint_opt] {opt_cfg['num_steps']} steps ...")
    pbar = tqdm(range(opt_cfg["num_steps"]), desc="joint_opt")
    for step in pbar:
        optimizer.zero_grad()

        z_0 = prior.ddim_reverse(z_T, opt_cfg["num_ddim_steps"])
        x_rgb = prior.decode(z_0)
        x_gray_raw = rgb_to_gray(x_rgb)
        # STE clamp: forward 截到 [-1,1], backward 原样透传梯度. 这样 z_T 能收到
        # "像素越界多少"的信号, 而不是被 clamp 吃掉梯度.
        x_gray = x_gray_raw + (x_gray_raw.clamp(-1, 1) - x_gray_raw).detach()
        y_hat = degradation(x_gray)

        data_loss = F.mse_loss(y_hat, y)
        reg_loss = degradation.regularizer()
        # z_T 高斯正则: 把 z_T 的均值推向 0, 方差推向 1, 防止漂出 SD 的训练流形.
        z_reg = (
            0.1 * (z_T.pow(2).mean() - 1.0).pow(2)
            + 0.01 * z_T.mean().pow(2)
        )
        loss = data_loss + reg_loss + z_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [z_T] + list(degradation.parameters()),
            max_norm=opt_cfg["grad_clip_norm"],
        )
        optimizer.step()

        log["loss"].append(loss.item())
        log["data_loss"].append(data_loss.item())
        log["reg_loss"].append(float(reg_loss.item()))
        log["z_reg"].append(float(z_reg.item()))

        with torch.no_grad():
            z_mean = z_T.mean().item()
            z_std = z_T.std().item()
            z_absmax = z_T.abs().max().item()
        log["z_mean"].append(z_mean)
        log["z_std"].append(z_std)
        log["z_absmax"].append(z_absmax)

        if step % snapshot_interval == 0 or step == opt_cfg["num_steps"] - 1:
            with torch.no_grad():
                # 保存用 [0,1] 空间, 方便 save_image 直接落盘
                snap = ((x_rgb.detach().float() + 1.0) / 2.0).clamp(0, 1).cpu()
                rgb_snapshots.append((step, snap))

        if step % 10 == 0:
            with torch.no_grad():
                rgb_min = x_rgb.min().item()
                rgb_max = x_rgb.max().item()
                rgb_hasnan = bool(torch.isnan(x_rgb).any().item())
            info = {
                "loss": f"{loss.item():.3f}",
                "data": f"{data_loss.item():.3f}",
                "zreg": f"{z_reg.item():.3f}",
                "z": f"{z_mean:+.2f}/{z_std:.2f}/{z_absmax:.1f}",
                "rgb": f"[{rgb_min:+.2f},{rgb_max:+.2f}]{'!NaN' if rgb_hasnan else ''}",
            }
            if deg_cfg["enable_gamma"]:
                info["gamma"] = f"{degradation.gamma.item():.3f}"
            pbar.set_postfix(info)

    # ---- Final ----
    with torch.no_grad():
        z_0_final = prior.ddim_reverse(z_T, opt_cfg["num_ddim_steps"])
        x_rgb_final_m11 = prior.decode(z_0_final).clamp(-1, 1)
        x_gray_final_m11 = rgb_to_gray(x_rgb_final_m11).clamp(-1, 1)
        # 返回给 main 时统一 map 到 [0,1], main 无需感知 [-1,1] 内部约定.
        x_rgb_final = (x_rgb_final_m11 + 1.0) / 2.0
        x_gray_final = (x_gray_final_m11 + 1.0) / 2.0

    return {
        "x_restored": x_gray_final,
        "x_rgb_debug": x_rgb_final,
        "degradation": degradation,
        "log": log,
        "rgb_snapshots": rgb_snapshots,
    }


# =====================================================================
# I/O (支持任意宽高比)
# =====================================================================

def load_ir_image(
    path: str,
    long_edge: int,
    max_long_edge: int,
    force_square: bool = True,
) -> tuple:
    """
    加载 IR 图像. 两种模式:
      - force_square=True (推荐): AR-preserving resize 使最长边 ≤ s=long_edge,
        然后 reflect pad 到 s×s. SD 2.1 base 严格 512×512 训练, 正方形画面质量
        远高于非正方形.
      - force_square=False: 老路径, 保持非正方形处理分辨率.

    Returns:
        tensor: (1, 1, proc_H, proc_W), [0, 1]
        orig_size: (orig_H, orig_W) — 原图尺寸
        proc_size: (proc_H, proc_W) — SD 看到的尺寸 (正方形模式下是 s×s)
        content_bbox: (top, left, fit_h, fit_w) — 真正内容在 tensor 里的矩形.
            非正方形模式下等于 (0, 0, proc_H, proc_W).
    """
    img = Image.open(path).convert("L")
    orig_w, orig_h = img.size  # PIL 是 (W, H), 注意顺序

    if force_square:
        s = min(long_edge, max_long_edge)
        s = max(8, (s // 8) * 8)

        if orig_h >= orig_w:
            scale = s / orig_h
            fit_h = s
            fit_w = max(1, int(round(orig_w * scale)))
        else:
            scale = s / orig_w
            fit_w = s
            fit_h = max(1, int(round(orig_h * scale)))
        # fit_h / fit_w 不必是 8 的倍数, 因为最终 pad 到 s×s, 而 s 是 8 的倍数.

        img_resized = img.resize((fit_w, fit_h), Image.BICUBIC)
        # 与 SD VAE 原生空间对齐: [-1, 1]
        arr = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
        content = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        pad_top = (s - fit_h) // 2
        pad_bottom = s - fit_h - pad_top
        pad_left = (s - fit_w) // 2
        pad_right = s - fit_w - pad_left

        # reflect pad: 给 SD 一个"连续"的画面, 边界没有黑块突变.
        # 注意 F.pad 的顺序是 (last_left, last_right, 2nd_last_top, 2nd_last_bottom)
        tensor = F.pad(
            content,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="reflect",
        )
        return (
            tensor,
            (orig_h, orig_w),
            (s, s),
            (pad_top, pad_left, fit_h, fit_w),
        )

    # Legacy: 非正方形路径
    proc_h, proc_w = compute_processing_size(orig_h, orig_w, long_edge, max_long_edge)
    img_resized = img.resize((proc_w, proc_h), Image.BICUBIC)
    arr = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor, (orig_h, orig_w), (proc_h, proc_w), (0, 0, proc_h, proc_w)


def resize_to_original(
    tensor: torch.Tensor, orig_h: int, orig_w: int
) -> torch.Tensor:
    """把处理分辨率的 tensor resize 回原图尺寸."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    resized = F.interpolate(
        tensor, size=(orig_h, orig_w), mode="bicubic", align_corners=False
    )
    return resized.squeeze(0) if tensor.shape[0] == 1 else resized


def save_image(tensor: torch.Tensor, path):
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    Image.fromarray(arr).save(path)


def save_kernel_vis(degradation: CompositeDegradation, path):
    k = degradation.normalized_kernel().detach().cpu().squeeze().numpy()
    k_norm = (k - k.min()) / (k.max() - k.min() + 1e-8)
    Image.fromarray((k_norm * 255).astype(np.uint8)).resize(
        (256, 256), Image.NEAREST
    ).save(path)


def save_fpn_vis(degradation: CompositeDegradation, path):
    if not degradation.enable_fpn:
        return
    fpn = degradation.fpn_pattern().detach().cpu().numpy()
    fpn_norm = (fpn - fpn.min()) / (fpn.max() - fpn.min() + 1e-8)
    Image.fromarray((fpn_norm * 255).astype(np.uint8)).save(path)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="BIRD + SD 2.1 for IR blind enhancement")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--override", type=str, action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    print(f"[config] Loaded from {args.config}:")
    print(json.dumps(config, indent=2))

    seed = config["optimization"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    prior = SD21Prior(config["model"])

    # Load IR image
    long_edge = config["io"]["long_edge"]
    max_long_edge = config["io"]["max_long_edge"]
    force_square = config["io"].get("force_square", True)
    print(f"[io] Loading: {args.input}")
    y, (orig_h, orig_w), (proc_h, proc_w), content_bbox = load_ir_image(
        args.input,
        long_edge=long_edge,
        max_long_edge=max_long_edge,
        force_square=force_square,
    )
    y = y.to(config["model"]["device"])
    bbox_top, bbox_left, bbox_h, bbox_w = content_bbox
    print(f"[io] Original size: {orig_h}x{orig_w}")
    print(f"[io] Processing size: {proc_h}x{proc_w} (force_square={force_square})")
    print(f"[io] Content bbox in proc canvas: top={bbox_top} left={bbox_left} "
          f"h={bbox_h} w={bbox_w}")
    print(f"[io] IR tensor: {y.shape}, range [{y.min():.3f}, {y.max():.3f}]")

    if not force_square and proc_h != proc_w:
        ar = max(proc_w / proc_h, proc_h / proc_w)
        if ar > 2.0:
            print(f"[warn] Extreme aspect ratio {ar:.2f}:1 may cause SD artifacts.")

    # Run BIRD
    result = restore_ir_image(y, prior, config)

    # 先从 s×s 正方形画布里裁回真正内容区 (非正方形模式下 crop 等于恒等),
    # 然后再 resize 回原图尺寸.
    x_restored = result["x_restored"][
        ..., bbox_top:bbox_top + bbox_h, bbox_left:bbox_left + bbox_w
    ]
    x_rgb_debug = result["x_rgb_debug"][
        ..., bbox_top:bbox_top + bbox_h, bbox_left:bbox_left + bbox_w
    ]
    if config["io"]["restore_original_size"]:
        print(f"[io] Resizing output back to {orig_h}x{orig_w}")
        x_restored = resize_to_original(x_restored, orig_h, orig_w)
        x_rgb_debug = resize_to_original(x_rgb_debug, orig_h, orig_w)

    # Save outputs
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem

    save_image(x_restored, output_path)
    save_image(x_rgb_debug, output_dir / f"{stem}_rgb_debug.png")
    save_kernel_vis(result["degradation"], output_dir / f"{stem}_kernel.png")
    if config["degradation"]["enable_fpn"]:
        save_fpn_vis(result["degradation"], output_dir / f"{stem}_fpn.png")

    # 诊断用中间 RGB 快照
    for snap_step, snap_tensor in result["rgb_snapshots"]:
        save_image(snap_tensor, output_dir / f"{stem}_rgb_step{snap_step:04d}.png")

    summary = {
        "input": str(args.input),
        "output": str(output_path),
        "config_file": str(args.config),
        "original_size": [orig_h, orig_w],
        "processing_size": [proc_h, proc_w],
        "final_loss": result["log"]["loss"][-1],
        "final_data_loss": result["log"]["data_loss"][-1],
        "final_reg_loss": result["log"]["reg_loss"][-1],
        "z_stats": {
            "final_mean": result["log"]["z_mean"][-1],
            "final_std": result["log"]["z_std"][-1],
            "final_absmax": result["log"]["z_absmax"][-1],
            "max_absmax_over_training": max(result["log"]["z_absmax"]),
            "max_std_over_training": max(result["log"]["z_std"]),
        },
        "gamma": (
            result["degradation"].gamma.item()
            if config["degradation"]["enable_gamma"] else None
        ),
        "config": config,
    }
    with open(output_dir / f"{stem}_log.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] Restored: {output_path}")
    print(f"[done] Debug files saved to {output_dir}/")
    print(f"[done] Final data loss: {result['log']['data_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
