"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    return 1.0 - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    return a[t].view(-1, 1, 1, 1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    bar_alpha_t = extract_into_batch(alphas_cumprod, t, x0)
    return torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1.0 - bar_alpha_t) * noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "T": T,
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    return F.mse_loss(noise_pred, noise)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)
    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    assert dim % 2 == 0, "dim must be even"

    half = dim // 2

    if half == 1:
        exponent = torch.zeros(1, device=t.device, dtype=torch.float32)
    else:
        exponent = torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)

    freqs = 10000.0 ** exponent
    args = t.float().unsqueeze(1) / freqs.unsqueeze(0)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    def weight(shape):
        return (0.02 * torch.randn(*shape)).requires_grad_()

    def bias(size):
        return torch.zeros(size, requires_grad=True)

    return {
        "conv_in_w": weight((hidden, in_ch, 3, 3)),
        "conv_in_b": bias(hidden),

        "time_mlp_w": weight((hidden, time_dim)),
        "time_mlp_b": bias(hidden),

        "conv_mid_w": weight((hidden, hidden, 3, 3)),
        "conv_mid_b": bias(hidden),

        "conv_out_w": weight((in_ch, hidden, 3, 3)),
        "conv_out_b": bias(in_ch),
    }

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # Input convolution
    h = F.conv2d(
        x,
        params["conv_in_w"],
        params["conv_in_b"],
        padding=1,
    )

    # Time embedding
    time_dim = params["time_mlp_w"].shape[1]
    temb = timestep_embedding(t, time_dim)
    temb = F.relu(
        F.linear(
            temb,
            params["time_mlp_w"],
            params["time_mlp_b"],
        )
    )

    # Add time embedding to every spatial location
    h = h + temb[:, :, None, None]

    # Hidden layers
    h = F.relu(h)
    h = F.relu(
        F.conv2d(
            h,
            params["conv_mid_w"],
            params["conv_mid_b"],
            padding=1,
        )
    )

    # Output noise prediction
    return F.conv2d(
        h,
        params["conv_out_w"],
        params["conv_out_b"],
        padding=1,
    )

# Step 12 - make_blob_dataset
import torch

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    torch.manual_seed(seed)

    radius = size // 4
    images = torch.zeros((n, 1, size, size), dtype=torch.float32)

    # Coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing="ij"
    )

    for i in range(n):
        cy, cx = torch.randint(radius, size - radius, (2,))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        images[i, 0][mask] = 1.0

    return images

# Step 13 - ddpm_train_step
import torch

def ddpm_train_step(
    params: dict,
    x0,
    schedule: dict,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[dict, float]:

    torch.manual_seed(seed)

    B = x0.shape[0]
    T = schedule["T"]

    # Sample timesteps and Gaussian noise
    t = torch.randint(0, T, (B,), device=x0.device)
    noise = torch.randn_like(x0)

    # Clear any existing gradients
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    # Compute loss
    loss = diffusion_training_loss(
        lambda x, t: tiny_unet_forward(x, t, params),
        x0,
        t,
        noise,
        schedule["alphas_cumprod"],
    )

    # Backpropagate
    loss.backward()

    # SGD update
    new_params = {}
    for name, p in params.items():
        if p.grad is not None:
            new_params[name] = (p - lr * p.grad).detach().requires_grad_(True)
        else:
            new_params[name] = p.detach().clone().requires_grad_(True)

    return new_params, float(loss.item())

# Step 14 - train_ddpm
import torch

def train_ddpm(
    dataset,
    params: dict,
    schedule: dict,
    num_steps: int = 50,
    batch_size: int = 16,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[dict, list]:

    history = []
    n = dataset.shape[0]

    for step in range(num_steps):
        # Seed and sample a minibatch
        torch.manual_seed(seed + step)
        idx = torch.randint(0, n, (batch_size,))
        x0 = dataset[idx]

        # One DDPM training step
        params, loss = ddpm_train_step(
            params,
            x0,
            schedule,
            lr=lr,
            seed=seed + step,
        )

        history.append(loss)

    return params, history

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    bar_alpha_t = extract_into_batch(alphas_cumprod, t, x_t)

    sqrt_bar_alpha = torch.sqrt(bar_alpha_t)
    sqrt_one_minus_bar_alpha = torch.sqrt(1.0 - bar_alpha_t)

    x0_hat = (x_t - sqrt_one_minus_bar_alpha * eps) / sqrt_bar_alpha
    return x0_hat

# Step 16 - ddpm_p_mean_variance
import torch

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    alphas = schedule["alphas"]
    alphas_cumprod = schedule["alphas_cumprod"]
    betas = schedule["betas"]

    # Predict x0 and clamp
    x0_hat = predict_x0_from_eps(x_t, t, eps, alphas_cumprod)
    x0_hat = x0_hat.clamp(-1.0, 1.0)

    # Extract schedule values for current timestep
    alpha_t = extract_into_batch(alphas, t, x_t)
    beta_t = extract_into_batch(betas, t, x_t)
    bar_alpha_t = extract_into_batch(alphas_cumprod, t, x_t)

    # Compute bar_alpha_{t-1}, with bar_alpha_{-1} = 1
    bar_alpha_prev = torch.ones_like(bar_alpha_t)
    mask = t > 0
    if mask.any():
        bar_alpha_prev[mask] = extract_into_batch(
            alphas_cumprod,
            t[mask] - 1,
            x_t[mask],
        )

    # Posterior mean coefficients
    coef1 = torch.sqrt(bar_alpha_prev) * beta_t / (1.0 - bar_alpha_t)
    coef2 = torch.sqrt(alpha_t) * (1.0 - bar_alpha_prev) / (1.0 - bar_alpha_t)

    mean = coef1 * x0_hat + coef2 * x_t
    variance = beta_t

    return mean, variance, x0_hat

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # Sample noise if not provided
    if noise is None:
        noise = torch.randn_like(x_t)

    # Predict noise
    eps = tiny_unet_forward(x_t, t, params)

    # Compute posterior mean and variance
    mean, var, _ = ddpm_p_mean_variance(x_t, t, eps, schedule)

    # No noise added at the final step (t == 0)
    noise = noise * (t > 0).float().view(-1, 1, 1, 1)

    # Reverse diffusion step
    x_prev = mean + torch.sqrt(var) * noise

    return x_prev

# Step 18 - ddpm_sample_loop
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    torch.manual_seed(seed)

    # Start from pure Gaussian noise
    x = torch.randn(shape)

    B = shape[0]
    T = schedule["T"]

    # Reverse diffusion: T-1 -> 0
    for t in range(T - 1, -1, -1):
        t_batch = torch.full((B,), t, dtype=torch.long, device=x.device)
        x = ddpm_p_sample(x, t_batch, params, schedule)

    return x

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

