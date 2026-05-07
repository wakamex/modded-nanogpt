"""
Local Track 3 proxy runner.

This keeps the Track 3 model family, FineWeb tokenization, Muon matrix
optimizer, and Adam-style non-matrix optimizer split. By default it scales the
model and batch down so a single 24GB GPU can run short optimizer ablations;
the ordinal preset keeps the benchmark architecture shape and rescales reduced
local batches to the benchmark gradient magnitude.
"""

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

with open(sys.argv[0]) as f:
    code = f.read()


########################################
#              Utilities               #
########################################

def dist_ready():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if dist_ready() else 1

def get_rank():
    return dist.get_rank() if dist_ready() else 0

def is_master():
    return get_rank() == 0

def maybe_all_reduce(x: Tensor):
    if dist_ready():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


########################################
#              Dataloader              #
########################################

def download_fineweb10b(data_dir: Path, train_chunks: int):
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)

    def get(fname):
        if not (data_dir / fname).exists():
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=data_dir,
            )

    get("fineweb_val_%06d.bin" % 0)
    for i in range(1, train_chunks + 1):
        get("fineweb_train_%06d.bin" % i)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, f"magic number mismatch in {file}"
    assert header[1] == 1, f"unsupported version in {file}"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, f"number of tokens read does not match header in {file}"
    return tokens

def data_generator(data_dir: Path, split: str, batch_tokens: int, seq_len: int, device: torch.device):
    world_size = get_world_size()
    rank = get_rank()
    files = sorted(data_dir.glob(f"fineweb_{split}_*.bin"))
    assert files, f"No FineWeb {split} shards found in {data_dir}"
    assert batch_tokens % (world_size * seq_len) == 0
    local_batch_tokens = batch_tokens // world_size
    assert local_batch_tokens % seq_len == 0
    file_idx = 0
    tokens, pos = _load_data_shard(files[file_idx]), 0
    while True:
        if pos + batch_tokens + 1 >= len(tokens):
            file_idx = (file_idx + 1) % len(files)
            tokens, pos = _load_data_shard(files[file_idx]), 0
        buf = tokens[pos + rank * local_batch_tokens:][:local_batch_tokens + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += batch_tokens
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        angular_freq = (1 / seq_len) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, seq_len: int):
        super().__init__()
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim, seq_len)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=0.12, is_causal=True,
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = Linear(dim, 4 * dim)
        self.proj = Linear(4 * dim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, seq_len: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, seq_len)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int, seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, seq_len) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizers              #
########################################

def zeropower_via_newtonschulz5(G: Tensor, ns_iters: int) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(ns_iters):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def first_moment_variance_scale(beta1: float, step: int) -> float:
    """Var(m_hat) / Var(g) for an EMA first moment under iid gradient noise."""
    beta1_t = beta1**step
    return (1 - beta1) ** 2 * (1 - beta1 ** (2 * step)) / ((1 - beta1**2) * (1 - beta1_t) ** 2)

def snr_shrinkage_gate(m_hat: Tensor, s_hat: Tensor, lambda_effective) -> Tensor:
    r = m_hat.square() / s_hat.clamp_min(torch.finfo(s_hat.dtype).tiny)
    return r / (r + lambda_effective)

@torch.compile
def muon_update(grad, momentum, mu: float, ns_iters: int, nesterov: bool = True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update, ns_iters)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.025, weight_decay=0.0125, mu=0.95, ns_iters=12):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, ns_iters=ns_iters)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = get_world_size()
        rank = get_rank()
        for group in self.param_groups:
            params = group["params"]
            if world_size == 1:
                for p in params:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], group["mu"], group["ns_iters"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                continue
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], group["mu"], group["ns_iters"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

def symmetric_inverse_power(factor: Tensor, exponent: float, eps: float) -> Tensor:
    factor = 0.5 * (factor + factor.T)
    diag_mean = factor.diagonal().mean().abs().clamp_min(eps)
    factor = factor / diag_mean
    eigvals, eigvecs = torch.linalg.eigh(factor.float())
    scales = eigvals.clamp_min(eps).pow(-exponent)
    return (eigvecs * scales.unsqueeze(0)) @ eigvecs.T

class PMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.035,
        weight_decay=0.025,
        mu=0.95,
        ns_iters=12,
        beta=0.95,
        gamma=0.3,
        eps=1e-8,
    ):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            ns_iters=ns_iters,
            beta=beta,
            gamma=gamma,
            eps=eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _update_covariances(state, grad: Tensor, beta: float) -> None:
        row_cov = grad @ grad.T
        col_cov = grad.T @ grad
        if "row_cov" not in state:
            state["row_cov"] = row_cov
            state["col_cov"] = col_cov
        else:
            state["row_cov"].mul_(beta).add_(row_cov, alpha=1 - beta)
            state["col_cov"].mul_(beta).add_(col_cov, alpha=1 - beta)

    @torch.no_grad()
    def step(self):
        world_size = get_world_size()
        rank = get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params if world_size == 1 else params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            stride = 1 if world_size == 1 else world_size
            for base_i in range(0, len(params), stride):
                if world_size == 1:
                    p = params[base_i]
                elif base_i + rank < len(params):
                    p = params[base_i + rank]
                else:
                    p = None
                if p is not None:
                    grad = p.grad.detach().float()
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    self._update_covariances(state, grad, group["beta"])
                    momentum = state["momentum"]
                    momentum.lerp_(grad, 1 - group["mu"])
                    update = grad.lerp(momentum, group["mu"])
                    row_power = symmetric_inverse_power(state["row_cov"], group["gamma"], group["eps"])
                    col_power = symmetric_inverse_power(state["col_cov"], group["gamma"], group["eps"])
                    update = row_power @ update @ col_power
                    update = orthogonalized_matrix_update(update, group["ns_iters"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                if world_size > 1:
                    dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

class KFACFactorStore:
    def __init__(self, modules, rho=0.95, damping=0.03, refresh_steps=1):
        self.rho = rho
        self.damping = damping
        self.refresh_steps = max(refresh_steps, 1)
        self.states = {}
        self.handles = []
        for module in modules:
            state = {"module": module, "step": 0}
            self.states[module.weight] = state
            self.handles.append(module.register_forward_hook(self._forward_hook(state)))
            self.handles.append(module.register_full_backward_hook(self._backward_hook(state)))

    @staticmethod
    def _flatten_features(x: Tensor) -> Tensor:
        return x.detach().float().reshape(-1, x.size(-1))

    @staticmethod
    def _accumulate_cov(state, prefix: str, x: Tensor) -> None:
        flat = KFACFactorStore._flatten_features(x)
        cov = (flat.T @ flat).mul_(1.0 / max(flat.size(0), 1))
        sum_key = f"{prefix}_sum"
        count_key = f"{prefix}_count"
        if sum_key in state:
            state[sum_key].add_(cov)
            state[count_key] += 1
        else:
            state[sum_key] = cov
            state[count_key] = 1

    def _forward_hook(self, state):
        def hook(module, inputs, output):
            if not module.training or not torch.is_grad_enabled():
                return
            with torch.no_grad():
                self._accumulate_cov(state, "a", inputs[0])
        return hook

    def _backward_hook(self, state):
        def hook(module, grad_input, grad_output):
            if not module.training or not grad_output or grad_output[0] is None:
                return
            with torch.no_grad():
                self._accumulate_cov(state, "g", grad_output[0])
        return hook

    @staticmethod
    def _sync_average(x: Tensor) -> Tensor:
        if dist_ready():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x.mul_(1.0 / get_world_size())
        return x

    def _update_ema(self, state, prefix: str) -> bool:
        sum_key = f"{prefix}_sum"
        count_key = f"{prefix}_count"
        if sum_key not in state:
            return False
        batch = state.pop(sum_key).mul_(1.0 / state.pop(count_key))
        self._sync_average(batch)
        ema_key = f"{prefix}_ema"
        if ema_key not in state:
            state[ema_key] = batch
        else:
            state[ema_key].mul_(self.rho).add_(batch, alpha=1 - self.rho)
        return True

    def _inverse_factor(self, factor: Tensor) -> Tensor:
        factor = 0.5 * (factor + factor.T)
        diag_mean = factor.diagonal().mean().abs().clamp_min(1e-8)
        eye = torch.eye(factor.size(0), device=factor.device, dtype=factor.dtype)
        damping = self.damping * diag_mean
        for scale in (1.0, 10.0, 100.0, 1000.0):
            damped = factor + (damping * scale + 1e-8) * eye
            chol, info = torch.linalg.cholesky_ex(damped, check_errors=False)
            if not bool(info.any().item()):
                return torch.cholesky_inverse(chol)
        return torch.linalg.pinv(factor + (damping * 1000.0 + 1e-8) * eye)

    @torch.no_grad()
    def update_factors(self) -> None:
        for state in self.states.values():
            updated_a = self._update_ema(state, "a")
            updated_g = self._update_ema(state, "g")
            if not (updated_a and updated_g):
                continue
            state["step"] += 1
            if state["step"] == 1 or state["step"] % self.refresh_steps == 0:
                state["a_inv"] = self._inverse_factor(state["a_ema"])
                state["g_inv"] = self._inverse_factor(state["g_ema"])

    @torch.no_grad()
    def precondition(self, p: Tensor, update: Tensor) -> Tensor:
        state = self.states.get(p)
        if state is None or "a_inv" not in state or "g_inv" not in state:
            return update.float()
        return state["g_inv"] @ update.float() @ state["a_inv"]

def kfac_linear_modules(model: nn.Module):
    return [module for module in model.blocks.modules() if isinstance(module, Linear)]

def orthogonalized_matrix_update(update: Tensor, ns_iters: int) -> Tensor:
    out = zeropower_via_newtonschulz5(update, ns_iters)
    out *= max(1, update.size(-2) / update.size(-1))**0.5
    return out

class KFACMuon(torch.optim.Optimizer):
    def __init__(self, params, factor_store, lr=0.025, weight_decay=0.0125, mu=0.95, ns_iters=12):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, ns_iters=ns_iters, factor_store=factor_store)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        factor_store = self.param_groups[0]["factor_store"]
        factor_store.update_factors()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().float()
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                momentum = state["momentum"]
                momentum.lerp_(grad, 1 - group["mu"])
                update = grad.lerp(momentum, group["mu"])
                update = factor_store.precondition(p, update)
                update = orthogonalized_matrix_update(update, group["ns_iters"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

class KFAC(torch.optim.Optimizer):
    def __init__(self, params, factor_store, lr=0.001, weight_decay=0.025, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, factor_store=factor_store)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        factor_store = self.param_groups[0]["factor_store"]
        factor_store.update_factors()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().float()
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                momentum = state["momentum"]
                momentum.lerp_(grad, 1 - group["mu"])
                update = grad.lerp(momentum, group["mu"])
                update = factor_store.precondition(p, update)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

@torch.no_grad()
def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10) -> None:
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)

class AdamH(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.018,
        betas=(0.9, 0.95),
        eps=1e-8,
        poprisk=False,
        rho=0.95,
        alpha=1.0,
        lambda_pop=1.0,
        lambda_mode="fixed",
        lambda_final=0.0,
        lambda_decay_start=0,
        lambda_decay_end=1,
        target_q=0.5,
        warmup_steps=20,
        gate="snr",
        kfac_store=None,
    ):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        if poprisk and lambda_mode != "fixed" and gate != "snr":
            raise ValueError("adaptive lambda modes are currently defined only for --pop-gate snr")
        if not 0 < target_q < 1:
            raise ValueError("target_q must be strictly between 0 and 1")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            poprisk=poprisk,
            rho=rho,
            alpha=alpha,
            lambda_pop=lambda_pop,
            lambda_mode=lambda_mode,
            lambda_final=lambda_final,
            lambda_decay_start=lambda_decay_start,
            lambda_decay_end=lambda_decay_end,
            target_q=target_q,
            warmup_steps=warmup_steps,
            gate=gate,
            kfac_store=kfac_store,
        )
        super().__init__(params, defaults)
        self.last_gate_stats = {}

    @staticmethod
    def _scheduled_lambda(group, step):
        if group["lambda_mode"] != "cosine-decay":
            return group["lambda_pop"]
        start = group["lambda_decay_start"]
        end = group["lambda_decay_end"]
        if step <= start:
            return group["lambda_pop"]
        if step >= end:
            return group["lambda_final"]
        progress = (step - start) / max(end - start, 1)
        weight = 0.5 * (1 + math.cos(math.pi * progress))
        return group["lambda_final"] + weight * (group["lambda_pop"] - group["lambda_final"])

    @torch.no_grad()
    def step(self):
        kfac_store = self.param_groups[0].get("kfac_store")
        if kfac_store is not None:
            kfac_store.update_factors()
        snr_chunks = []
        lambda_mode = self.param_groups[0]["lambda_mode"]
        target_q = self.param_groups[0]["target_q"]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            rho = group["rho"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                    if group["poprisk"]:
                        state["s"] = torch.zeros_like(p, dtype=torch.float32)
                m = state["m"]
                v = state["v"]
                s = state.get("s")
                state["step"] += 1
                step = state["step"]

                if group["poprisk"]:
                    diff = grad - m
                    s.mul_(rho).addcmul_(diff, diff, value=1 - rho)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["poprisk"] and lambda_mode == "target-median-q":
                    m_hat = m / (1 - beta1**step)
                    s_hat = s / (1 - rho**step)
                    snr_chunks.append((m_hat.square() / (s_hat + eps)).flatten())

        adaptive_lambda = None
        if lambda_mode == "target-median-q" and snr_chunks:
            r_median = torch.median(torch.cat(snr_chunks))
            adaptive_lambda = r_median * (1 - target_q) / target_q

        gate_sum = 0.0
        gate_numel = 0
        gate_low = 0
        gate_high = 0
        lambda_sum = 0.0
        lambda_numel = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            rho = group["rho"]
            alpha = group["alpha"]
            warmup_steps = group["warmup_steps"]
            gate_kind = group["gate"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                step = state["step"]
                m = state["m"]
                v = state["v"]
                m_hat = m / (1 - beta1**step)
                v_hat = v / (1 - beta2**step)
                update = m_hat / (v_hat.sqrt() + eps)

                if group["poprisk"]:
                    s = state["s"]
                    s_hat = s / (1 - rho**step)
                    margin = m_hat.square() - alpha * s_hat
                    lambda_effective = adaptive_lambda if adaptive_lambda is not None else self._scheduled_lambda(group, step)
                    if gate_kind == "snr-wiener":
                        lambda_effective = 1.0
                        q = snr_shrinkage_gate(m_hat, s_hat, lambda_effective)
                    elif gate_kind == "snr-var":
                        lambda_effective = first_moment_variance_scale(beta1, step)
                        q = snr_shrinkage_gate(m_hat, s_hat, lambda_effective)
                    elif step <= warmup_steps:
                        q = torch.ones_like(m_hat, dtype=torch.float32)
                    elif gate_kind == "hard":
                        q = (margin > 0).to(dtype=torch.float32)
                    elif gate_kind == "snr":
                        q = m_hat.square() / (m_hat.square() + lambda_effective * s_hat + eps)
                    else:
                        delta = margin.clamp_min(0)
                        q = delta / (delta + lambda_effective * s_hat + eps)
                    update = q * update

                    qf = q.float()
                    q_numel = qf.numel()
                    lambda_value = float(lambda_effective.item() if isinstance(lambda_effective, Tensor) else lambda_effective)
                    gate_sum += float(qf.sum().item())
                    gate_numel += q_numel
                    gate_low += int((qf < 0.01).sum().item())
                    gate_high += int((qf > 0.99).sum().item())
                    lambda_sum += lambda_value * q_numel
                    lambda_numel += q_numel

                if kfac_store is not None:
                    update = kfac_store.precondition(p, update)
                scale_invariant_update_(p, update, group["lr"])

        self.last_gate_stats = {}
        if gate_numel:
            self.last_gate_stats = {
                "q_mean": gate_sum / gate_numel,
                "q_lt_0.01": gate_low / gate_numel,
                "q_gt_0.99": gate_high / gate_numel,
                "lambda_pop": lambda_sum / max(lambda_numel, 1),
            }

class PopRiskAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        betas=(0.8, 0.95),
        rho=0.95,
        eps=1e-10,
        weight_decay=0.0,
        alpha=1.0,
        lambda_pop=1.0,
        lambda_mode="fixed",
        lambda_final=0.0,
        lambda_decay_start=0,
        lambda_decay_end=1,
        target_q=0.5,
        warmup_steps=20,
        gate="soft",
    ):
        if lambda_mode != "fixed" and gate != "snr":
            raise ValueError("adaptive lambda modes are currently defined only for --pop-gate snr")
        if not 0 < target_q < 1:
            raise ValueError("target_q must be strictly between 0 and 1")
        defaults = dict(
            betas=betas,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            lambda_pop=lambda_pop,
            lambda_mode=lambda_mode,
            lambda_final=lambda_final,
            lambda_decay_start=lambda_decay_start,
            lambda_decay_end=lambda_decay_end,
            target_q=target_q,
            warmup_steps=warmup_steps,
            gate=gate,
        )
        super().__init__(params, defaults)
        self.last_gate_stats = {}

    @staticmethod
    def _scheduled_lambda(group, step):
        if group["lambda_mode"] != "cosine-decay":
            return group["lambda_pop"]
        start = group["lambda_decay_start"]
        end = group["lambda_decay_end"]
        if step <= start:
            return group["lambda_pop"]
        if step >= end:
            return group["lambda_final"]
        progress = (step - start) / max(end - start, 1)
        weight = 0.5 * (1 + math.cos(math.pi * progress))
        return group["lambda_final"] + weight * (group["lambda_pop"] - group["lambda_final"])

    @torch.no_grad()
    def step(self):
        snr_chunks = []
        lambda_mode = self.param_groups[0]["lambda_mode"]
        target_q = self.param_groups[0]["target_q"]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                    state["s"] = torch.zeros_like(p, dtype=torch.float32)
                m = state["m"]
                v = state["v"]
                s = state["s"]
                state["step"] += 1
                step = state["step"]

                diff = grad - m
                s.mul_(rho).addcmul_(diff, diff, value=1 - rho)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if lambda_mode == "target-median-q":
                    m_hat = m / (1 - beta1**step)
                    s_hat = s / (1 - rho**step)
                    snr_chunks.append((m_hat.square() / (s_hat + eps)).flatten())

        adaptive_lambda = None
        if lambda_mode == "target-median-q" and snr_chunks:
            r_median = torch.median(torch.cat(snr_chunks))
            adaptive_lambda = r_median * (1 - target_q) / target_q

        gate_sum = 0.0
        gate_numel = 0
        gate_low = 0
        gate_high = 0
        lambda_sum = 0.0
        lambda_numel = 0
        group_stats = []
        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            alpha = group["alpha"]
            warmup_steps = group["warmup_steps"]
            gate_kind = group["gate"]
            local_sum = 0.0
            local_numel = 0
            local_low = 0
            local_high = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state["m"]
                v = state["v"]
                s = state["s"]
                step = state["step"]

                m_hat = m / (1 - beta1**step)
                v_hat = v / (1 - beta2**step)
                s_hat = s / (1 - rho**step)
                margin = m_hat.square() - alpha * s_hat
                lambda_effective = adaptive_lambda if adaptive_lambda is not None else self._scheduled_lambda(group, step)

                if gate_kind == "snr-wiener":
                    lambda_effective = 1.0
                    q = snr_shrinkage_gate(m_hat, s_hat, lambda_effective)
                elif gate_kind == "snr-var":
                    lambda_effective = first_moment_variance_scale(beta1, step)
                    q = snr_shrinkage_gate(m_hat, s_hat, lambda_effective)
                elif step <= warmup_steps:
                    q = torch.ones_like(m_hat, dtype=torch.float32)
                elif gate_kind == "hard":
                    q = (margin > 0).to(dtype=torch.float32)
                elif gate_kind == "snr":
                    q = m_hat.square() / (m_hat.square() + lambda_effective * s_hat + eps)
                else:
                    delta = margin.clamp_min(0)
                    q = delta / (delta + lambda_effective * s_hat + eps)

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                update = q * m_hat / (v_hat.sqrt() + eps)
                p.add_(update.to(dtype=p.dtype), alpha=-group["lr"])

                qf = q.float()
                q_numel = qf.numel()
                q_sum = float(qf.sum().item())
                low = int((qf < 0.01).sum().item())
                high = int((qf > 0.99).sum().item())
                lambda_value = float(lambda_effective.item() if isinstance(lambda_effective, Tensor) else lambda_effective)
                gate_sum += q_sum
                gate_numel += q_numel
                gate_low += low
                gate_high += high
                lambda_sum += lambda_value * q_numel
                lambda_numel += q_numel
                local_sum += q_sum
                local_numel += q_numel
                local_low += low
                local_high += high
            if local_numel:
                group_stats.append({
                    "group": group_idx,
                    "q_mean": local_sum / local_numel,
                    "q_lt_0.01": local_low / local_numel,
                    "q_gt_0.99": local_high / local_numel,
                })
        self.last_gate_stats = {
            "q_mean": gate_sum / max(gate_numel, 1),
            "q_lt_0.01": gate_low / max(gate_numel, 1),
            "q_gt_0.99": gate_high / max(gate_numel, 1),
            "lambda_pop": lambda_sum / max(lambda_numel, 1),
            "groups": group_stats,
        }


########################################
#                Setup                 #
########################################

def default_data_dir():
    nvme = Path("/nvme2/modded-nanogpt-data/fineweb10B")
    return nvme if nvme.exists() else Path("data/fineweb10B")

PROXY_PRESETS = {
    "small": {},
    "ordinal-3090": {
        "num_layers": 12,
        "model_dim": 768,
        "head_dim": 128,
        "batch_tokens": 32768,
        "microbatch_seqs": 1,
        "val_tokens": 262144,
        "val_interval": 50,
        "log_interval": 10,
        "reference_batch_tokens": 8 * 64 * 1024,
    },
    "realbatch-3090": {
        "num_layers": 12,
        "model_dim": 768,
        "head_dim": 128,
        "train_steps": 4875,
        "stop_after_step": 100,
        "batch_tokens": 8 * 64 * 1024,
        "microbatch_seqs": 1,
        "val_tokens": 1024 * 1024,
        "val_interval": 50,
        "log_interval": 10,
        "reference_batch_tokens": 8 * 64 * 1024,
    },
}

def _arg_was_provided(argv, dest: str) -> bool:
    flag = "--" + dest.replace("_", "-")
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)

def apply_proxy_preset(args, argv):
    for dest, value in PROXY_PRESETS[args.proxy_preset].items():
        if not _arg_was_provided(argv, dest):
            setattr(args, dest, value)
    return args

def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU/local proxy for Track 3 optimizer tests")
    parser.add_argument("--proxy-preset", choices=sorted(PROXY_PRESETS), default="small",
                        help="Apply a reusable local proxy shape before explicit CLI overrides")
    parser.add_argument("--optimizer", choices=["adamw", "poprisk-adamw"], default="adamw")
    parser.add_argument("--matrix-optimizer",
                        choices=["muon", "pmuon", "adamh", "poprisk-adamh", "kfac-adamh", "kfac-muon", "kfac"],
                        default="muon")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--stop-after-step", type=int, default=-1,
                        help="Stop after this step while preserving --train-steps for schedules")
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--dense-val-start", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--reference-batch-tokens", type=int, default=0,
                        help="If positive, multiply gradients by reference_batch_tokens / batch_tokens")
    parser.add_argument("--microbatch-seqs", type=int, default=4)
    parser.add_argument("--val-tokens", type=int, default=262144)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--model-dim", type=int, default=384)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--download-chunks", type=int, default=0,
                        help="Download this many train shards plus val into --data-dir before training")
    parser.add_argument("--muon-lr", type=float, default=0.025)
    parser.add_argument("--muon-wd", type=float, default=0.0125)
    parser.add_argument("--muon-ns-iters", type=int, default=12)
    parser.add_argument("--pmuon-beta", type=float, default=0.95)
    parser.add_argument("--pmuon-gamma", type=float, default=0.3)
    parser.add_argument("--pmuon-eps", type=float, default=1e-8)
    parser.add_argument("--adamh-lr", type=float, default=0.018)
    parser.add_argument("--adamh-warmup-steps", type=int, default=-1)
    parser.add_argument("--adamh-cooldown-frac", type=float, default=1.0)
    parser.add_argument("--adamh-aux-cooldown-frac", type=float, default=0.4)
    parser.add_argument("--embed-lr", type=float, default=0.3)
    parser.add_argument("--proj-lr", type=float, default=1/320)
    parser.add_argument("--scalar-lr", type=float, default=0.01)
    parser.add_argument("--cooldown-frac", type=float, default=0.7)
    parser.add_argument("--kfac-lr", type=float, default=0.001)
    parser.add_argument("--kfac-wd", type=float, default=0.025)
    parser.add_argument("--kfac-mu", type=float, default=0.95)
    parser.add_argument("--kfac-factor-rho", type=float, default=0.95)
    parser.add_argument("--kfac-damping", type=float, default=0.03)
    parser.add_argument("--kfac-refresh-steps", type=int, default=1)
    parser.add_argument("--pop-alpha", type=float, default=1.0)
    parser.add_argument("--pop-lambda", type=float, default=1.0)
    parser.add_argument("--pop-lambda-mode", choices=["fixed", "target-median-q", "cosine-decay"], default="fixed")
    parser.add_argument("--pop-lambda-final", type=float, default=0.0)
    parser.add_argument("--pop-lambda-decay-start-frac", type=float, default=0.5)
    parser.add_argument("--pop-target-q", type=float, default=0.5)
    parser.add_argument("--pop-rho", type=float, default=0.95)
    parser.add_argument("--pop-warmup-steps", type=int, default=20)
    parser.add_argument("--pop-gate", choices=["soft", "hard", "snr", "snr-wiener", "snr-var"], default="soft")
    return apply_proxy_preset(parser.parse_args(), sys.argv[1:])

args = parse_args()
if not 0 <= args.pop_lambda_decay_start_frac <= 1:
    raise ValueError("--pop-lambda-decay-start-frac must be between 0 and 1")
if args.stop_after_step < -1:
    raise ValueError("--stop-after-step must be -1 or non-negative")
if args.stop_after_step > args.train_steps:
    raise ValueError("--stop-after-step must be <= --train-steps")
pop_lambda_decay_start_step = int(args.train_steps * args.pop_lambda_decay_start_frac)
adamh_warmup_steps = int(args.train_steps * 0.05) if args.adamh_warmup_steps < 0 else args.adamh_warmup_steps
stop_after_step = args.train_steps if args.stop_after_step < 0 else args.stop_after_step
data_dir = args.data_dir or default_data_dir()
if args.download_chunks:
    download_fineweb10b(data_dir, args.download_chunks)

if "RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "This proxy is intended for CUDA GPUs"
assert args.batch_tokens % args.seq_len == 0
assert args.val_tokens % args.seq_len == 0
if args.reference_batch_tokens < 0:
    raise ValueError("--reference-batch-tokens must be non-negative")
reference_batch_tokens = args.reference_batch_tokens or args.batch_tokens
grad_scale = reference_batch_tokens / args.batch_tokens

torch.manual_seed(args.seed + get_rank())
torch.cuda.manual_seed(args.seed + get_rank())

if is_master():
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/local_proxy_{args.optimizer}_{args.matrix_optimizer}_{uuid.uuid4()}.txt"
    print(logfile)

def print0(s, console=False, log=True):
    if is_master():
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

config = vars(args).copy()
config["data_dir"] = str(data_dir)
config["world_size"] = get_world_size()
config["device_name"] = torch.cuda.get_device_name(device)
config["pop_lambda_decay_start_step"] = pop_lambda_decay_start_step
config["adamh_warmup_steps_resolved"] = adamh_warmup_steps
config["stop_after_step_resolved"] = stop_after_step
config["reference_batch_tokens_resolved"] = reference_batch_tokens
config["grad_scale"] = grad_scale

print0(code)
print0("=" * 100)
print0(json.dumps(config, indent=2, sort_keys=True))
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0("=" * 100)

train_loader = data_generator(data_dir, "train", args.batch_tokens, args.seq_len, device)
val_loader = data_generator(data_dir, "val", args.val_tokens, args.seq_len, device)
val_inputs, val_targets = next(val_loader)

model = GPT(
    vocab_size=50304,
    num_layers=args.num_layers,
    model_dim=args.model_dim,
    head_dim=args.head_dim,
    seq_len=args.seq_len,
).to(device)
if args.compile:
    model.compile(dynamic=False)

if args.matrix_optimizer in ("adamh", "poprisk-adamh", "kfac-adamh"):
    for name, p in model.named_parameters():
        if name.endswith(".attn.proj.weight"):
            p.data.mul_(1.25)
        elif name.endswith(".mlp.proj.weight"):
            p.data.mul_(3.0)
        elif name.endswith(".mlp.fc.weight"):
            p.data.mul_(1.5)
        elif name == "proj.weight":
            p.data.zero_()
        elif "proj" in name:
            p.data.zero_()
else:
    for name, p in model.named_parameters():
        if "proj" in name:
            p.data.zero_()

adam_groups = [
    dict(params=[model.embed.weight], lr=args.embed_lr),
    dict(params=[model.proj.weight], lr=args.proj_lr),
    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=args.scalar_lr),
]
if args.optimizer == "poprisk-adamw":
    optimizer1 = PopRiskAdamW(
        adam_groups,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0,
        rho=args.pop_rho,
        alpha=args.pop_alpha,
        lambda_pop=args.pop_lambda,
        lambda_mode=args.pop_lambda_mode,
        lambda_final=args.pop_lambda_final,
        lambda_decay_start=pop_lambda_decay_start_step,
        lambda_decay_end=args.train_steps,
        target_q=args.pop_target_q,
        warmup_steps=args.pop_warmup_steps,
        gate=args.pop_gate,
    )
else:
    optimizer1 = torch.optim.AdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
matrix_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
kfac_store = None
if args.matrix_optimizer in ("kfac-adamh", "kfac-muon", "kfac"):
    kfac_store = KFACFactorStore(
        kfac_linear_modules(model),
        rho=args.kfac_factor_rho,
        damping=args.kfac_damping,
        refresh_steps=args.kfac_refresh_steps,
    )
if args.matrix_optimizer == "muon":
    optimizer2 = Muon(
        matrix_params,
        lr=args.muon_lr,
        weight_decay=args.muon_wd,
        ns_iters=args.muon_ns_iters,
    )
elif args.matrix_optimizer == "pmuon":
    optimizer2 = PMuon(
        matrix_params,
        lr=args.muon_lr,
        weight_decay=args.muon_wd,
        ns_iters=args.muon_ns_iters,
        beta=args.pmuon_beta,
        gamma=args.pmuon_gamma,
        eps=args.pmuon_eps,
    )
elif args.matrix_optimizer in ("adamh", "poprisk-adamh", "kfac-adamh"):
    optimizer2 = AdamH(
        matrix_params,
        lr=args.adamh_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        poprisk=args.matrix_optimizer == "poprisk-adamh",
        rho=args.pop_rho,
        alpha=args.pop_alpha,
        lambda_pop=args.pop_lambda,
        lambda_mode=args.pop_lambda_mode,
        lambda_final=args.pop_lambda_final,
        lambda_decay_start=pop_lambda_decay_start_step,
        lambda_decay_end=args.train_steps,
        target_q=args.pop_target_q,
        warmup_steps=args.pop_warmup_steps,
        gate=args.pop_gate,
        kfac_store=kfac_store,
    )
elif args.matrix_optimizer == "kfac-muon":
    optimizer2 = KFACMuon(
        matrix_params,
        kfac_store,
        lr=args.muon_lr,
        weight_decay=args.muon_wd,
        ns_iters=args.muon_ns_iters,
        mu=args.kfac_mu,
    )
else:
    optimizer2 = KFAC(
        matrix_params,
        kfac_store,
        lr=args.kfac_lr,
        weight_decay=args.kfac_wd,
        mu=args.kfac_mu,
    )
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
        group["schedule_type"] = "default"
if args.matrix_optimizer in ("adamh", "poprisk-adamh", "kfac-adamh"):
    for group in optimizer1.param_groups:
        group["schedule_type"] = "adamh_aux"
    for group in optimizer2.param_groups:
        group["schedule_type"] = "adamh_matrix"

if dist_ready():
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

def set_hparams(step):
    progress = step / args.train_steps
    assert 0 <= progress < 1
    for opt in optimizers:
        for group in opt.param_groups:
            if group["schedule_type"] == "adamh_matrix":
                if step < adamh_warmup_steps:
                    eta = step / max(adamh_warmup_steps, 1)
                elif progress < 1 - args.adamh_cooldown_frac:
                    eta = 1.0
                else:
                    eta = (1 - progress) / args.adamh_cooldown_frac
            elif group["schedule_type"] == "adamh_aux":
                if progress < 1 - args.adamh_aux_cooldown_frac:
                    eta = 1.0
                else:
                    eta = (1 - progress) / args.adamh_aux_cooldown_frac
            elif progress < 1 - args.cooldown_frac:
                eta = 1.0
            else:
                eta = (1 - progress) / args.cooldown_frac
            group["lr"] = group["initial_lr"] * eta

def format_gate_stats():
    items = [
        ("aux", getattr(optimizer1, "last_gate_stats", {})),
        ("matrix", getattr(optimizer2, "last_gate_stats", {})),
    ]
    items = [(name, stats) for name, stats in items if stats]
    if not items:
        return ""
    parts = []
    use_prefix = len(items) > 1
    for name, stats in items:
        prefix = f"{name}_" if use_prefix else ""
        parts.append(
            f" {prefix}q_mean:{stats.get('q_mean', 0):.4f}"
            + f" {prefix}q_lt_0.01:{stats.get('q_lt_0.01', 0):.4f}"
            + f" {prefix}q_gt_0.99:{stats.get('q_gt_0.99', 0):.4f}"
            + f" {prefix}lambda_pop:{stats.get('lambda_pop', 0):.6g}"
        )
    return "".join(parts)

def validate(step, training_time):
    if dist_ready():
        dist.barrier()
    model.eval()
    val_loss = torch.zeros((), device=device)
    with torch.no_grad():
        assert len(val_inputs) % args.microbatch_seqs == 0
        for i in range(len(val_inputs) // args.microbatch_seqs):
            sl = slice(i * args.microbatch_seqs, (i + 1) * args.microbatch_seqs)
            val_loss += model(val_inputs[sl], val_targets[sl])
    maybe_all_reduce(val_loss)
    val_loss /= args.val_tokens
    print0(
        f"step:{step}/{args.train_steps} val_loss:{val_loss.item():.8f} "
        + f"train_time:{training_time:.3f}s step_avg:{1000 * training_time / max(step, 1):.2f}ms",
        console=True,
    )
    model.train()

training_time = 0.0
if dist_ready():
    dist.barrier()
t0 = time.perf_counter()
for step in range(stop_after_step + 1):
    should_validate = (
        step == stop_after_step
        or step % args.val_interval == 0
        or (args.dense_val_start >= 0 and step >= args.dense_val_start)
    )
    if should_validate:
        training_time += time.perf_counter() - t0
        validate(step, training_time)
        if dist_ready():
            dist.barrier()
        t0 = time.perf_counter()
    if step == stop_after_step:
        break

    inputs, targets = next(train_loader)
    assert len(inputs) % args.microbatch_seqs == 0
    loss_sum = torch.zeros((), device=device)
    for i in range(len(inputs) // args.microbatch_seqs):
        sl = slice(i * args.microbatch_seqs, (i + 1) * args.microbatch_seqs)
        loss = model(inputs[sl], targets[sl])
        loss_sum += loss.detach()
        loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        maybe_all_reduce(p.grad)
        if grad_scale != 1:
            p.grad.mul_(grad_scale)
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    approx_training_time = training_time + (time.perf_counter() - t0)
    if step + 1 == 1 or (step + 1) % args.log_interval == 0:
        maybe_all_reduce(loss_sum)
        train_loss = loss_sum.item() / args.batch_tokens
        gate_text = format_gate_stats()
        print0(
            f"step:{step + 1}/{args.train_steps} train_loss:{train_loss:.8f} "
            + f"train_time:{approx_training_time:.3f}s "
            + f"step_avg:{1000 * approx_training_time / (step + 1):.2f}ms"
            + gate_text,
            console=True,
        )

print0(f"logfile: {logfile}", console=True, log=False)

if dist_ready():
    dist.destroy_process_group()
