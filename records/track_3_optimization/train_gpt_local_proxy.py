"""
Local Track 3 proxy runner.

This keeps the Track 3 model family, FineWeb tokenization, Muon matrix
optimizer, and Adam-style non-matrix optimizer split, but scales the model and
batch down so a single 24GB GPU can run short optimizer ablations.
"""

import argparse
import json
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
            target_q=target_q,
            warmup_steps=warmup_steps,
            gate=gate,
        )
        super().__init__(params, defaults)
        self.last_gate_stats = {}

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
        lambda_for_stats = adaptive_lambda.item() if adaptive_lambda is not None else self.param_groups[0]["lambda_pop"]

        gate_sum = 0.0
        gate_numel = 0
        gate_low = 0
        gate_high = 0
        group_stats = []
        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            alpha = group["alpha"]
            lambda_pop = group["lambda_pop"]
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
                lambda_effective = adaptive_lambda if adaptive_lambda is not None else lambda_pop

                if step <= warmup_steps:
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
                gate_sum += q_sum
                gate_numel += q_numel
                gate_low += low
                gate_high += high
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
            "lambda_pop": float(lambda_for_stats),
            "groups": group_stats,
        }


########################################
#                Setup                 #
########################################

def default_data_dir():
    nvme = Path("/nvme2/modded-nanogpt-data/fineweb10B")
    return nvme if nvme.exists() else Path("data/fineweb10B")

def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU/local proxy for Track 3 optimizer tests")
    parser.add_argument("--optimizer", choices=["adamw", "poprisk-adamw"], default="adamw")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--dense-val-start", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-tokens", type=int, default=32768)
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
    parser.add_argument("--embed-lr", type=float, default=0.3)
    parser.add_argument("--proj-lr", type=float, default=1/320)
    parser.add_argument("--scalar-lr", type=float, default=0.01)
    parser.add_argument("--cooldown-frac", type=float, default=0.7)
    parser.add_argument("--pop-alpha", type=float, default=1.0)
    parser.add_argument("--pop-lambda", type=float, default=1.0)
    parser.add_argument("--pop-lambda-mode", choices=["fixed", "target-median-q"], default="fixed")
    parser.add_argument("--pop-target-q", type=float, default=0.5)
    parser.add_argument("--pop-rho", type=float, default=0.95)
    parser.add_argument("--pop-warmup-steps", type=int, default=20)
    parser.add_argument("--pop-gate", choices=["soft", "hard", "snr"], default="soft")
    return parser.parse_args()

args = parse_args()
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

torch.manual_seed(args.seed + get_rank())
torch.cuda.manual_seed(args.seed + get_rank())

if is_master():
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/local_proxy_{args.optimizer}_{uuid.uuid4()}.txt"
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
        target_q=args.pop_target_q,
        warmup_steps=args.pop_warmup_steps,
        gate=args.pop_gate,
    )
else:
    optimizer1 = torch.optim.AdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
optimizer2 = Muon(
    [p for p in model.blocks.parameters() if p.ndim >= 2],
    lr=args.muon_lr,
    weight_decay=args.muon_wd,
    ns_iters=args.muon_ns_iters,
)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

if dist_ready():
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

def set_hparams(step):
    progress = step / args.train_steps
    assert 0 <= progress < 1
    if progress < 1 - args.cooldown_frac:
        eta = 1.0
    else:
        eta = (1 - progress) / args.cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta

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
for step in range(args.train_steps + 1):
    should_validate = (
        step == args.train_steps
        or step % args.val_interval == 0
        or (args.dense_val_start >= 0 and step >= args.dense_val_start)
    )
    if should_validate:
        training_time += time.perf_counter() - t0
        validate(step, training_time)
        if dist_ready():
            dist.barrier()
        t0 = time.perf_counter()
    if step == args.train_steps:
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
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    approx_training_time = training_time + (time.perf_counter() - t0)
    if step + 1 == 1 or (step + 1) % args.log_interval == 0:
        maybe_all_reduce(loss_sum)
        train_loss = loss_sum.item() / args.batch_tokens
        gate_text = ""
        if isinstance(optimizer1, PopRiskAdamW):
            s = optimizer1.last_gate_stats
            gate_text = (
                f" q_mean:{s.get('q_mean', 0):.4f}"
                + f" q_lt_0.01:{s.get('q_lt_0.01', 0):.4f}"
                + f" q_gt_0.99:{s.get('q_gt_0.99', 0):.4f}"
                + f" lambda_pop:{s.get('lambda_pop', 0):.6g}"
            )
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
