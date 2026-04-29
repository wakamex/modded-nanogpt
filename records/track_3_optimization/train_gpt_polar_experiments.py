"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import bisect
import json
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#          Experiment Config           #
########################################

POLAR_MODE = os.environ.get("TRACK3_POLAR_MODE", "baseline")
POLAR_COEFFS_JSON = os.environ.get("TRACK3_POLAR_COEFFS_JSON", "")
DENSE_VAL_START = int(os.environ.get("TRACK3_DENSE_VAL_START", "3400"))
TRAIN_STEPS_OVERRIDE = os.environ.get("TRACK3_TRAIN_STEPS")
STOP_AFTER_OVERRIDE = os.environ.get("TRACK3_STOP_AFTER")
SPECTRUM_DIR = os.environ.get("TRACK3_SPECTRUM_DIR", "spectrum_logs")
SPECTRUM_STEPS_ENV = os.environ.get(
    "TRACK3_SPECTRUM_STEPS",
    "1,2,4,8,16,32,64,125,250,500,750,1000,1500,2000,2500,3000,3400",
)


def parse_int_set(raw: str) -> set[int]:
    return {int(x) for x in raw.replace(" ", "").split(",") if x}


def load_coeff_payload(path: str) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


COEFF_PAYLOAD = load_coeff_payload(POLAR_COEFFS_JSON)
FIXED_POLAR_COEFFS = COEFF_PAYLOAD.get("coeffs")
ADAPTIVE_STEP_COEFFS = sorted(
    (int(k), v) for k, v in COEFF_PAYLOAD.get("step_coeffs", {}).items()
)
ADAPTIVE_STEPS = [x[0] for x in ADAPTIVE_STEP_COEFFS]
SPECTRUM_STEPS = parse_int_set(SPECTRUM_STEPS_ENV)
SPECTRUM_ENABLED = POLAR_MODE == "spectrum" or os.environ.get("TRACK3_SPECTRUM", "0") == "1"
CURRENT_MUON_STEP = 0
MUON_PARAM_NAMES: dict[int, str] = {}


def get_polar_coeffs(step: int) -> list[tuple[float, float, float]]:
    if POLAR_MODE == "baseline" or POLAR_MODE == "spectrum":
        return [(2.0, -1.5, 0.5)] * 12
    if POLAR_MODE == "fixed":
        assert FIXED_POLAR_COEFFS, "TRACK3_POLAR_COEFFS_JSON must contain coeffs for fixed mode"
        return [tuple(map(float, row)) for row in FIXED_POLAR_COEFFS]
    if POLAR_MODE == "adaptive":
        assert ADAPTIVE_STEP_COEFFS, "TRACK3_POLAR_COEFFS_JSON must contain step_coeffs for adaptive mode"
        i = bisect.bisect_right(ADAPTIVE_STEPS, step) - 1
        if i < 0:
            i = 0
        return [tuple(map(float, row)) for row in ADAPTIVE_STEP_COEFFS[i][1]]
    raise ValueError(f"unknown TRACK3_POLAR_MODE={POLAR_MODE!r}")


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
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
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
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
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the polynomial polar iterations, not optimizing for wallclock speed
    for a, b, c in get_polar_coeffs(CURRENT_MUON_STEP):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def maybe_log_spectrum(step: int, name: str, matrix: Tensor):
    if not SPECTRUM_ENABLED or step not in SPECTRUM_STEPS:
        return
    X = matrix.detach()
    if X.size(-2) > X.size(-1):
        X = X.mT
    fro = X.float().norm()
    X = X.float() / (fro + 1e-7)
    singular_values = torch.linalg.svdvals(X).float().cpu()
    quantiles = torch.tensor([0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0])
    values = torch.quantile(singular_values, quantiles).tolist()
    row = {
        "step": step,
        "rank": dist.get_rank(),
        "name": name,
        "shape": list(matrix.shape),
        "prepared_shape": list(X.shape),
        "fro_norm": float(fro.cpu()),
        "num_singular_values": int(singular_values.numel()),
        "q000": values[0],
        "q001": values[1],
        "q010": values[2],
        "q050": values[3],
        "q100": values[4],
        "q500": values[5],
        "q900": values[6],
        "q990": values[7],
        "q1000": values[8],
    }
    Path(SPECTRUM_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(SPECTRUM_DIR) / f"spectrum_rank{dist.get_rank()}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def muon_update(update):
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update


if POLAR_MODE != "adaptive":
    muon_update = torch.compile(muon_update)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self):
        global CURRENT_MUON_STEP
        self.step_count += 1
        CURRENT_MUON_STEP = self.step_count
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    state["momentum"].lerp_(p.grad, 1 - group["mu"])
                    update_input = p.grad.lerp(state["momentum"], group["mu"])
                    maybe_log_spectrum(self.step_count, MUON_PARAM_NAMES.get(id(p), ""), update_input)
                    update = muon_update(update_input)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running on device_name={torch.cuda.get_device_name(device)} with world_size={dist.get_world_size()}")
print0(f"TRACK3_POLAR_MODE={POLAR_MODE}")
print0(f"TRACK3_POLAR_COEFFS_JSON={POLAR_COEFFS_JSON or '<none>'}")
print0(f"TRACK3_TRAIN_STEPS={TRAIN_STEPS_OVERRIDE or '3500'}")
print0(f"TRACK3_STOP_AFTER={STOP_AFTER_OVERRIDE or '<train_steps>'}")
print0(f"TRACK3_DENSE_VAL_START={DENSE_VAL_START}")
print0(f"TRACK3_SPECTRUM_ENABLED={SPECTRUM_ENABLED}")
if SPECTRUM_ENABLED:
    print0(f"TRACK3_SPECTRUM_STEPS={sorted(SPECTRUM_STEPS)}")
    print0(f"TRACK3_SPECTRUM_DIR={SPECTRUM_DIR}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = int(TRAIN_STEPS_OVERRIDE) if TRAIN_STEPS_OVERRIDE else 3500
stop_after = int(STOP_AFTER_OVERRIDE) if STOP_AFTER_OVERRIDE else train_steps
assert 0 <= stop_after <= train_steps

# initialize model parameters
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
muon_named_params = [(name, p) for name, p in model.blocks.named_parameters() if p.ndim >= 2]
MUON_PARAM_NAMES = {id(p): name for name, p in muon_named_params}
optimizer2 = Muon([p for _, p in muon_named_params], lr=0.025, weight_decay=0.0125)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def set_hparams(step, cooldown_frac=0.7):
    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < 1 - cooldown_frac:
        eta = 1.0
    else:
        eta = (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


########################################
#        Training and Validation       #
########################################

for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
dist.barrier()
t0 = time.perf_counter()
for step in range(stop_after + 1):

    # --------------- VALIDATION SECTION -----------------
    if step == stop_after or step == train_steps or step % 125 == 0 or step >= DENSE_VAL_START:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == stop_after or step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    for i in range(len(inputs) // mbs):
        model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    # set optimization hyperparameters and take a step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
