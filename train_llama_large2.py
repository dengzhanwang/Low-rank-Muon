import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Muon optimizer (modified)
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

def zeropower_via_newtonschulz5(G, use_Newton=True, k=400) -> Tensor:
    """
    Modified Newton-Schulz iteration with dimension checks and stability improvements
    """
    if G.dim() < 2:
        return G  # Skip non-2D tensors
    
    orig_shape = G.shape
    orig_dtype = G.dtype
    X = G.float()  # Convert to float32 for stability
    if G.size(-2) <= 500 or G.size(-1) <= 500:
        use_Newton = True  # use Newton-Schulz for 1024x1024 matrices, as it is faster than QR decomposition
    else:
        use_Newton = False
    # Handle transpose case for tall matrices
    transpose = False
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
        transpose = True
    a, b, c = torch.tensor(3.4445), torch.tensor(-4.7750), torch.tensor(2.0315)
    if use_Newton:
        # Constants for quintic iteration
        
        
        # Normalize to ensure spectral norm <= 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        
        # Fixed 5 iterations
        for _ in range(5):
            A = X @ X.transpose(-2, -1)
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    else:
        n = G.size(-1)  # 取最后一个维度的大小
        col_indices = torch.randperm(n, device=X.device)[:k]
        
        # 使用index_select选择列，保持批量维度
        Y = torch.index_select(X, dim=-1, index=col_indices)
        
        # 对最后两个维度做QR分解
        Q, _ = torch.linalg.qr(Y.float())  # 转换为 float32
        Q = Q.to(X.dtype)  # 转回 bfloat16

        # 使用matmul进行批量矩阵乘法，正确转置Q的最后两个维度
        B = Q.mT @ X
        # Newton-Schulz近似
        X = B / (B.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        # print(f"Using Newton-Schulz for {X.size(-2)}x{X.size(-1)} matrix with k={k}")
        for _ in range(5):
            A = X @ X.mT
            B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X
        X = Q @ X  # m×k
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Modified Muon optimizer with dimension checks and stability improvements
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                       backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # Generate updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            
            for i, p in enumerate(group['params']):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue
                        
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    
                    # Only apply orthogonalization to 2D parameters
                    if g.dim() >= 2:
                        try:
                            g = zeropower_backend(g)
                            g *= max(g.size(0), g.size(1))**0.5  # Scale update
                        except Exception as e:
                            print(f"Error in zeropower_backend: {e}")
                            raise
                    
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # Sync updates across devices
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # Apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# Llama model components (unchanged)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),)) * self.weight

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.o_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)
        
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.o_proj(y)
        return y

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.attention_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)

    def forward(self, x):
        h = x + self.attn(self.attention_norm(x))
        out = h + self.mlp(self.ffn_norm(h))
        return out

@dataclass
class LlamaConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

    def forward(self, idx, targets=None, return_logits=True):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        if targets is not None:
            logits = self.output(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(x[:, [-1], :])
            logits = logits.float()
            loss = None
            
        return logits if return_logits else None, loss

# -----------------------------------------------------------------------------
# Data loader and main training code (unchanged)
def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    return header[2]

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520
        assert header[1] == 1
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0
        
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

@dataclass
class Hyperparameters:
    input_bin: str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin: str = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size: int = 8 * 60
    device_batch_size: int = 8
    sequence_length: int = 1024
    num_iterations: int = 2 * 10172
    learning_rate: float = 0.0036 / 2
    warmup_iters: int = 500
    warmdown_iters: int = 2 * 2906
    weight_decay: float = 0
    val_loss_every: int = 125
    val_tokens: int = 10403840
    save_every: int = 0

args = Hyperparameters()

# Setup DDP
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)
run_id = int(os.environ.get("RUN_ID", 0))
if master_process:
    run_id_full = f"large_{run_id:03d}_{uuid.uuid4()}"
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id_full}.txt"
    print(logfile)

B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# Load data
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# Initialize Llama model with same config as original GPT
num_vocab = 50304
model = Llama(LlamaConfig(vocab_size=num_vocab, n_layer=32, n_head=16, n_embd=2048))
if master_process:
    print(sum(p.numel() for p in model.parameters()))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True

# Disable torch.compile for stability
# model = torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# Optimizers
optimizer1 = torch.optim.AdamW(raw_model.output.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon([p for layer in raw_model.layers for p in layer.parameters()], lr=0.025, momentum=0.95, use_Newton=args.use_Newton, k=args.k)
# optimizer2 = torch.optim.SGD([p for layer in raw_model.layers for p in layer.parameters()], #
#                   lr=args.learning_rate, momentum=0.95,
#                   weight_decay=args.weight_decay, nesterov=True, fused=True)
optimizers = [optimizer1, optimizer2]

def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        return (args.num_iterations - it) / args.warmdown_iters

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    model.train()
    for i in range(1, train_accumulation_steps+1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    if step % 100 == 2 and master_process:
        model_to_plot = model.module if hasattr(model, 'module') else model
        print(f"\nStep {step} gradient check:")
        data_dir = f"logs/{run_id_full}/grad_svd_data/"
        os.makedirs(data_dir, exist_ok=True)
        
        n_layers = len(model_to_plot.layers)
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        fig.suptitle(f'Layer-wise Gradient Singular Values (Step {step})', fontsize=16)
        
        if n_rows == 1:
            axs = axs.reshape(1, -1)
        if n_cols == 1:
            axs = axs.reshape(-1, 1)
        
        for layer_idx, block in enumerate(model_to_plot.layers):
            row = layer_idx // n_cols
            col = layer_idx % n_cols
            ax = axs[row, col]
            layer_data = {}
            
            for name, param in [('Q', block.attn.q_proj.weight),
                                ('K', block.attn.k_proj.weight),
                                ('V', block.attn.v_proj.weight)]:
                if param in optimizer2.state and 'momentum_buffer' in optimizer2.state[param]:
                        momentum = optimizer2.state[param]['momentum_buffer']
                        _, s, _ = torch.svd(momentum.float())
                        s = s.cpu().numpy()[:200]  # 只取前200个奇异值
                        ax.plot(s, label=name, alpha=0.7)
                        layer_data[name] = s  # 保存数
                elif param.grad is not None:
                        _, s, _ = torch.svd(param.grad.float())
                        s = s.cpu().numpy()[:200]  # 只取前100个奇异值
                        ax.plot(s, label=name, alpha=0.7)
                        layer_data[name] = s  # 保存数据      
            
            if layer_data:
                np.savez(f"{data_dir}layer_{layer_idx}_step_{step:04d}.npz", **layer_data)
                ax.set_title(f'Layer {layer_idx}')
                ax.set_yscale('log')
                ax.set_xlabel('Singular Value Index')
                ax.set_ylabel('Log Scale Magnitude')
                ax.legend()
                ax.grid(True, which="both", ls="--")
            else:
                ax.axis('off')
                ax.set_title(f'Layer {layer_idx} (No grad)')
        
        for layer_idx in range(n_layers, n_rows * n_cols):
            row = layer_idx // n_cols
            col = layer_idx % n_cols
            axs[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(f"logs/{run_id_full}/grad_svd/", exist_ok=True)
        plt.savefig(f"logs/{run_id_full}/grad_svd/step_{step:04d}.png")
        plt.close(fig)

    for p in model.parameters():
        p.grad /= train_accumulation_steps

    for opt, sched in zip(optimizers, schedulers):
        start_time = time.perf_counter() # 使用高精度计时器
        opt.step()
        end_time = time.perf_counter()
        print(f"Optimizer step time: {(end_time - start_time)*1000:.2f} ms")
        sched.step()

    model.zero_grad(set_to_none=True)

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")