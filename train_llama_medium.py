import os
import sys
try:
    with open(__file__, 'r', encoding='utf-8') as f:
        code = f.read()  # 读取当前文件的代码
except UnicodeDecodeError:
    try:
        with open(__file__, 'r', encoding='gbk') as f:
            code = f.read()
    except Exception as e:
        print(f"无法读取文件: {e}")
        code = ""
import uuid
import time
import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 避免使用GUI后端
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass, asdict, fields
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from dataclasses import asdict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
torch._inductor.config.coordinate_descent_tuning = True # we allow this flag for medium track
torch._dynamo.config.compiled_autograd = True

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_newtonschulz5(G: Tensor, use_Newton = 1, k = 200) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    if G.size(-2) <= 500 or G.size(-1) <= 500:
        use_Newton = 1  # use Newton-Schulz for 1024x1024 matrices, as it is faster than QR decomposition
    a, b, c = (3.4445, -4.7750, 2.0315)
    # start_time = time.time() 
    if use_Newton:
    # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        
        # Perform the NS iterations
        # print(f"Using Newton-Schulz for {X.size(-2)}x{X.size(-1)} matrix with k={k}")
        for _ in range(5):  # 10 iterations
            A = X @ X.mT
            B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X

        C = X @ X.mT  # m×k
        if G.size(-2) > G.size(-1):
            X = X.mT
        
        return X
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

@torch.compile
def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor,use_Newton: bool = True, k: int = 200):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)

    
    # v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad,use_Newton=use_Newton, k=k)
    # print(f"Using Newton-Schulz for use_Newton={use_Newton}, k={k}, acc_bf16_view_u16.size()={acc_bf16_view_u16.size()}, mantissa.size()={mantissa.size()}")
    # if dist.get_rank() == 0:
    #     print(f"Time: {(time.time()-start)*1000:.2f} ms")

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1,use_Newton=1, k=200):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.use_Newton = use_Newton
        self.k = k
        super().__init__(params, defaults)
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    
                    update(
                        p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                        p.grad, momentum,
                        eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                        eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                        use_Newton=self.use_Newton, k=self.k
                    )
                futures.append(dist.all_gather(params_pad[base_i:base_i + self.world_size], params_pad[base_i + self.rank], async_op=True).get_future())
        torch.futures.collect_all(futures).wait()

# -----------------------------------------------------------------------------
# Llama 350M model implementation

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class LlamaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        
        # Llama uses separate Q, K, V projections
        self.q_proj = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.k_proj = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.v_proj = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.o_proj = nn.Parameter(torch.zeros(dim, hdim).bfloat16())
        
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 1.0 / (head_dim ** 0.5)  # Standard attention scaling

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask, lambdas: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        
        # Project inputs to Q, K, V
        q = F.linear(x, self.q_proj).view(B, T, self.num_heads, self.head_dim)
        k = F.linear(x, self.k_proj).view(B, T, self.num_heads, self.head_dim)
        v = F.linear(x, self.v_proj).view(B, T, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings to Q and K
        q, k = self.rotary(q), self.rotary(k)
        
        # Apply value embedding if provided
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            v = lambdas[0] * v
            
        # Compute attention
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                         block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        
        # Project output
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.o_proj)
        return y

class LlamaMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = 4 * dim * 2 // 3  # Llama uses 4/3 * dim * 2 for hidden dim
        hidden_dim = int(2 * hidden_dim / 3)  # Make sure it's divisible by 2
        
        self.gate_proj = nn.Parameter(init_linear(torch.empty(hidden_dim, dim)).bfloat16())
        self.up_proj = nn.Parameter(init_linear(torch.empty(hidden_dim, dim)).bfloat16())
        self.down_proj = nn.Parameter(torch.zeros(dim, hidden_dim).bfloat16())
        
        self.gate_proj.wd_mul = 2.0
        self.up_proj.wd_mul = 2.0
        self.down_proj.wd_mul = 2.0

    def forward(self, x: Tensor):
        gate = F.linear(x, self.gate_proj)
        up = F.linear(x, self.up_proj)
        x = F.silu(gate) * up  # SwiGLU activation
        x = F.linear(x, self.down_proj)
        return x

class LlamaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = LlamaAttention(dim, num_heads, max_seq_len)
        self.mlp = LlamaMLP(dim)
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask, lambdas: Tensor, sa_lambdas: Tensor):
        residual = x
        x = self.input_layernorm(x)
        x = self.attn(x, ve, block_mask, sa_lambdas)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class Llama(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([LlamaBlock(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head_w = nn.Parameter(torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim))
        
        # Initialize scalar parameters
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)
    
    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        loss = checkpoint(self._forward, input_seq, target_seq, sliding_window_num_blocks, use_reentrant=False)
        return loss
    
    def _forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = []
        for i in range(len(self.blocks)):
            if i % 4 == 0 or i == len(self.blocks) - 1:
                block_masks.append(long_bm)
            else:
                block_masks.append(short_bm)
        assert len(block_masks) == len(self.blocks)

        x = x0 = self.embed(input_seq)[None]

        skip_connections = []
        skip_map = {
            6: 3,
            7: 0
        }
        skip_weights = self.scalars[:len(self.blocks)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        
        for i in range(len(self.blocks)):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
            skip_connections.append(x)

        x = self.norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1), self.lm_head_w.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        loss = 0
        for i in range(4):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(4)[i], self.lm_head_w.bfloat16()).float()
            loss += F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq.chunk(4)[i]) / 4
        return loss

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets
########################################
#    Construct model and optimizer     #
########################################

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型训练参数设置')
    
    # 数据参数
    parser.add_argument('--train_files', type=str, default="data/finewebedu10B/finewebedu_train_*.bin",
                       help='训练数据文件路径模式')
    parser.add_argument('--val_files', type=str, default="data/finewebedu10B/finewebedu_val_*.bin",
                       help='验证数据文件路径模式')
    parser.add_argument('--val_tokens', type=int, default=10485760,
                       help='验证数据token数量')
    parser.add_argument('--train_seq_len', type=int, default=48 * 1024,
                       help='训练序列长度')
    parser.add_argument('--val_seq_len', type=int, default=4 * 64 * 1024,
                       help='验证序列长度')
    parser.add_argument('--optimizer', type=str, default='Muon',
                   choices=['Muon', 'Adam', 'SGD', 'RMSprop'],
                   help='选择优化器 (默认: Muon)')
    
    # 优化参数
    parser.add_argument('--num_iterations', type=int, default=5960,
                       help='训练迭代次数')
    parser.add_argument('--cooldown_frac', type=float, default=0.7,
                       help='学习率冷却阶段占总训练的比例')
    
    # 架构参数
    parser.add_argument('--vocab_size', type=int, default=50257,
                       help='词汇表大小')
    
    # 评估和日志参数
    parser.add_argument('--val_loss_every', type=int, default=10,
                       help='每隔多少步评估验证损失(0表示仅在结束时评估)')
    parser.add_argument('--plot_every', type=int, default=500,
                       help='每隔多少步绘制图表')
    parser.add_argument('--save_checkpoint', type=bool, default=False,
                       help='是否保存检查点')
    
    # 特殊参数
    parser.add_argument('--use_Newton', type=lambda x: x.lower() == 'true', default=False,
                   help='是否使用Newton方法（默认True，支持--use_Newton=False）')

    parser.add_argument('--k', type=int, default=200,
                       help='k值参数')
    
    return parser.parse_args()

args = parse_args()

run_id = int(os.environ.get("RUN_ID", 0))
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0)

# begin logging
if master_process:
    run_id_full = f"tiny_{run_id:03d}_{uuid.uuid4()}"
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id_full}.txt"
    print(logfile)
    
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# Construct Llama 350M model
model: nn.Module = Llama(
    vocab_size=args.vocab_size,
    num_layers=24,       # 24 layers for 350M model
    num_heads=16,        # 16 attention heads
    model_dim=1024,      # 1024 hidden dimension
    max_seq_len=max(args.train_seq_len, args.val_seq_len)
).cuda()

for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True)
embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
scalar_params = [model.scalars]
head_params: list[nn.Parameter] = [model.lm_head_w]

# init the optimizer(s)
adam_param_groups = [dict(params=head_params, lr=1/320), dict(params=embed_params, lr=0.3), dict(params=scalar_params, lr=0.015)]
optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)

if args.optimizer == "Adam":
    optimizer2 = torch.optim.AdamW(hidden_matrix_params, lr=1e-4, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.01)
elif args.optimizer == "Muon":
    optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size, use_Newton=args.use_Newton, k=args.k)
elif args.optimizer == "SGD":
    optimizer2 = torch.optim.SGD(hidden_matrix_params, lr=0.01, momentum=0.95, weight_decay=0.01)
    
optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]

def opt_params(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [p for group in opt.param_groups for p in group["params"]]
opt2params = {opt: opt_params(opt) for opt in optimizers}
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        return (1 - x) / args.cooldown_frac

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations
    assert 0 <= x <= 1
    factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x
    window_size = next_multiple_of_n(3456 * factor, n=128)
    return get_window_size_blocks_helper(window_size)

model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

warmup_steps = 10
initial_state = copy.deepcopy(dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers]))
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#        Training and validation       #
########################################
if master_process:
    loss_data = defaultdict(list)
    loss_data['train_steps'] = []
    loss_data['train_loss'] = []
    loss_data['val_loss'] = []
    loss_data['training_time_ms'] = []

torch.cuda.reset_peak_memory_stats()
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
dist.barrier()
t0 = time.perf_counter()

train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    model.zero_grad(set_to_none=False)
    train_loss = model(inputs, targets, get_window_size_blocks(step))
    train_loss.backward()
    if master_process:
        loss_data['train_steps'].append(step)
        loss_data['train_loss'].append(train_loss.item())
        loss_data['training_time_ms'].append(training_time_ms + 1000 * (time.perf_counter() - t0))
        
    if step % 500 == 0 and step > 0:
        if master_process:
            try:
                # 动态计算子图网格大小
                num_layers = len(model.blocks)
                ncols = 4  # 每行4个子图
                nrows = (num_layers + ncols - 1) // ncols  # 计算所需行数
                
                # 创建子图，调整图形大小适应层数
                fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
                fig.suptitle(f'Layer-wise Gradient Singular Values (Step {step})', fontsize=16)
                
                # 展平子图数组以便遍历
                if nrows == 1:
                    axs = axs.reshape(1, -1)  # 处理单行情况
                axs_flat = axs.ravel()
                
                for layer_idx, block in enumerate(model.blocks):
                    # 检查梯度是否存在
                    if not hasattr(block.attn, 'q_proj') or block.attn.q_proj.grad is None:
                        axs_flat[layer_idx].text(0.5, 0.5, 'No Grad', ha='center')
                        axs_flat[layer_idx].set_title(f'Layer {layer_idx}')
                        continue
                    
                    # 计算Q投影的奇异值
                    grad_Q = block.attn.q_proj.grad.float()
                    _, s_Q, _ = torch.svd(grad_Q)
                    s_Q = s_Q.cpu().numpy()
                    
                    # 绘制当前层
                    ax = axs_flat[layer_idx]
                    ax.plot(s_Q, label='Q', alpha=0.7, color='blue')
                    ax.set_title(f'Layer {layer_idx}')
                    ax.set_yscale('log')
                    ax.set_xlabel('SV Index')
                    ax.set_ylabel('Magnitude (log)')
                    ax.legend()
                    ax.grid(True, which="both", ls="--")
                
                # 隐藏多余子图
                for i in range(num_layers, nrows*ncols):
                    axs_flat[i].axis('off')
                
                # 调整布局并保存
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                os.makedirs(f"logs/{run_id_full}/grad_svd/", exist_ok=True)
                plt.savefig(f"logs/{run_id_full}/grad_svd/step_{step:04d}.png", 
                        bbox_inches='tight', dpi=150)
                plt.close(fig)
                
            except Exception as e:
                print(f"绘制梯度奇异值时出错: {str(e)}")
                if 'fig' in locals():
                    plt.close(fig)
        
        model.zero_grad(set_to_none=True)

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        dist.barrier()
        t0 = time.perf_counter()

        if master_process:
            loss_data['val_loss'].append(val_loss.item())
        
        if master_process and step % args.plot_every == 0:
            os.makedirs(f"logs/{run_id_full}", exist_ok=True)
            os.makedirs(f"logs/{run_id_full}/grad_svd", exist_ok=True)
            os.makedirs(f"logs/{run_id_full}/loss_plots", exist_ok=True)
            
            with open(f"logs/{run_id_full}/loss_data.pkl", 'wb') as f:
                pickle.dump(loss_data, f)
            
            plt.figure(figsize=(12, 6))
            
            if len(loss_data['train_loss']) > 0:
                plt.plot(loss_data['train_steps'], loss_data['train_loss'], 
                        label='Train Loss', alpha=0.7, color='blue')
            
            if len(loss_data['val_loss']) > 0:
                val_steps = loss_data['train_steps'][::args.val_loss_every][:len(loss_data['val_loss'])]
                plt.plot(val_steps, loss_data['val_loss'], 
                        label='Validation Loss', alpha=0.7, color='red')
            
            plt.title('Training and Validation Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--")
            plt.legend()
            
            os.makedirs(f"logs/{run_id_full}/loss_plots/", exist_ok=True)
            plt.savefig(f"logs/{run_id_full}/loss_plots/step_{step:04d}.png")
            plt.close()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id_full}", exist_ok=True)
            torch.save(log, f"logs/{run_id_full}/state_step{step:06d}.pt")
        break

    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step)).backward()
    opt2futures = {
        opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
        for opt, params in opt2params.items()
    }
    
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    
    for opt in optimizers:
        torch.futures.collect_all(opt2futures[opt]).wait()
        opt.step()
        
    model.zero_grad(set_to_none=True)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

if master_process:
    with open(f"logs/{run_id_full}/loss_data_final.pkl", 'wb') as f:
        pickle.dump(loss_data, f)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(loss_data['train_steps'], loss_data['train_loss'], 
            label='Train Loss', alpha=0.7, color='blue')
    
    if len(loss_data['val_loss']) > 0:
        val_steps = loss_data['train_steps'][::args.val_loss_every][:len(loss_data['val_loss'])]
        plt.plot(val_steps, loss_data['val_loss'], 
                label='Validation Loss', alpha=0.7, color='red')
    
    plt.title('Final Training and Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f"logs/{run_id_full}/loss_plots/final_loss.png")
    plt.close()

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()