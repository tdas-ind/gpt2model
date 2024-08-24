from dataclasses import dataclass
import torch
import inspect
import math
import time
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------

#FlashAttention is a technique designed to make the attention mechanism in transformers faster and more memory-efficient,
#  especially when dealing with large sequences. The standard attention mechanism is both computationally expensive and 
# memory-intensive, as it involves computing and storing large matrices. FlashAttention addresses these issues by optimizing
#  the way attention is calculated. torch.compile cannot read the attention mechanism so to make the attention mechanishm faster

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias' more of a mask, but following the openAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1,
                                                                                                        config.block_size,
                                                                                                        config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence length, embedding dimmensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads" , hs is "head size" and C (number of channels) = nh * hs
        # e.g in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C-768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # attention (materialises the large(T,T) matrx for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash Attention

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


# The projection layer in the MLP (Multi-Layer Perceptron) is used to adjust the dimensionality of the intermediate hidden states. Let’s break down why this is done:

# Understanding the MLP Structure

# In the context of transformer models (like GPT), the MLP typically follows the self-attention mechanism and is responsible for further processing the output. The MLP block often has two linear layers:

# 	1.	First Linear Layer (self.c_fc):
# 	•	This expands the embedding dimension from config.n_embd to a higher dimension, often 4 * config.n_embd.
# 	•	This expansion allows the model to project the input into a higher-dimensional space, which enables it to model more complex relationships.
# 	2.	Non-linearity (self.gelu):
# 	•	The non-linear activation function (in this case, GELU) introduces non-linearity to the model, allowing it to learn more complex functions.
# 	3.	Second Linear Layer (self.c_proj):
# 	•	This layer projects the expanded dimension back down to the original embedding size (config.n_embd).
# 	•	The purpose is to return to the original dimensionality of the input while incorporating the more complex features learned during the intermediate expansion.

# Why Projection is Used

# 	1.	Dimensionality Reduction:
# 	•	After expanding the dimension and applying non-linearity, the second projection layer reduces the dimensionality back to the original size. This ensures that the output has the same dimensionality as the input, making it compatible with other layers in the model.

#   2.	Modeling Capacity:
# 	•	The projection (expanding then contracting) allows the model to temporarily operate in a higher-dimensional space where it can capture richer patterns and features. The contraction afterward ensures that the model remains efficient and doesn’t explode in size.
# 	3.	Efficiency and Flexibility:
# 	•	The process of expanding and then contracting enables the model to represent more complex transformations without significantly increasing the model size permanently.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|>
    n_layer: int = 12  # num of layers
    n_head: int = 12  # num of heads
    n_embd: int = 768  # embedding dimmension


class GPT(nn.Module):


    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B,T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posittion embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embedding of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
 
    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pre-trained GPT-2 moel weights from hugging face"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, n_embs are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)  # 124M parameters

        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask /buffer

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special tratment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other patterns
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e, all weight tensors in matmuls _ embedding decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_deca': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f" num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f" using fused AdamW {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps = 1e-8,fused=use_fused)
        return optimizer

        

# -------------------------------

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open("input.txt", 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # tensor
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    
# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
device = "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.manual_seed(1337)

total_batch_size = 524288 # 2**19 ~ 0.5M, in number of tokens
B = 8 # micro_batch_size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total batch_size available is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=8, T= 128)

torch.set_float32_matmul_precision('high')
# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
# logits, loss = model(x, y)
# print(loss)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, we use cosine decay to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps = 1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate = 6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits,loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    # torch.cuda.synchronize()
    t1 =time.time()
    dt = (t1 - t0)* 1000 # time difference in milli seconds
    # tokens_per_sec = (train_loader.B * train_loader.T) /(t1 -t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step}, loss: {loss_accum.item()} | lr: {lr:.4e} | norm: {norm} | dt: {dt:.2f} ms | tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a Language model,") # (8,)
tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
tokens = tokens.unsqueeze(0). repeat(num_return_sequences, 1) # (5,8)

x = tokens.to(device)


# generate ! right now x is (B, T) where B=5, T=8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) #(B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (hugging face pipeline default)
        # top-k_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) #(B,1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)


# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded
          )
