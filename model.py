import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32     # for Q
    n_kv_heads: Optional[int] = None   # for Grouped Multi-Query Attention
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_hidden_mult: Optional[float] = None
    norm_eps: float = 1e-5

    # For KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None



def pos_freq_theta(h_dim: int, seq_len: int, device:str, factor:float = 10000):
    
    assert h_dim % 2 == 0, "Head dimension must be even!!"

    # build theta vector acc. to the formula thetas = 10000 ^ -(2*i/dim) for i = [0,1,...h/2]
    num = torch.arange(0, h_dim, 2)
    thetas = 1.0/ (factor ** (num/h_dim)).to(device)

    # positions
    m = torch.arange(seq_len, device=device)

    #outer product b/w m and thetas to give a matrix of shape: (seq_len, h/2)
    mat = torch.outer(m, thetas).float()

    #convert to polar coordinates; unit vectors in the orientations given by {mat}
    freqs_theta = torch.polar(torch.ones_like(mat), mat)

    return freqs_theta




def calc_rope(x: torch.Tensor, freqs_theta: torch.Tensor, device:str):

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # in the [x1 + ix2] form
    
    freqs_theta = freqs_theta.unsqueeze(0).unsqueeze(2) # (s,h_dim/2) -> (1, s, 1, h_dim/2)
    
    mat = x_complex * freqs_theta # (b, s, h, h_dim/2) * (1, s, 1, h_dim/2) -> (b, s, h, h_dim/2)

    x_out = torch.view_as_real(mat) # (b, s, h, h_dim) -> (b, s, h, h_dim/2, 2); extra dimension as each element goes from a+ib to [a, b] 
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


def replicate_kv(x:torch.Tensor, n_rep:int):
    b, s, kv_h, h_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (x[:, :, :, None, :].expand(b, s, kv_h, n_rep, h_dim).reshape(b, s, kv_h * n_rep, h_dim))


class FeedForward(nn.Module):

    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        self.hidden_dim = int(8 * args.dim / 3)

        # Round to nearest multiple of args.multiple_of greater than hidden dim
        self.hidden_dim = args.multiple_of * ((self.hidden_dim + args.multiple_of -1) // args.multiple_of)

        # swiglu (x, V, W, w2) = (swish(xW) @ xV) w2
        self.w1 = nn.Linear(args.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, self.hidden_dim, bias=False)

    def forward(self, x:torch.Tensor):
        swish = F.silu(self.w1(x))
        xV = self.w3(x)
        x = swish * xV
        return self.w2(x)    



class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps:float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor):
        # (b,s,d) * (b,s,1) = (b,s,d)
        # rsqrt = 1/sqrt
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor):
        # (dim) * (b,s,d) = (b,s,d)
        return self.weight * self._norm(x.float()).type_as(x)  
    


class SelfAttention(nn.Module):

    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        # TODO: parallelization removed here; check out in og LLaMA repo
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.q_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.q_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, self.head_dim * self.q_heads, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.k_cache = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.v_cache = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))


    def forward(self, x:torch.Tensor, start:int, freqs:torch.Tensor):
        b, s, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        xq = xq.view(b, s, self.q_heads, self.head_dim)
        xk = xk.view(b, s, self.n_kv_heads, self.head_dim)
        xv = xv.view(b, s, self.n_kv_heads, self.head_dim)

        # apply RoPE to Q and K only
        xq = calc_rope(xq, freqs, x.device)
        xk = calc_rope(xk, freqs, x.device)

        # Replace entry in cache with current xk and xv
        self.k_cache[:b, start:start+s] = xk
        self.v_cache[:b, start:start+s] = xv

        # Retrieve all cached keys and values till current position
        keys = self.k_cache[:b, :start+s]
        values = self.v_cache[:b, :start+s]

        # replicate key and value heads to match q heads
        keys = replicate_kv(keys, self.n_rep)
        values = replicate_kv(values, self.n_rep)

        # (b, 1, h_q, h_dim) -> (b, h_q, 1, h_dim); each head needs to look at entire seq_len (which is 1 for q, and seq_len_kv for k and v)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # (b,h_q,1,h_dim) @ (b, h_q, h_dim, seq_len_kv) -> (b, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (b, h_q, 1, seq_len_kv) @ (b, h_q, seq_len_kv, h_dim) -> (b, h_q, 1, h_dim)
        out = torch.matmul(scores, values)

        # (b, h_q, 1, h_dim) -> (b, 1, h_q, h_dim) -> (b, 1, dim)
        out = (out.transpose(1,2).contiguous().view(b, s, -1))
        return self.wo(out)

    

class EncoderBlock(nn.Module):

    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention  = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)

    def forward(self, x:torch.Tensor, start:int, freqs: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start, freqs)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out    


class Transformer(nn.Module):

    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, "Vocab size not set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.complex_freq = pos_freq_theta(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)    

    def forward(self, tokens: torch.Tensor, start: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "KV Cache; use only latest token's query"
        
        # (b, s) -> (b, s, d)
        inp = self.tok_embeddings(tokens)

        # RoPE freqs
        complex_freqs = self.complex_freq[start:start + seq_len]

        for layer in self.layers:
            inp = layer(inp, start, complex_freqs)

        inp = self.norm(inp)
        out = self.output(inp).float() 
        return out    


