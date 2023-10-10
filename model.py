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
    kv_heads: Optional[int] = None   # for Grouped Multi-Query Attention
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

    # build theta vector acc. to formula thetas = 10000 ^ -(2*i/dim) for i = [0,1,...h/2]
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
    
    mat = x_complex @ freqs_theta # (b, s, h, h_dim/2) * (1, s, 1, h_dim/2) -> (b, s, h, h_dim/2)

    x_out = torch.view_as_real(mat) # (b, s, h, h_dim) -> (b, s, h, h_dim/2, 2); extra dimension as each element goes from a+ib to [a, b] 
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)




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
            inp = layer(tokens, start, complex_freqs)

        inp = self.norm(inp)
        out = self.output(inp).float() 
        return out    


