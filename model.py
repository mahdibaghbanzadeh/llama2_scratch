import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 4
    n_head : int = 4
    n_kv_head: Optional[int] = None # if None, n_kv_head = n_head
    vocab_size: int = -1 # set through tokenizer
    multiple_of: int = 8
    ffn_dim_multiplier: Optional[float] = None
    norms_eps: float = 1e-6

    # KV cache
    max_batch_size: int = 8
    max_seq_len: int = 256

    device: str = None

def precomputed_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta : float = 10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0, "dim must be even"

    # formula : theta_i = 10000 ^ (-2 (i-1)/dim)) for i = [1, 2, 3, ..., dim/2]
    theta_numerator = torch.arange(1, head_dim, 2, device=device).float() / head_dim
    theta_denominator = theta ** theta_numerator
    theta = 1.0 / theta_denominator # shape (head_dim // 2)

    m = torch.arange(seq_len, device=device).float() # shape (seq_len)
    # outer product
    freqs = torch.outer(m, theta) # shape (seq_len, head_dim // 2)

    # compute complex forms c = R * exp(i * theta), here R = 1
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    # (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(x.size(0), x.size(1), -1, 2))

    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (B, seq_len, H, head_dim /2) * (1, seq_len, 1, head_dim / 2) -> (B, seq_len, H, head_dim / 2)
    x_rot = x_complex * freqs_complex

    # (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim)
    x_out = torch.view_as_real(x_rot).reshape(x.size())
    return x_out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        assert args.voce_size > 0, "vocab_size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norms_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precomputed_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                               self.args.max_seq_len * 2,
                                                               device = self.args.device)
    
    def forward(self, tokens: torch.Tensor, start_pos: int):

        # (B, seq_len)
        batch_szie, seq_len = tokens.size()
        assert seq_len == 1 # only support autoregressive generation

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m , theta) for the positional encoding [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # apply the layers in transformer block
        for layer in self.layers:
            h = layer(h, start_pos , freqs_complex)
        
        h = self.norm(h)
        logits = self.output(h).float()
        return logits