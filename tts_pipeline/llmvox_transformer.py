import math

import torch
import torch.nn.functional as F
from torch import nn


def llmvox_gelu(x: torch.Tensor) -> torch.Tensor:
    """The tanh GELU used by the public LLMVoX GPT implementation."""
    return 0.5 * x * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        )
    )


class LLMVoxCausalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.output = nn.Linear(dim, dim, bias=bias)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.residual_dropout(self.output(output))


class LLMVoxMLP(nn.Module):
    def __init__(self, dim: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim, bias=bias)
        self.down = nn.Linear(4 * dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(llmvox_gelu(self.up(x))))


class LLMVoxBlock(nn.Module):
    """Pre-norm causal GPT block matching the LLMVoX decoder design."""

    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim, bias=bias)
        self.attention = LLMVoxCausalAttention(dim, num_heads, dropout, bias)
        self.mlp_norm = nn.LayerNorm(dim, bias=bias)
        self.mlp = LLMVoxMLP(dim, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class LLMVoxDecoder(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 8192,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.position_embedding = nn.Embedding(max_length, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [LLMVoxBlock(dim, num_heads, dropout, bias) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim, bias=bias)
        self.apply(self._initialize)
        for name, parameter in self.named_parameters():
            if name.endswith("output.weight") or name.endswith("down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[1]
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length {self.max_length}")
        positions = torch.arange(length, device=inputs.device)
        hidden = self.dropout(inputs + self.position_embedding(positions)[None])
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

