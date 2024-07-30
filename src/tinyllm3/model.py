import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tokenizers import Tokenizer


@dataclass
class Config:
    dim: int = 768
    n_layers: int = 6
    n_heads: int = 6
    vocab_size: int = -1
    multiple_of: int = 1
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    max_seq_len: int = 1024
    max_batch_size: int = 8
    ffn_dim_multiplier: int | None = None 


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(
    freqs_cis: torch.Tensor,
    x: torch.Tensor,
):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        self.dim = conf.dim
        self.n_heads = conf.n_heads
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bsz, n_heads, cache_len+seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bsz, n_heads, cache_len+seqlen, head_dim)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask[None, None, :, :]
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, conf: Config):
        super().__init__()
        self.dim = conf.dim
        self.n_heads = conf.n_heads
        self.head_dim = self.dim // self.n_heads
        self.attention = Attention(conf)
        self.feed_forward = FeedForward(
            dim=self.dim,
            hidden_dim=4 * self.dim,
            multiple_of=conf.multiple_of,
            ffn_dim_multiplier=conf.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(self.dim, eps=conf.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=conf.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        self.conf = conf
        self.vocab_size = conf.vocab_size
        self.n_layers = conf.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, conf.dim)
        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, conf))
        self.norm = RMSNorm(conf.dim, eps=conf.norm_eps)
        self.output = nn.Linear(conf.dim, conf.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(
            conf.dim // conf.n_heads,
            conf.max_seq_len * 2,
            conf.rope_theta,
        ).cuda()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # mask = torch.hstack(
            #     [torch.zeros((seqlen, 0), device=tokens.device), mask]
            # ).type_as(h)
            mask = mask.type_as(h)
        # print(f"mask={mask}")

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float()
        return output

    @torch.inference_mode()
    def generate(
        self,
        bos: str,
        sentence_size: int,
        tokenizer: Tokenizer,
        device: str,
    ) -> torch.Tensor:
        self.eval()
        bos_tokenized = np.array(tokenizer.encode(bos).ids)
        bos_tokenized = bos_tokenized[-sentence_size:]
        bos_tokenized = bos_tokenized.reshape(1, -1)
        bos_tokenized = torch.LongTensor(bos_tokenized)
        add_sentence = self(bos_tokenized.to(device))
        self.train()
        return add_sentence

    @torch.inference_mode()
    def generate_sentence(
        self,
        bos: str,
        sentence_size: int,
        generate_tokens,
        tokenizer: Tokenizer,
        device: str,
        top_K=None,
        temperature: float = 1.0,
    ):
        return_sentence = bos
        for i in range(generate_tokens):
            add_sentence = self.generate(
                return_sentence, sentence_size, tokenizer, device
            )
            add_sentence = add_sentence[:, -1, :] / temperature

            if top_K is not None:
                v, _ = torch.topk(add_sentence, min(top_K, add_sentence.size(-1)))
                add_sentence[add_sentence < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(add_sentence, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            return_sentence += tokenizer.decode_batch(idx_next.tolist())[0]
        return return_sentence

    def calculate_params(self) -> int:
        count_params = 0
        for params in self.parameters():
            count_params += params.contiguous().view(-1).size(0)

        return count_params
