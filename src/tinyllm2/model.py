import torch
import torch.nn.functional as F
from torch import nn


class PreLNGPTDecoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_dim: int,
        num_heads: int,
        drop_out_rate: float = 0.0,
        layer_eps: float = 1.0e-5,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.masked_multihead_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=batch_first
        )
        self.dropout_self_attn = nn.Dropout(p=drop_out_rate)
        self.layer_norm_self_attn = nn.LayerNorm(embedding_dim, eps=layer_eps)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
        )
        self.layer_norm_ffn = nn.LayerNorm(embedding_dim, eps=layer_eps)
        self.dropout_ffn = nn.Dropout(p=drop_out_rate)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask_self: torch.Tensor | None = None,
        mask_self: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_input = self.layer_norm_self_attn(x)
        attention_output, _ = self.masked_multihead_attention(
            attention_input,
            attention_input,
            attention_input,
            key_padding_mask=pad_mask_self,
            attn_mask=mask_self,
        )
        x = x + attention_output

        ffn_input = self.layer_norm_ffn(x)
        ffn_output = self.dropout_ffn(self.ffn(ffn_input))
        x = x + ffn_output

        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        ffn_dim: int,
        num_heads: int,
        drop_out_rate: float = 0.0,
        layer_eps: float = 1.0e-5,
        batch_first: bool = False,
        T: int = 10000,
        N: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(T, embedding_dim)
        self.decoder = nn.ModuleList(
            [
                PreLNGPTDecoderLayer(
                    embedding_dim,
                    ffn_dim,
                    num_heads,
                    drop_out_rate,
                    layer_eps,
                    batch_first,
                )
                for _ in range(N)
            ]
        )
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        pad_mask_self: torch.Tensor | None = None,
        mask_self: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        x = self.embedding(x)
        pos_ = torch.arange(0, x.size(1), dtype=torch.long).unsqueeze(0).to(x.device)
        pos = self.positional_embedding(pos_)
        x = x + pos
        for layer in self.decoder:
            x = layer(x, pad_mask_self=pad_mask_self, mask_self=mask_self)
        x = self.linear(x)

        if y is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-1)
            pred = x.argmax(dim=-1).detach().cpu()
            return loss, pred

        loss = None
        pred = x[:, [-1], :]
        return loss, pred

    def create_mask(
        self,
        x: torch.Tensor,
        x_pad: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        padding_mask = (x == x_pad)
        mask = torch.triu(torch.ones(size=(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0.0, float("-inf"))
            .masked_fill(mask == 1.0, float(0.0))
            .to(device)
        )

        return padding_mask, mask

    @torch.no_grad()
    def generate(
        self,
        bos: str,
        sentence_size: int,
        tokenizer,
        device,
    ) -> torch.Tensor:
        self.eval()
        bos_tokenized = tokenizer.encode_ordinary(bos)
        bos_tokenized = bos_tokenized[-sentence_size:]
        bos_tokenized = torch.LongTensor([bos_tokenized])
        _, add_sentence = self(bos_tokenized.to(device))
        self.train()
        return add_sentence

    @torch.no_grad()
    def generate_sentence(
        self,
        bos: str,
        sentence_size: int,
        generate_tokens,
        tokenizer,
        device,
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
