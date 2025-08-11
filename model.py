
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.qk_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Fixed:  register once!
        self.register_buffer('rope_freqs', self._compute_rope_freqs())

    def _compute_rope_freqs(self):
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(self.max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, freqs)
        rope_freqs = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
        return rope_freqs  

    def _apply_rope(self, x, seq_len):
        x_pairs = x.view(*x.shape[:-1], -1, 2)  
        freqs = self.rope_freqs[:seq_len].unsqueeze(0).unsqueeze(0)  
        cos_freq = freqs[..., 0] 
        sin_freq = freqs[..., 1]
        x_rotated = torch.stack([
            x_pairs[..., 0] * cos_freq - x_pairs[..., 1] * sin_freq,
            x_pairs[..., 0] * sin_freq + x_pairs[..., 1] * cos_freq
        ], dim=-1)
        return x_rotated.view(*x.shape)

    def forward(self, x, mask=None, kv_cache=None):
        B, T, C = x.shape
        qk = self.qk_proj(x)
        q, k = qk.chunk(2, dim=-1)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)


        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2) 
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask[:, :, :T, :T]  # Adjust mask for current sequence length
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output), new_kv_cache


# class PureAttentionBlock(nn.Module):
#     def __init__(self, d_model, n_heads, max_seq_len=1024):
#         super().__init__()
#         self.attention = RoPEAttention(d_model, n_heads, max_seq_len)
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x, mask=None, kv_cache=None):
#         attn_output, new_kv_cache = self.attention(self.norm(x), mask, kv_cache)
#         return x + attn_output, new_kv_cache

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.attention = RoPEAttention(d_model, n_heads, max_seq_len)
        self.ffn = FeedForward(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        # Attention block
        attn_output, new_kv_cache = self.attention(self.norm1(x), mask, kv_cache)
        x = x + self.dropout(attn_output)
        
        # FFN block
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x, new_kv_cache


class TinyShakespeareTransformer(nn.Module):
    # def __init__(self, config):
    #     super().__init__()
    #     self.config = config
    #     self.max_seq_len = config['max_seq_len']

    #     self.token_embed = nn.Embedding(config['vocab_size'], config['d_model'])
        
    #     self.blocks = nn.ModuleList([
    #         TransformerBlock(
    #             config['d_model'],
    #             config['n_heads'],
    #             config['max_seq_len'],
    #             dropout=config.get('dropout', 0.1)
    #         ) for _ in range(config['n_layers'])
    #     ])
        
    #     self.norm_f = nn.LayerNorm(config['d_model'])
    #     self.output_proj = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
    #     self.output_proj.weight = self.token_embed.weight  # Tie weights

    #     self.apply(self._init_weights)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_len = config['max_seq_len']

        self.token_embed = nn.Embedding(config['vocab_size'], config['d_model'])
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config['d_model'], 
                config['n_heads'], 
                config['max_seq_len']
            ) for _ in range(config['n_layers'])
        ])
        
        self.norm_f = nn.LayerNorm(config['d_model'])
        self.output_proj = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.output_proj.weight = self.token_embed.weight  # Tie weights

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, kv_cache=None):
        B, T = x.shape
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        x = self.token_embed(x)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
            return_kv_cache = False
        else:
            return_kv_cache = True

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            block_kv_cache = kv_cache[i]
            x, layer_kv_cache = block(x, mask, block_kv_cache)
            new_kv_cache.append(layer_kv_cache)

        x = self.norm_f(x)
        logits = self.output_proj(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_kv_cache:
            return logits, loss, new_kv_cache
        else:
            return logits, loss  # Standard training return

    def count_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)