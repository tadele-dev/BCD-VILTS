
Attention with motif/CpG bias and optional RoPE

---------------------------

class GenomicAttention(nn.Module):
    def init(self, config: GenomicConfig):
        super().init()
        self.embed_dim = config.d_model
        self.num_heads = config.num_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = config.use_rope
        self.max_positions = config.max_positions
        self.rope_base = config.rope_base
        if self.use_rope:
            self.register_buffer('_rope_cache', fixed_pos_embedding(self.max_positions, self.head_dim, self.rope_base), persistent=False)
        else:
            self._rope_cache = None
        # small learnable motif bias table (example for k=2 motifs like CpG)
        # motif_bias_map: dict from motif_id -> bias scalar (learnable)
        self.motif_bias = nn.Parameter(torch.zeros(16))  # e.g., for small motifs up to length 2^4
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, motif_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        if self.use_rope:
            rope_emb = self._rope_cache[:seq_len]
            q, k = apply_rope(q, k, rope_emb)
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale  # (b, heads, seq, seq)
        # apply motif bias if provided: motif_mask shape (batch, seq_len, seq_len) or (seq_len, seq_len)
        if motif_mask is not None:
            # motif_mask holds small motif ids per token-token pair (or scalar)
            # we'll map motif ids into bias via self.motif_bias and add scaled
            bias = torch.matmul(motif_mask.float(), self.motif_bias[:motif_mask.size(-1)]) if motif_mask.dim()==3 else motif_mask
            attn_scores = attn_scores + bias.unsqueeze(1) * 1.0
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1,2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.out_proj(attn_out)
        return out, attn_probs
