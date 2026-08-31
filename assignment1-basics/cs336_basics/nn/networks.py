import torch

from torch import nn
from cs336_basics.nn.basic import Linear, Embedding, RMSNorm, RotaryPositionalEmbedding
from cs336_basics.nn.functional import scaled_dot_product_attention, softmax
from cs336_basics.nn.utils import *
from einops import rearrange

class SwiGLU(nn.Module):
    d_model: int
    d_ff: int
    def __init__(self, d_model: int, d_ff: int | None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is not None:
            self.d_ff = d_ff
        else:
            self.d_ff = int(8/3 * d_model + 63) // 64 * 64
        
        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
            
    def SiLU(self, x:torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x:torch.Tensor):
        # return self.w2(self.SiLU(self.w1(x)) * self.w3(x))
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate)
        value = self.w3(x)
        return self.w2(gate * value)
    

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, theta:float=None, max_seq_len:int=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.q_proj_weight = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.k_proj_weight = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.v_proj_weight = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.o_proj_weight = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.theta = theta
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)

        self.register_buffer("causal_mask", None ,persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor|None=None):
        '''
        input x: shape(... sequence_length d_model)
        '''
        h = self.num_heads
        d_v = d_k = self.d_model // h
        seq_len = x.shape[-2]

        Q_raw = self.q_proj_weight(x)   # shape(... seq_len, dk*h)
        Q = rearrange(Q_raw, "... seq_len (h d_k) -> ... h seq_len d_k", h=h)    # Q[i] = head i = shape(..., seq_len, d_k)

        K_raw = self.k_proj_weight(x)
        K = rearrange(K_raw, "... seq_len (h d_k) -> ... h seq_len d_k", h=h)

        V_raw = self.v_proj_weight(x)
        V = rearrange(V_raw, "... seq_len (h d_v) -> ... h seq_len d_v", h=h)

        if self.theta:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            self.causal_mask = generate_causal_mask(dim=seq_len, device=x.device)
        mask = self.causal_mask[:seq_len, :seq_len]
        attention = scaled_dot_product_attention(K, Q, V, mask=mask)     #每个 head 独立做 Attention, 返回 shape(..., h, d_k, d_v)

        concat = rearrange(attention, "... h seq_len d_v -> ... seq_len (h d_v)", h=h, seq_len=seq_len)

        return self.o_proj_weight(concat)


class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta:float, max_seq_len:int, device=None, dtype=None):
        super().__init__()
        self.rms1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor):
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        sublayer1 = x + self.attn(self.rms1(x))
        sublayer2 = sublayer1 + self.ffn(self.rms2(sublayer1))
        return sublayer2
    

class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size:int, 
                 context_length:int, 
                 d_model:int, 
                 num_layers:int, 
                 num_heads:int, 
                 d_ff:int, 
                 rope_theta:float, 
                 device=None, 
                 dtype=None):
        super().__init__()
        factory_kwargs = {"device":device, "dtype":dtype}
        self.vocab_embed = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model=d_model, 
                              num_heads=num_heads, 
                              d_ff=d_ff, 
                              theta=rope_theta, 
                              max_seq_len=context_length, 
                              **factory_kwargs)
                              for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model=d_model, **factory_kwargs)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs) # Weights of the language model output embedding


    def forward(self, token_ids:torch.Tensor):
        '''
        input : shape(batch_size, sequence_length)
        这里传入的 sequence_length 会在 prompt 超过CONTEXT_LENGTH时被截断
        '''
        tok_embedding = self.vocab_embed(token_ids) # (B, seq_len, d_model)
        hidden_states = self.ln_final(self.layers(tok_embedding))
        logits = self.lm_head(hidden_states)
        return logits