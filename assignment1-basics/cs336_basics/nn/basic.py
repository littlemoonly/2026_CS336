import torch
import torch.nn as nn
import math

# uv run pytest -k test_linear
class Linear(nn.Module):
    ''' 不设置 bias '''
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)) # 先创建一个 empty tensor，之后包进 nn.Parameter
        self._reset_parameters()

    def _reset_parameters(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        bound = 3 * std
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-bound, b=bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' y = Wx , 支持 leading dimensions，只改变最后一维 
        x 的最后一维应是 in_features
        '''
        return x @ self.weight.mT   # mT: 转置最后两个维度


class Embedding(nn.Module):
    weight: torch.Tensor # (vocab_size, d_model)

    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device=None, 
        dtype=None):
        '''
        num_embeddings 是词汇表大小
        embedding_dim 即每个词嵌入的维度(d_model)
        '''
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        '''N(miu = 0, sigma^2 = 1) truncated at [-3, 3]'''
        nn.init.trunc_normal_(self.weight, 
                              mean=0, std=1, 
                              a=-3, b=3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        输入一批 tokenIDs：(batch_size, sequence_length)
        查表返回 (batch_size, sequence_length, dmodel)
        '''
        assert token_ids.dtype == torch.long
        return self.weight[token_ids] # advanced indexing
    

class RMSNorm(nn.Module):
    ''' 在 d_model 维度上进行 RMSNorm '''
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # RMSNorm 的 gain 初始化为1, 从标准归一化开始学习
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Process an input tensor of shape(batch_size, sequence_length, d_model) and return a tensor of the same shape.'''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)   #[B, S, 1] 方便广播
        result = x * inv_rms * self.weight
        return result.to(in_dtype)
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__() 
        self.d_k = d_k
        
        i = torch.arange(0, d_k // 2, device=device)
        self.freqs = theta ** (-2*i / d_k)  # shape(d_k//2, ) 行向量

        self.pos = torch.arange(0, max_seq_len, device=device).unsqueeze(-1) # shape(max_seq_len, 1) 列向量
        # 计算每个可能的 token 位置的旋转角度 (max_seq_len, d_k/2)
        self.angles = self.pos * self.freqs
        self.cos_tensor = torch.cos(self.angles)
        self.sin_tensor = torch.sin(self.angles)
        self.register_buffer("cos", self.cos_tensor, persistent=False)
        self.register_buffer("sin", self.sin_tensor, persistent=False)


    def forward(self, x: torch.Tensor, token_positions:torch.Tensor|None=None) -> torch.Tensor:
        '''
        Process an input tensor (..., seq_len, d_k) and return the same shape
        token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension
        '''
        #  根据 token_positions 从预计算好的 cos、sin buffer 中取出对应位置的值
        ###### 注意 dtype!!!
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
        pos_cos = self.cos[token_positions].to(x.dtype) # shape(seq_len, dk/2)
        pos_sin = self.sin[token_positions].to(x.dtype)

        d_even = self.d_k // 2 * 2
        has_extra = self.d_k % 2


        x_even = x[..., 0:d_even:2]   # shape (..., seq_len, d_k/2)
        x_odd = x[..., 1:d_even:2]

        x_even_new = x_even * pos_cos - x_odd * pos_sin
        x_odd_new = x_odd * pos_cos + x_even * pos_sin

        ######### 拼回一起???
        interleaved = torch.stack([x_even_new, x_odd_new], dim=-1)  # (..., seq_len, d_k/2, 2)
        output = interleaved.flatten(start_dim=-2)
        if has_extra:
            output = torch.cat([output, x[..., -1:]], dim=-1)
        return output