import torch
from pathlib import Path
import torch.nn as nn
import json
from typing import Optional

from utils import get_device
from cs336_basics.nn.networks import TransformerLM
from cs336_basics.tokenizer.bpe import Tokenizer
from default_config import default_cfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoint' / "ckp_final_5000.pt"

VOCAB_PATH = PROJECT_ROOT / "assets" / "tinystories_bpe_vocab.json"
MERGES_PATH = PROJECT_ROOT / "assets" / "tinystories_bpe_merges.txt"

CONTEXT_LENGTH = 256


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    logits: [batch_size, vocab_size]
    返回值: [batch_size, 1]
    """
    # temperature=0 时使用 greedy decoding
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p is None or top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)

    if top_p <= 0:
        raise ValueError("top_p must be in (0, 1]")

    # 按概率从大到小排列
    sorted_probs, sorted_indices = torch.sort(
        probs,
        dim=-1,
        descending=True,
    )

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 删除累积概率超过 top_p 的 token
    remove_mask = cumulative_probs > top_p

    # 保留第一个使累积概率达到 top_p 的 token
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(
        dim=-1,
        keepdim=True,
    )

    sampled_position = torch.multinomial(
        sorted_probs,
        num_samples=1,
    )

    return sorted_indices.gather(
        dim=-1,
        index=sampled_position,
    )

def load_model(ckp_path:Path)->nn.Module:
    device = get_device()
    checkpoint = torch.load(
        ckp_path,
        map_location=get_device(),
    )

    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        config = default_cfg
        print(f"Warning: using default model config")

    model = TransformerLM(
        vocab_size=config.get('vocab_size', 10000),
        context_length=config.get('context_length', 256),
        num_layers=config.get('num_layers', 4),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 16),
        rope_theta=config.get('rope_theta', 10000.0),
        d_ff=config.get('d_ff', 1344),
        dtype=torch.float32
    ).to(device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint['model'])  # 不需要 optimizer
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    
    return model

    
@torch.inference_mode()
def decode(
    model: nn.Module,
    tokenizer, 
    prompt: str,
    max_length: int=256,
    temperature: float=1.0,
    top_p: Optional[float] = 0.9,
    eos_token:Optional[str]="<endoftext>"
) -> str:
    """
    从 prompt 开始自回归生成文本, max_length 表示 prompt 加生成内容的总 token 数。
    """
    device = get_device()
    model.eval()

    prompt_ids = tokenizer.encode(prompt)   # list of int

    if not prompt_ids:
        raise ValueError("prompt cannot be empty")

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device) # 形状 [B=1, len(prompt)]

    while token_ids.shape[1] < max_length:
        # 模型一次最多读取 context_length 个 token
        model_input = token_ids[:, -CONTEXT_LENGTH:]
        # logits: [batch_size, sequence_length, vocab_size]
        logits = model(model_input)
        next_token_logits = logits[:, -1, :]    # 只使用最后一个位置预测下一个 token
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
        )   # [B, 1]

        if eos_token is not None:
            decoded_next_tok = tokenizer.decode([next_token[0].item()])
            if decoded_next_tok.strip() == eos_token.strip():
                break

        token_ids = torch.cat(
            [token_ids, next_token],
            dim=1,
        )

    return tokenizer.decode(token_ids[0].tolist())


if __name__ == "__main__":
    # 固定随机种子，方便重复测试
    torch.manual_seed(0)
    tokenizer = Tokenizer.from_files(
        vocab_filepath=VOCAB_PATH,
        merges_filepath=MERGES_PATH,
    )
    model = load_model(ckp_path=CHECKPOINT_PATH)
    result = decode(
        model=model,
        tokenizer=tokenizer,
        prompt="你好呀！",
        max_length=120,
        temperature=1.0,
        top_p=0.9,
        eos_token='<|endoftext|>'
    )
    print(result)