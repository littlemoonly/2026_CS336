import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(
    model_id_or_dir: str,
    device: str,
    torch_dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
):
    device_type = torch.device(device).type
    if torch_dtype is None:
        torch_dtype = torch.float32 if device_type == "cpu" else torch.float16
    if attn_implementation is None:
        attn_implementation = "eager" if device_type == "cpu" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer
