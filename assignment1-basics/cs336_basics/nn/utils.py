import torch


def generate_causal_mask(dim:int, device=None):
    # 生成[dim x dim]大小的掩码矩阵
    all_true = torch.ones(dim, dim, dtype=torch.bool, device=device)
    causal_mask = torch.tril(all_true) # tril 保留对角线及左下元素，其余置零
    return causal_mask


def get_flops(d=1600, n=1024, dff=None, vocab=50257, num_layers=48):
    ''' calculate FLOPs for Problem: transformer_accounting '''
    if dff is None:
        dff = round(d * 8 / 3 / 64) * 64

    mha = (4 * (2 * d * d * n) + 2 * (2 * d * n * n)) * num_layers
    ffn = (3 * (2 * d * n * dff)) * num_layers
    lm_head = 2 * d * n * vocab
    total = mha + ffn + lm_head

    items = [
        ("MHA (MultiHeadSelfAttention)", mha),
        ("FFN (SwiGLU)", ffn),
        ("LM Head", lm_head),
        ("Total", total),
    ]

    label_width = max(len(label) for label, _ in items)
    value_width = max(len(f"{value:,}") for _, value in items)

    for label, value in items:
        percentage = value / total * 100
        print(
            f"{label:<{label_width}} : "
            f"{value:>{value_width},}  "
            f"({percentage:6.2f}%)"
        )

    return total

def print_flops():
    gpt2_small = {"d":768, "num_layers":12}
    print("====== gpt2_small ======")
    get_flops(**gpt2_small)

    gpt2_medium = {"d":1024, "num_layers":24}
    print("====== gpt2_medium ======")
    get_flops(**gpt2_medium)

    gpt2_large = {"d":1280, "num_layers":36}
    print("====== gpt2_large ======")
    get_flops(**gpt2_large)

    print("====== GPT-2 XL ======")
    get_flops(dff=4288)

    gpt2_large_longcontext = {"d":1280, "num_layers":36, "n":16384}
    print("====== gpt2_large_longcontext ======")
    get_flops(**gpt2_large_longcontext)

def get_memory_usage(d_model=1600, n=1024, B=None, dff=None, vocab=50257, num_layers=48):
    '''
    n 即 seq_len
    B 即 batch size
    '''
    if dff is None:
        dff = round(d_model * 8 / 3 / 64) * 64
    

if __name__ == '__main__':
    print_flops()