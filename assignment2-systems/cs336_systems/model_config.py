from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int

# 各规格配置定义
small = ModelConfig(
    d_model=768,
    d_ff=3072,
    num_layers=12,
    num_heads=12
)

medium = ModelConfig(
    d_model=1024,
    d_ff=4096,
    num_layers=24,
    num_heads=16
)

large = ModelConfig(
    d_model=1280,
    d_ff=5120,
    num_layers=36,
    num_heads=20
)

xl = ModelConfig(
    d_model=2560,
    d_ff=10240,
    num_layers=32,
    num_heads=32
)

# 注意：Python 标识符不能以数字开头，因此 10B 命名为 m10b（或 b10）
m10b = ModelConfig(
    d_model=4608,
    d_ff=12288,
    num_layers=50,
    num_heads=36
)