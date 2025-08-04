from dataclasses import dataclass
from typing import Optional

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 10000
    eval_interval: int = 500
    save_interval: int = 1000
    gradient_clip_norm: float = 1.0
    seed: int = 42