import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Tuple, Any
import matplotlib.pyplot as plt
import os

def create_learning_rate_schedule(warmup_steps: int, learning_rate: float):
    def schedule(step):
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        decay_factor = jnp.maximum(0.1, 1.0 - (step - warmup_steps) / 50000)
        return learning_rate * warmup_factor * decay_factor
    return schedule

def create_train_state(model, config, key, input_shape):
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
    params = model.init(key, dummy_input)
    
    schedule = create_learning_rate_schedule(config.warmup_steps, config.learning_rate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def compute_loss(params, apply_fn, x, y, key):
    logits = apply_fn(params, x, deterministic=False, rngs={'dropout': key})
    
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1)
    ).mean()
    
    return loss, logits

def train_step(state, x, y, key):
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, x, y, key)
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def eval_step(state, x, y):
    logits = state.apply_fn(state.params, x, deterministic=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1)
    ).mean()
    return loss

def calculate_perplexity(loss):
    return jnp.exp(loss)

def generate_sample(state, dataset, prompt="Hello", max_length=100, temperature=0.8, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)
    
    context = jnp.array([[dataset.char_to_idx.get(ch, 0) for ch in prompt]])
    
    for _ in range(max_length):
        if context.shape[1] >= dataset.seq_length:
            context = context[:, -dataset.seq_length+1:]
        
        logits = state.apply_fn(state.params, context, deterministic=True)
        next_logits = logits[0, -1, :] / temperature
        
        key, subkey = jax.random.split(key)
        next_idx = jax.random.categorical(subkey, next_logits)
        context = jnp.concatenate([context, next_idx[None, None]], axis=1)
    
    generated_indices = context[0].tolist()
    generated_text = ''.join([dataset.idx_to_char.get(idx, '?') for idx in generated_indices])
    
    return generated_text

def plot_metrics(train_losses, eval_losses, eval_steps, save_path="training_metrics.png"):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(eval_steps, eval_losses, label='Eval Loss', marker='o')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    train_perplexity = [jnp.exp(loss) for loss in train_losses]
    eval_perplexity = [jnp.exp(loss) for loss in eval_losses]
    plt.plot(train_perplexity, label='Train Perplexity', alpha=0.7)
    plt.plot(eval_steps, eval_perplexity, label='Eval Perplexity', marker='o')
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.title('Training and Evaluation Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {save_path}")

def save_checkpoint(state, step, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{step}.pkl")
    
    # Only save the parameters, not the entire state
    checkpoint_data = {
        'params': state.params,
        'step': step
    }
    
    with open(checkpoint_path, 'wb') as f:
        import pickle
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved to {checkpoint_path}")