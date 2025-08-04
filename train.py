import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import argparse
import os

from config import GPT2Config, TrainingConfig
from model import GPT2LMHeadModel
from data_utils import get_dataset
from training_utils import (
    create_train_state, train_step, eval_step, 
    generate_sample, plot_metrics, save_checkpoint
)
from benchmark import run_benchmarks

def train_model(model_config, training_config, dataset_name="simple"):
    key = jax.random.PRNGKey(training_config.seed)
    
    print("Loading dataset...")
    dataset = get_dataset(dataset_name, seq_length=model_config.n_positions//16)
    print(f"Dataset loaded: {len(dataset.text)} characters, vocab size: {dataset.vocab_size}")
    
    model_config.vocab_size = dataset.vocab_size
    model = GPT2LMHeadModel(model_config)
    
    print("Initializing model...")
    key, init_key = jax.random.split(key)
    state = create_train_state(
        model, 
        training_config, 
        init_key, 
        (training_config.batch_size, dataset.seq_length)
    )
    
    # JIT compile the training and eval functions
    jit_train_step = jax.jit(train_step)
    jit_eval_step = jax.jit(eval_step)
    
    total_params = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Model initialized: {total_params:,} parameters")
    
    train_losses = []
    eval_losses = []
    eval_steps = []
    
    print("Starting training...")
    pbar = tqdm(range(training_config.max_steps), desc="Training")
    
    for step in pbar:
        key, data_key = jax.random.split(key)
        x, y = dataset.get_batch(training_config.batch_size, data_key)
        
        key, train_key = jax.random.split(key)
        state, loss = jit_train_step(state, x, y, train_key)
        train_losses.append(float(loss))
        
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'ppl': f'{jnp.exp(loss):.2f}',
            'step': step + 1
        })
        
        if (step + 1) % training_config.eval_interval == 0:
            eval_loss_total = 0
            eval_batches = 10
            for _ in range(eval_batches):
                key, eval_key = jax.random.split(key)
                x_eval, y_eval = dataset.get_batch(training_config.batch_size, eval_key)
                eval_loss = jit_eval_step(state, x_eval, y_eval)
                eval_loss_total += eval_loss
            
            avg_eval_loss = eval_loss_total / eval_batches
            eval_losses.append(float(avg_eval_loss))
            eval_steps.append(step + 1)
            
            sample_text = generate_sample(
                state, dataset, prompt="The", max_length=100, 
                temperature=0.8, key=jax.random.split(key)[0]
            )
            
            tqdm.write(f"\nStep {step + 1}:")
            tqdm.write(f"Train Loss: {loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
            tqdm.write(f"Train PPL: {jnp.exp(loss):.2f}, Eval PPL: {jnp.exp(avg_eval_loss):.2f}")
            tqdm.write(f"Sample: {sample_text[:200]}...")
            tqdm.write("-" * 50)
        
        if (step + 1) % training_config.save_interval == 0:
            save_checkpoint(state, step + 1)
            plot_metrics(train_losses, eval_losses, eval_steps, 
                        f"training_metrics_step_{step + 1}.png")
    
    pbar.close()
    
    print("\nTraining completed!")
    print("Running benchmarks...")
    benchmark_results = run_benchmarks(model, state, dataset)
    
    print("\nFinal evaluation:")
    final_eval_loss = 0
    eval_batches = 50
    for _ in range(eval_batches):
        key, eval_key = jax.random.split(key)
        x_eval, y_eval = dataset.get_batch(training_config.batch_size, eval_key)
        eval_loss = jit_eval_step(state, x_eval, y_eval)
        final_eval_loss += eval_loss
    
    final_eval_loss /= eval_batches
    print(f"Final evaluation loss: {final_eval_loss:.4f}")
    print(f"Final evaluation perplexity: {jnp.exp(final_eval_loss):.2f}")
    
    print("\nGenerating samples:")
    for i, prompt in enumerate(["Hello", "The quick", "In the beginning", "Once upon"]):
        sample = generate_sample(
            state, dataset, prompt=prompt, max_length=200, 
            temperature=0.8, key=jax.random.split(key)[i]
        )
        print(f"Prompt: '{prompt}' -> {sample}")
        print()
    
    plot_metrics(train_losses, eval_losses, eval_steps, "final_training_metrics.png")
    save_checkpoint(state, "final")
    
    return state, benchmark_results

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 model")
    parser.add_argument("--dataset", default="simple", choices=["simple", "shakespeare"],
                       help="Dataset to use for training")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    model_config = GPT2Config(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=512,
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        eval_interval=200,
        save_interval=1000,
    )
    
    print("Configuration:")
    print(f"Model: {args.n_layer} layers, {args.n_embd} dim, {args.n_head} heads")
    print(f"Training: {args.batch_size} batch size, {args.max_steps} steps")
    print(f"Dataset: {args.dataset}")
    print()
    
    state, benchmark_results = train_model(model_config, training_config, args.dataset)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()