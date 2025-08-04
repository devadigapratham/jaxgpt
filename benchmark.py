import jax
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt
from model import GPT2LMHeadModel
from config import GPT2Config
from data_utils import get_dataset
from training_utils import create_train_state
import psutil
import os

class GPT2Benchmark:
    def __init__(self, model, state, dataset):
        self.model = model
        self.state = state
        self.dataset = dataset
        
    def benchmark_forward_pass(self, batch_sizes=[1, 2, 4, 8], seq_lengths=[64, 128, 256, 512]):
        results = []
        key = jax.random.PRNGKey(42)
        
        print("Benchmarking forward pass...")
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                if seq_length > self.dataset.seq_length:
                    continue
                    
                x, _ = self.dataset.get_batch(batch_size, key)
                x = x[:, :seq_length]
                
                warmup_runs = 3
                for _ in range(warmup_runs):
                    _ = self.state.apply_fn(self.state.params, x, deterministic=True)
                
                times = []
                num_runs = 10
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = self.state.apply_fn(self.state.params, x, deterministic=True)
                    jax.block_until_ready(_)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                throughput = (batch_size * seq_length) / avg_time
                
                results.append({
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'avg_time': avg_time,
                    'throughput': throughput
                })
                
                print(f"Batch {batch_size}, Seq {seq_length}: {avg_time:.4f}s, {throughput:.1f} tokens/s")
        
        return results
    
    def benchmark_generation(self, prompt_lengths=[10, 20, 50], generation_lengths=[50, 100, 200]):
        results = []
        key = jax.random.PRNGKey(42)
        
        print("Benchmarking text generation...")
        for prompt_len in prompt_lengths:
            for gen_len in generation_lengths:
                prompt = jnp.ones((1, prompt_len), dtype=jnp.int32)
                
                times = []
                num_runs = 5
                for _ in range(num_runs):
                    start_time = time.time()
                    
                    current_ids = prompt
                    for _ in range(gen_len):
                        if current_ids.shape[1] >= self.dataset.seq_length:
                            current_ids = current_ids[:, -self.dataset.seq_length+1:]
                        
                        logits = self.state.apply_fn(self.state.params, current_ids, deterministic=True)
                        next_token_logits = logits[:, -1, :]
                        key, subkey = jax.random.split(key)
                        next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
                        current_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)
                    
                    jax.block_until_ready(current_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                tokens_per_second = gen_len / avg_time
                
                results.append({
                    'prompt_length': prompt_len,
                    'generation_length': gen_len,
                    'avg_time': avg_time,
                    'tokens_per_second': tokens_per_second
                })
                
                print(f"Prompt {prompt_len}, Gen {gen_len}: {avg_time:.4f}s, {tokens_per_second:.1f} tokens/s")
        
        return results
    
    def benchmark_memory_usage(self):
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        key = jax.random.PRNGKey(42)
        batch_sizes = [1, 2, 4, 8, 16]
        memory_usage = []
        
        print("Benchmarking memory usage...")
        for batch_size in batch_sizes:
            try:
                x, _ = self.dataset.get_batch(batch_size, key)
                _ = self.state.apply_fn(self.state.params, x, deterministic=True)
                jax.block_until_ready(_)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_diff = current_memory - initial_memory
                
                memory_usage.append({
                    'batch_size': batch_size,
                    'memory_mb': memory_diff,
                    'memory_per_sample': memory_diff / batch_size
                })
                
                print(f"Batch {batch_size}: {memory_diff:.1f} MB total, {memory_diff/batch_size:.1f} MB/sample")
                
            except Exception as e:
                print(f"OOM at batch size {batch_size}: {e}")
                break
        
        return memory_usage
    
    def plot_benchmarks(self, forward_results, generation_results, memory_results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        batch_sizes = sorted(set([r['batch_size'] for r in forward_results]))
        seq_lengths = sorted(set([r['seq_length'] for r in forward_results]))
        
        throughput_matrix = np.zeros((len(batch_sizes), len(seq_lengths)))
        for i, bs in enumerate(batch_sizes):
            for j, sl in enumerate(seq_lengths):
                for r in forward_results:
                    if r['batch_size'] == bs and r['seq_length'] == sl:
                        throughput_matrix[i][j] = r['throughput']
                        break
        
        im = axes[0, 0].imshow(throughput_matrix, aspect='auto', cmap='viridis')
        axes[0, 0].set_xticks(range(len(seq_lengths)))
        axes[0, 0].set_xticklabels(seq_lengths)
        axes[0, 0].set_yticks(range(len(batch_sizes)))
        axes[0, 0].set_yticklabels(batch_sizes)
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Batch Size')
        axes[0, 0].set_title('Forward Pass Throughput (tokens/s)')
        plt.colorbar(im, ax=axes[0, 0])
        
        gen_lengths = [r['generation_length'] for r in generation_results]
        gen_speeds = [r['tokens_per_second'] for r in generation_results]
        axes[0, 1].scatter(gen_lengths, gen_speeds, alpha=0.7)
        axes[0, 1].set_xlabel('Generation Length')
        axes[0, 1].set_ylabel('Generation Speed (tokens/s)')
        axes[0, 1].set_title('Text Generation Speed')
        axes[0, 1].grid(True, alpha=0.3)
        
        if memory_results:
            batch_sizes_mem = [r['batch_size'] for r in memory_results]
            memory_usage = [r['memory_mb'] for r in memory_results]
            axes[1, 0].plot(batch_sizes_mem, memory_usage, 'o-')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Memory Usage (MB)')
            axes[1, 0].set_title('Memory Usage by Batch Size')
            axes[1, 0].grid(True, alpha=0.3)
        
        model_params = sum(x.size for x in jax.tree.leaves(self.state.params))
        model_size_mb = model_params * 4 / (1024 * 1024)
        
        axes[1, 1].bar(['Parameters', 'Model Size (MB)'], [model_params, model_size_mb])
        axes[1, 1].set_title('Model Statistics')
        axes[1, 1].set_ylabel('Count / Size')
        
        for i, v in enumerate([model_params, model_size_mb]):
            axes[1, 1].text(i, v + max(model_params, model_size_mb) * 0.01, f'{v:.1f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Benchmark results saved to benchmark_results.png")

def run_benchmarks(model, state, dataset):
    benchmark = GPT2Benchmark(model, state, dataset)
    
    forward_results = benchmark.benchmark_forward_pass()
    generation_results = benchmark.benchmark_generation()
    memory_results = benchmark.benchmark_memory_usage()
    
    benchmark.plot_benchmarks(forward_results, generation_results, memory_results)
    
    return {
        'forward_pass': forward_results,
        'generation': generation_results,
        'memory': memory_results
    }