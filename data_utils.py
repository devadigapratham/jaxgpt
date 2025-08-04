import jax
import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple
import requests
import os

class TextDataset:
    def __init__(self, text: str, seq_length: int = 1024):
        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        self.data = jnp.array([self.char_to_idx[ch] for ch in text])
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def get_batch(self, batch_size: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ix = jax.random.randint(key, (batch_size,), 0, len(self) - 1)
        x = jnp.stack([self.data[i:i+self.seq_length] for i in ix])
        y = jnp.stack([self.data[i+1:i+self.seq_length+1] for i in ix])
        return x, y

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "shakespeare.txt"
    
    if not os.path.exists(filename):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text

def create_simple_dataset():
    text = """Hello world! This is a simple dataset for testing our GPT-2 implementation.
The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
Machine learning is fascinating. Neural networks can learn complex patterns from data.
GPT models are autoregressive language models that generate text by predicting the next token.
JAX is a powerful library for machine learning research with automatic differentiation and JIT compilation.
Flax provides a flexible neural network library for JAX with a focus on composability and performance.
"""
    return text * 100

def get_dataset(dataset_name: str = "simple", seq_length: int = 64):
    if dataset_name == "shakespeare":
        try:
            text = download_shakespeare()
        except:
            print("Failed to download Shakespeare, using simple dataset")
            text = create_simple_dataset()
    else:
        text = create_simple_dataset()
    
    return TextDataset(text, seq_length)