import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
from config import GPT2Config

class GPT2Attention(nn.Module):
    config: GPT2Config
    
    def setup(self):
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_attn = nn.Dense(3 * self.n_embd, use_bias=True)
        self.c_proj = nn.Dense(self.n_embd, use_bias=True)
        self.attn_dropout = nn.Dropout(self.config.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.config.resid_pdrop)

    def __call__(self, x, mask=None, deterministic=True):
        B, T, C = x.shape
        
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / jnp.sqrt(self.head_dim))
        
        causal_mask = jnp.tril(jnp.ones((T, T)))
        att = jnp.where(causal_mask == 0, -jnp.inf, att)
        
        if mask is not None:
            att = jnp.where(mask[:, None, None, :] == 0, -jnp.inf, att)
        
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=deterministic)
        
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        y = self.c_proj(y)
        y = self.resid_dropout(y, deterministic=deterministic)
        
        return y

class GPT2MLP(nn.Module):
    config: GPT2Config
    
    def setup(self):
        n_inner = self.config.n_inner or 4 * self.config.n_embd
        self.c_fc = nn.Dense(n_inner, use_bias=True)
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=True)
        self.dropout = nn.Dropout(self.config.resid_pdrop)

    def __call__(self, x, deterministic=True):
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

class GPT2Block(nn.Module):
    config: GPT2Config
    
    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)
        self.attn = GPT2Attention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)
        self.mlp = GPT2MLP(self.config)

    def __call__(self, x, mask=None, deterministic=True):
        x = x + self.attn(self.ln_1(x), mask=mask, deterministic=deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x

class GPT2Model(nn.Module):
    config: GPT2Config
    
    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.n_positions, self.config.n_embd)
        self.drop = nn.Dropout(self.config.embd_pdrop)
        self.h = [GPT2Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)

    def __call__(self, input_ids, mask=None, deterministic=True):
        B, T = input_ids.shape
        pos = jnp.arange(0, T)[None, :]
        
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb, deterministic=deterministic)
        
        for block in self.h:
            x = block(x, mask=mask, deterministic=deterministic)
        
        x = self.ln_f(x)
        return x

class GPT2LMHeadModel(nn.Module):
    config: GPT2Config
    
    def setup(self):
        self.transformer = GPT2Model(self.config)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(self, input_ids, mask=None, deterministic=True):
        x = self.transformer(input_ids, mask=mask, deterministic=deterministic)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_length=50, temperature=1.0, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        
        for _ in range(max_length - input_ids.shape[1]):
            logits = self(input_ids, deterministic=True)
            next_token_logits = logits[:, -1, :] / temperature
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
            input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
        
        return input_ids