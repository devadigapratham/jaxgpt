# jaxgpt

A complete implementation of GPT-2 using JAX and Flax, optimized for training and inference with comprehensive benchmarking.

## Features

- **Multi-file Architecture**: Clean separation of concerns across multiple modules
- **JAX/Flax Implementation**: Leverages JAX's JIT compilation and auto-differentiation
- **Progress Tracking**: Uses tqdm for training progress visualization
- **Comprehensive Benchmarking**: Forward pass, generation speed, and memory usage analysis
- **Automatic Plotting**: Training metrics and benchmark results visualization
- **Flexible Configuration**: Easy model and training parameter adjustment
- **Memory Optimized**: Designed for systems with 16GB RAM and mid-range GPUs

## Project Structure

```
├── config.py           # Model and training configurations
├── model.py            # GPT-2 model implementation
├── data_utils.py       # Dataset handling and utilities
├── training_utils.py   # Training loops and utilities
├── benchmark.py        # Performance benchmarking
├── train.py           # Main training script
├── inference.py       # Inference and text generation
├── requirements.txt   # Python dependencies
└── README.md         
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
python train.py --dataset simple --max_steps 2000
```

### Advanced Training

```bash
python train.py \
    --dataset shakespeare \
    --n_layer 8 \
    --n_embd 512 \
    --n_head 8 \
    --batch_size 6 \
    --max_steps 5000 \
    --learning_rate 3e-4
```

### Interactive Text Generation

```bash
python inference.py --checkpoint checkpoints/checkpoint_final.pkl --interactive
```

### Batch Generation

```bash
python inference.py \
    --checkpoint checkpoints/checkpoint_final.pkl \
    --prompts "Hello world" "The future of AI" "Once upon a time" \
    --output generated_samples.txt
```

## Configuration Options

### Model Configuration (config.py)

- `n_layer`: Number of transformer layers (default: 12)
- `n_embd`: Embedding dimension (default: 768)  
- `n_head`: Number of attention heads (default: 12)
- `n_positions`: Maximum sequence length (default: 1024)
- `vocab_size`: Vocabulary size (auto-detected from dataset)

### Training Configuration

- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Learning rate (default: 6e-4)
- `max_steps`: Maximum training steps (default: 10000)
- `warmup_steps`: Learning rate warmup steps (default: 1000)
- `eval_interval`: Evaluation frequency (default: 500)

## Memory Optimization

The implementation is optimized for systems with limited memory:

- **Gradient Checkpointing**: Reduces memory usage during backpropagation
- **Mixed Precision**: Uses bfloat16 where appropriate
- **Batch Size Scaling**: Automatically adjusts for available memory
- **Sequence Length Optimization**: Uses shorter sequences for memory efficiency

## Benchmarking

The benchmark suite includes:

1. **Forward Pass Performance**: Throughput across different batch sizes and sequence lengths
2. **Generation Speed**: Tokens per second for text generation
3. **Memory Usage**: RAM consumption patterns
4. **Model Statistics**: Parameter count and model size

Results are automatically plotted and saved as `benchmark_results.png`.

## Output Files

During training, the following files are generated:

- `training_metrics_step_X.png`: Training progress plots
- `final_training_metrics.png`: Complete training history
- `benchmark_results.png`: Performance benchmark visualization
- `checkpoints/checkpoint_X.pkl`: Model checkpoints
- `shakespeare.txt`: Downloaded dataset (if using Shakespeare)

## Hardware Requirements

**Minimum:**
- 8GB RAM
- GTX 1050 Ti or equivalent
- 2GB VRAM

**Recommended:**
- 16GB RAM  
- GTX 1650 Ti or better
- 4GB+ VRAM

## Troubleshooting

**Out of Memory Errors:**
- Reduce `batch_size` to 4 or 2
- Decrease `n_embd` and `n_layer`
- Use shorter sequence lengths

**Slow Training:**
- Enable JAX's JIT compilation
- Use GPU if available
- Increase batch size if memory allows

**Poor Generation Quality:**
- Train for more steps
- Use a larger model
- Adjust temperature during generation

## Advanced Usage

### Custom Dataset

Create your own dataset by modifying `data_utils.py`:

```python
def create_custom_dataset():
    with open('your_data.txt', 'r') as f:
        text = f.read()
    return text
```

### Model Architecture Changes

Modify `config.py` to experiment with different architectures:

```python
config = GPT2Config(
    n_layer=6,        # Smaller model
    n_embd=384,       # Reduced embedding size
    n_head=6,         # Fewer attention heads
    n_positions=256,  # Shorter context
)
```
