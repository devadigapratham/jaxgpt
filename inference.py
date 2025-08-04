import jax
import jax.numpy as jnp
import pickle
import argparse
from config import GPT2Config
from model import GPT2LMHeadModel
from data_utils import get_dataset
from training_utils import generate_sample

def load_checkpoint(checkpoint_path, model, dataset):
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Handle both old and new checkpoint formats
    if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        params = checkpoint_data['params']
        step = checkpoint_data.get('step', 'unknown')
    else:
        # Old format - entire state was saved
        params = checkpoint_data.params
        step = 'unknown'
    
    # Create a minimal state-like object for inference
    class InferenceState:
        def __init__(self, params, apply_fn):
            self.params = params
            self.apply_fn = apply_fn
    
    config = GPT2Config(vocab_size=dataset.vocab_size)
    model_instance = GPT2LMHeadModel(config)
    
    # Create dummy apply function
    def apply_fn(params, x, deterministic=True, rngs=None):
        return model_instance.apply(params, x, deterministic=deterministic, rngs=rngs)
    
    state = InferenceState(params, apply_fn)
    
    print(f"Loaded checkpoint from step {step}")
    return state

def interactive_generation(state, dataset):
    print("Interactive text generation (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        prompt = input("\nEnter prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            prompt = "The"
        
        try:
            temperature = input("Temperature (0.1-2.0, default 0.8): ").strip()
            temperature = float(temperature) if temperature else 0.8
            temperature = max(0.1, min(2.0, temperature))
        except:
            temperature = 0.8
        
        try:
            max_length = input("Max length (default 150): ").strip()
            max_length = int(max_length) if max_length else 150
            max_length = max(10, min(500, max_length))
        except:
            max_length = 150
        
        print(f"\nGenerating with prompt: '{prompt}', temp: {temperature}, length: {max_length}")
        print("-" * 30)
        
        key = jax.random.PRNGKey(jax.random.randint(jax.random.PRNGKey(42), (), 0, 10000))
        generated_text = generate_sample(
            state, dataset, prompt=prompt, max_length=max_length, 
            temperature=temperature, key=key
        )
        
        print(generated_text)
        print("-" * 50)

def batch_generation(state, dataset, prompts, output_file=None):
    results = []
    key = jax.random.PRNGKey(42)
    
    for i, prompt in enumerate(prompts):
        print(f"Generating {i+1}/{len(prompts)}: '{prompt}'")
        
        key, subkey = jax.random.split(key)
        generated_text = generate_sample(
            state, dataset, prompt=prompt, max_length=200, 
            temperature=0.8, key=subkey
        )
        
        results.append({
            'prompt': prompt,
            'generated': generated_text
        })
        
        print(f"Result: {generated_text[:100]}...")
        print()
    
    if output_file:
        with open(output_file, 'w') as f:
            for result in results:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Generated: {result['generated']}\n")
                f.write("-" * 80 + "\n")
        print(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained GPT-2 model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default="simple", choices=["simple", "shakespeare"],
                       help="Dataset used for training (for vocab)")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive generation")
    parser.add_argument("--prompts", nargs="+", 
                       help="List of prompts for batch generation")
    parser.add_argument("--output", help="Output file for batch generation")
    
    args = parser.parse_args()
    
    print("Loading dataset for vocabulary...")
    dataset = get_dataset(args.dataset, seq_length=64)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    state = load_checkpoint(args.checkpoint, None, dataset)
    
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {dataset.vocab_size}")
    
    if args.interactive:
        interactive_generation(state, dataset)
    elif args.prompts:
        batch_generation(state, dataset, args.prompts, args.output)
    else:
        default_prompts = [
            "Hello world",
            "The quick brown fox",
            "In the beginning",
            "Once upon a time",
            "Machine learning is"
        ]
        print("Running with default prompts...")
        batch_generation(state, dataset, default_prompts)

if __name__ == "__main__":
    main()