# Training LLMs from Scratch 🚀

A minimal implementation of GPT (Generative Pre-trained Transformer) built from scratch in PyTorch. This project is adapted from Andrej Karpathy's [nanoGPT lecture series](https://github.com/karpathy/build-nanogpt) as part of **CS7CS4 Machine Learning** coursework at Trinity College Dublin [web:13][web:14][web:22].

## Overview

This repository implements a character-level or token-level language model based on the transformer architecture described in ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). The goal is to understand the fundamentals of large language models by building a simplified but functional GPT from the ground up [web:16].

### Key Features

- **Pure PyTorch implementation** of the GPT-2 architecture
- **Multi-head self-attention** with causal masking
- **Layer normalization** and residual connections
- **Configurable model size** (embedding dimensions, layers, heads)
- **Training from scratch** on custom datasets
- **Text generation** with sampling strategies (top-k, temperature)

## Architecture

The model implements the standard GPT-2 architecture with:
- **Transformer blocks** containing multi-head causal self-attention
- **Feed-forward MLP** with GELU activation
- **Layer normalization** applied before attention and MLP
- **Positional embeddings** for sequence ordering
- **Residual connections** for stable gradient flow

```
TransformerBlock(
  ├── LayerNorm
  ├── Multi-Head Attention (causal masking)
  ├── Residual Connection
  ├── LayerNorm
  ├── MLP (FFN)
  └── Residual Connection
)
```

## Installation

### Requirements

```bash
# Python 3.8+
pip install torch numpy tiktoken
```

### Clone the Repository

```bash
git clone https://github.com/ADVIKBAHADUR/training_LLMs.git
cd training_LLMs
```

## Dataset Preparation

### Option 1: Using Shakespeare Dataset (Default)

```bash
# Download the tiny shakespeare dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/shakespeare.txt
```

### Option 2: Custom Dataset

Place your text file in the `data/` directory and update the data path in `gpt.py`.

## Training the Model

### Basic Training

```bash
python gpt.py
```

### Training with Custom Hyperparameters

Edit the hyperparameters in `gpt.py`:

```python
# Model configuration
batch_size = 64          # How many independent sequences to process in parallel
block_size = 256         # Maximum context length for predictions
max_iters = 5000         # Number of training iterations
eval_interval = 500      # Evaluate loss every N iterations
learning_rate = 3e-4     # Learning rate for optimizer
eval_iters = 200         # Number of iterations for evaluation

# Model architecture
n_embd = 384            # Embedding dimension
n_head = 6              # Number of attention heads
n_layer = 6             # Number of transformer blocks
dropout = 0.2           # Dropout probability
```

### Training on GPU

```bash
# CUDA (NVIDIA)
CUDA_VISIBLE_DEVICES=0 python gpt.py

# MPS (Apple Silicon)
# Training will automatically use MPS if available
python gpt.py
```

## Testing the Model

### Generate Text from Trained Model

```bash
# Generate 500 tokens
python gpt.py --mode=generate --max_new_tokens=500
```

### Interactive Generation

```python
# Add to gpt.py or create test.py
import torch
from gpt import GPTLanguageModel

# Load trained model
model = GPTLanguageModel()
model.load_state_dict(torch.load('model_checkpoint.pth'))
model.eval()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500).tolist()
print(decode(generated))
```

### Evaluate Loss

```bash
# Evaluate validation loss
python gpt.py --mode=eval
```

### Sample Generation Commands

```python
# Temperature sampling (higher = more random)
model.generate(context, max_new_tokens=200, temperature=0.8)

# Top-k sampling
model.generate(context, max_new_tokens=200, top_k=40)

# Nucleus (top-p) sampling
model.generate(context, max_new_tokens=200, top_p=0.9)
```

## Model Checkpointing

The model automatically saves checkpoints during training:

```python
# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'iter': iter,
    'best_val_loss': best_val_loss,
}, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Project Structure

```
training_LLMs/
│
├── gpt.py              # Main model implementation and training loop
├── data/               # Training data directory
│   └── shakespeare.txt # Example dataset
├── checkpoints/        # Saved model checkpoints
├── README.md          # This file
└── LICENSE            # MIT License
```

## Key Concepts Demonstrated

1. **Self-Attention Mechanism**: How tokens communicate through weighted aggregation [web:16]
2. **Causal Masking**: Preventing attention to future tokens
3. **Positional Encoding**: Embedding position information
4. **Residual Connections**: Enabling deep network training
5. **Layer Normalization**: Stabilizing training dynamics [web:17]

## Performance Expectations

On tiny shakespeare (~1MB text):
- **Training time**: ~10-30 minutes on GPU (depends on config)
- **Validation loss**: ~1.5-2.0 after convergence
- **Generated text quality**: Coherent character-level Shakespeare-style text [web:21]

## Troubleshooting

### Out of Memory Errors
```python
# Reduce batch_size or block_size
batch_size = 32  # Instead of 64
block_size = 128 # Instead of 256
```

### Slow Training
```python
# Use torch.compile for 2x speedup (PyTorch 2.0+)
model = torch.compile(model)
```

### Poor Generation Quality
- Train for more iterations
- Increase model size (n_embd, n_layer)
- Use larger/better quality dataset
- Adjust learning rate

## Educational Resources

- [Andrej Karpathy's nanoGPT Video Lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) [web:16]
- [Attention is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [CS7CS4 Machine Learning - Trinity College Dublin](https://isg.beel.org/students_corner/teaching/past-modules/machine-learning-trinity-college-dublin/) [web:22]

## Acknowledgments

This implementation is adapted from:
- **Andrej Karpathy's nanoGPT lecture series** ([GitHub](https://github.com/karpathy/build-nanogpt)) [web:13]
- **CS7CS4 Machine Learning Module** at Trinity College Dublin [web:19][web:22]
- Original **"Attention is All You Need"** paper by Vaswani et al.

## License

MIT License - See original [nanoGPT repository](https://github.com/karpathy/nanoGPT) for details [web:14].

## Author

**Advik Bahadur**  
Electronic & Computer Engineering  
Trinity College Dublin

---

*Built as part of learning journey to understand LLMs from first principles 🧠*
```
