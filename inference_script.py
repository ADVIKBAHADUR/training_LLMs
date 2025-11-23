import torch
import torch.nn as nn
from torch.nn import functional as F

# Model definition (same as training script)
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.25):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(checkpoint_path, device='xpu'):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract hyperparameters
    vocab_size = checkpoint['vocab_size']
    n_embd = checkpoint['n_embd']
    n_head = checkpoint['n_head']
    n_layer = checkpoint['n_layer']
    block_size = checkpoint['block_size']

    # Initialize model
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load encoding/decoding functions
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Print model info
    num_params = count_parameters(model)
    print(f"Model loaded from iteration {checkpoint['iter']}")
    print(f"Number of parameters: {num_params:,}")
    print(f"Train loss: {checkpoint['train_loss']:.4f}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")

    return model, encode, decode

def evaluate_on_test_set(model, test_data, block_size, device='cpu', batch_size=32):
    """Evaluate model on test dataset and return loss"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        # Process test data in batches
        for i in range(0, len(test_data) - block_size - 1, batch_size * block_size):
            # Create batch
            batch_data = []
            batch_targets = []

            for j in range(batch_size):
                start_idx = i + j * block_size
                if start_idx + block_size + 1 > len(test_data):
                    break
                batch_data.append(test_data[start_idx:start_idx + block_size])
                batch_targets.append(test_data[start_idx + 1:start_idx + block_size + 1])

            if len(batch_data) == 0:
                break

            # Convert to tensors
            x = torch.tensor(batch_data, dtype=torch.long, device=device)
            y = torch.tensor(batch_targets, dtype=torch.long, device=device)

            # Forward pass
            logits, loss = model(x, y)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def generate_text(model, encode, decode, prompt="", max_tokens=500, device='cpu'):
    """Generate text from the model"""
    model.eval()
    with torch.no_grad():
        if prompt:
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        generated = model.generate(context, max_new_tokens=max_tokens)
        text = decode(generated[0].tolist())

    return text

if __name__ == "__main__":
    # Configuration
    device = 'xpu' if torch.xpu.is_available() else 'cpu'
    checkpoint_path = 'selected_models/small_model_wo_skip/best_model.pth'  # Change to 'latest_model.pth' if needed
    test_dataset_path = 'datasets/input_shakespeare.txt'  # Set to path of test dataset if available

    print(f"Using device: {device}")
    print(f"Loading model from: {checkpoint_path}\n")

    # Load model
    model, encode, decode = load_model(checkpoint_path, device)

    # Evaluate on test set if provided
    if test_dataset_path:
        print(f"\nEvaluating on test dataset: {test_dataset_path}")
        print("="*50)

        # Load test data
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_data = encode(test_text)

        # Evaluate
        test_loss = evaluate_on_test_set(model, test_data, model.block_size, device)
        print(f"Test loss: {test_loss:.4f}")
        print("="*50)

    # Generate text without prompt
    print("\n" + "="*50)
    print("Generation without prompt:")
    print("="*50)
    text = generate_text(model, encode, decode, prompt="", max_tokens=500, device=device)
    print(text)

    # Generate text with prompt
    print("\n" + "="*50)
    print("Generation with prompt:")
    print("="*50)
    prompt = "once upon a time"
    print(f"Prompt: '{prompt}'")
    text = generate_text(model, encode, decode, prompt=prompt, max_tokens=300, device=device)
    print(text)

    # Interactive mode
    print("\n" + "="*50)
    print("Interactive mode (type 'quit' to exit):")
    print("="*50)

    while True:
        user_prompt = input("\nEnter prompt: ")
        if user_prompt.lower() == 'quit':
            break

        try:
            tokens = int(input("Max tokens (default 200): ") or "200")
        except:
            tokens = 200

        generated = generate_text(model, encode, decode, prompt=user_prompt, max_tokens=tokens, device=device)
        print("\nGenerated text:")
        print(generated)