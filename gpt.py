import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import os

# hyperparameters
batch_size = 128
block_size = 256
max_iters = 5000
eval_interval = 10
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embd = 120
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('datasets/input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = set(text)
# Need to add all special characters that may be missing in the dataset!
# Add all vocal from test set, and shakespeare dataset to ensure coverage
with open('datasets/input_childSpeech_testSet.txt', 'r', encoding='utf-8') as f:
    testtext = f.read()
testtext = set(text)

with open('datasets/input_shakespeare.txt', 'r', encoding='utf-8') as f:
    shaketext = f.read()
shaketext = set(shaketext)

#Make a union of all characters
chars = chars.union(testtext)
chars = chars.union(shaketext)  
chars = sorted(list(chars))

vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.95*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    if split == 'train':
        # Combine strategies: higher masking rate (25%) + BERT 80-10-10 + span masking
        mask_prob = 0.15  # Increased from 0.08, research shows 15-40% works well
        mean_span_length = 3  # Mask spans instead of individual tokens
        
        # Create span masks
        mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        
        for batch_idx in range(x.shape[0]):
            num_tokens_to_mask = int(mask_prob * block_size)
            masked_count = 0
            
            while masked_count < num_tokens_to_mask:
                # Random starting position
                start_pos = torch.randint(0, block_size, (1,)).item()
                # Random span length (1-5 tokens, avg ~3)
                span_len = min(
                    torch.randint(1, mean_span_length * 2, (1,)).item(),
                    block_size - start_pos,
                    num_tokens_to_mask - masked_count
                )
                
                end_pos = start_pos + span_len
                mask[batch_idx, start_pos:end_pos] = True
                masked_count += span_len
        
        # Apply BERT's 80-10-10 rule on masked positions
        if mask.any():
            rand_uniform = torch.rand(x.shape, device=x.device)
            
            # 80%: replace with random token
            replace_mask = mask & (rand_uniform < 0.8)
            rand_tokens = torch.randint(0, vocab_size, x.shape, dtype=torch.long, device=x.device)
            x = torch.where(replace_mask, rand_tokens, x)
            
            # 10%: replace with different random token (double randomization)
            swap_mask = mask & (rand_uniform >= 0.8) & (rand_uniform < 0.9)
            swap_tokens = torch.randint(0, vocab_size, x.shape, dtype=torch.long, device=x.device)
            x = torch.where(swap_mask, swap_tokens, x)
            
            # 10%: keep original (no change needed - helps model learn unmasked distribution)
    
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# Create checkpoint directory
os.makedirs('models/checkpoints_v9', exist_ok=True)

writer = SummaryWriter(log_dir="runs/childSpeech_experiment_v9")

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape #  B=batch, T=sequence length, C=embedding dimension
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
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)

    def forward(self, idx, targets=None):
        B, T = idx.shape
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize best loss tracker
best_val_loss = float('inf')

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_loss = losses['train'].item()
        val_loss = losses['val'].item()
        
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {
            'Train': train_loss,
            'Val': val_loss
        }, iter)
        
        writer.add_scalars('Perplexity', {
            'Train': calculate_perplexity(train_loss),
            'Val': calculate_perplexity(val_loss)
        }, iter)
        
        writer.add_scalar('Metrics/Overfitting_Gap', val_loss - train_loss, iter)
        writer.flush()
        # Save latest checkpoint
        checkpoint = {
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'stoi': stoi,
            'itos': itos
        }
        torch.save(checkpoint, 'models/checkpoints_v9/latest_model.pth')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, 'models/checkpoints_v9/best_model.pth')
            print(f"✓ Best model saved! Val loss: {val_loss:.4f}")

    # Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    grad_norm = get_grad_norm(model)
    writer.add_scalar('Gradients/Norm', grad_norm, iter)
    
    optimizer.step()
    
    writer.add_scalar('Hyperparameters/Learning_Rate', optimizer.param_groups[0]['lr'], iter)
    
    if iter % 50 == 0:
        writer.add_scalar('Metrics/Batch_Loss', loss.item(), iter)
        writer.flush()      

# Generate sample text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n" + "="*50)
print("Sample generation from trained model:")
print("="*50)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

writer.close()
print("\nTraining complete! Models saved in 'checkpoints_v9/' directory")
