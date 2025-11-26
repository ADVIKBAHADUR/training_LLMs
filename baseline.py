import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Load your data
with open('datasets/input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

# Encode dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Hyperparameters (match your training script)
batch_size = 128
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data):
    """Get random batch from data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def calculate_random_baseline_loss(num_batches=200):
    """Calculate loss for random predictions"""
    losses = []
    
    for _ in range(num_batches):
        x, y = get_batch(data)
        
        # Random predictions: uniform distribution over vocabulary
        # Shape: (batch_size, block_size, vocab_size)
        random_logits = torch.randn(batch_size, block_size, vocab_size, device=device)
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(random_logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    
    return sum(losses) / len(losses)

# Calculate baseline loss
print("Calculating random baseline loss...")
baseline_loss = calculate_random_baseline_loss()
print(f"Random Baseline Loss: {baseline_loss:.4f}")

# Log to TensorBoard as a horizontal line
writer = SummaryWriter('runs/baseline')  # Use same directory as your training

# Log baseline at multiple steps to create a line
for step in range(0, 5001, 10):
    writer.add_scalar('Loss', baseline_loss, step)

writer.close()

print(f"\n✓ Logged baseline to TensorBoard!")
print(f"  Baseline loss: {baseline_loss:.4f}")
print(f"  Your model should achieve loss BELOW this line!")
print(f"\nRun: tensorboard --logdir=runs/gpt_experiment")
