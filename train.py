import torch
import torch.nn as nn
import math
import time
from config import CONFIG
from data import get_data_loaders
from model import TinyShakespeareTransformer

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def estimate_loss(model, data_loader, device, eval_iters=50):
    """Estimate loss on dataset"""
    model.eval()
    losses = []
    
    for i, (x, y) in enumerate(data_loader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)

def train():
    """Main training loop"""
    config = CONFIG.copy()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, tokenizer = get_data_loaders(config)
    print(f"Vocabulary size: {config['vocab_size']}")
    
    # Model
    model = TinyShakespeareTransformer(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Compile model for speed (PyTorch 2.0+)
    if config['compile'] and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    max_steps = len(train_loader) * config['max_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        config['warmup_steps'], 
        max_steps
    )
    
    # Training loop
    model.train()
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config['max_epochs']):
        epoch_start = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            
            step += 1
            
            # Logging
            if step % config['log_interval'] == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
        
        # Validation
        val_loss = estimate_loss(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:2d} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Save best checkpoint
        if config['save_checkpoint'] and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'tokenizer_char_to_idx': tokenizer.char_to_idx,
                'tokenizer_idx_to_char': tokenizer.idx_to_char,
                'val_loss': val_loss,
                'step': step,
            }
            torch.save(checkpoint, config['checkpoint_path'])
            print(f"Saved checkpoint with val_loss: {val_loss:.4f}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train()