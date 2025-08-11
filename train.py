import torch
import torch.nn as nn
import math
import time
from config import CONFIG
from data import get_data_loaders
from model import TinyShakespeareTransformer

# Try to import rich, fall back to simple printing if not available
try:
    from rich.live import Live
    from tui_monitor import TrainingMonitor
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class TrainingMonitor:
        def __init__(self):
            self.training_data = []
            self.validation_data = []
            self.current_epoch = 0
            self.current_step = 0
            self.best_val_loss = float('inf')
        
        def update_training(self, step, loss, lr):
            pass
        
        def update_validation(self, epoch, val_loss):
            pass
            
        def display(self):
            return None

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

    # Initialize monitor
    monitor = TrainingMonitor()
    
    train_loader, val_loader, tokenizer = get_data_loaders(config)
    print(f"Vocabulary size: {config['vocab_size']}")
    
    model = TinyShakespeareTransformer(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    if config['compile'] and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
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
    
    # Use rich live display if available
    if RICH_AVAILABLE:
        with Live(monitor.display(), refresh_per_second=4) as live:
            train_loop(config, device, train_loader, val_loader, model, optimizer, scheduler, monitor, live)
    else:
        train_loop(config, device, train_loader, val_loader, model, optimizer, scheduler, monitor, None)
    
    return model, tokenizer

def train_loop(config, device, train_loader, val_loader, model, optimizer, scheduler, monitor, live_display):
    """Separate training loop function"""
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
            
            # Update monitor
            monitor.update_training(step, loss.item(), scheduler.get_last_lr()[0])
            if live_display and step % 10 == 0:  # Update every 10 steps to reduce flickering
                live_display.update(monitor.display())
            
            # Logging
            if step % config['log_interval'] == 0:
                lr = scheduler.get_last_lr()[0]
                if not RICH_AVAILABLE:  # Only print if rich is not available
                    print(f"Step {step:5d} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
        
        # Validation
        val_loss = estimate_loss(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        
        # Update monitor with validation data
        monitor.update_validation(epoch, val_loss)
        if live_display:
            live_display.update(monitor.display())
        
        # Print validation results
        print(f"Epoch {epoch:2d} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Save best checkpoint
        if config['save_checkpoint'] and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'tokenizer_char_to_idx': train_loader.dataset.tokenizer.char_to_idx,
                'tokenizer_idx_to_char': train_loader.dataset.tokenizer.idx_to_char,
                'val_loss': val_loss,
                'step': step,
            }
            torch.save(checkpoint, config['checkpoint_path'])
            print(f"Saved checkpoint with val_loss: {val_loss:.4f}")

if __name__ == "__main__":
    model, tokenizer = train()