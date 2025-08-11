from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from collections import deque
import time

class TrainingMonitor:
    def __init__(self):
        self.console = Console()
        self.training_losses = deque(maxlen=100)
        self.validation_losses = deque(maxlen=50)
        self.learning_rates = deque(maxlen=100)
        self.steps = deque(maxlen=100)
        self.epochs = deque(maxlen=50)
        self.val_losses_by_epoch = deque(maxlen=50)
        
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.current_loss = 0
        self.current_lr = 0
        
        # Training stats
        self.start_time = time.time()
        self.steps_per_second = 0
        self.last_step_time = time.time()
        
    def update_training(self, step, loss, lr):
        self.training_losses.append(loss)
        self.learning_rates.append(lr)
        self.steps.append(step)
        self.current_step = step
        self.current_loss = loss
        self.current_lr = lr
        
        # Calculate steps per second
        current_time = time.time()
        if len(self.steps) > 1:
            time_diff = current_time - self.last_step_time
            if time_diff > 0:
                self.steps_per_second = 1.0 / time_diff
        self.last_step_time = current_time
        
    def update_validation(self, epoch, val_loss):
        self.validation_losses.append(val_loss)
        self.epochs.append(epoch)
        self.val_losses_by_epoch.append(val_loss)
        self.current_epoch = epoch
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
    def create_simple_trend(self, values, width=20):
        """Create a simple ASCII trend representation"""
        if len(values) < 2:
            return "..."
        
        # Take last few values
        recent = list(values)[-min(len(values), width//2):]
        if len(recent) < 2:
            return "..."
        
        # Create simple up/down trend
        trend_chars = []
        for i in range(1, len(recent)):
            diff = recent[i] - recent[i-1]
            if abs(diff) < 1e-6:
                trend_chars.append("─")
            elif diff < 0:
                trend_chars.append("↘")
            else:
                trend_chars.append("↗")
        
        return "".join(trend_chars[-width:]) if trend_chars else "..."
        
    def get_training_table(self):
        table = Table(title=" Training Metrics", border_style="blue")
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="green", width=20)
        table.add_column("Trend", style="yellow", width=25)
        
        # Average loss
        if self.training_losses:
            avg_loss = sum(self.training_losses) / len(self.training_losses)
            trend = self.create_simple_trend(self.training_losses)
            table.add_row("Avg Loss", f"{avg_loss:.4f}", trend)
        
        # Current loss
        table.add_row("Current Loss", f"{self.current_loss:.4f}", "")
        
        # Learning rate
        trend = self.create_simple_trend(self.learning_rates)
        table.add_row("Learning Rate", f"{self.current_lr:.2e}", trend)
        
        # Steps per second
        if self.steps_per_second > 0:
            table.add_row("Speed", f"{self.steps_per_second:.1f} steps/sec", "")
            
        return table
        
    def get_validation_table(self):
        table = Table(title=" Validation Metrics", border_style="magenta")
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="green", width=20)
        table.add_column("Trend", style="yellow", width=25)
        
        # Best validation loss
        table.add_row("Best Val Loss", f"{self.best_val_loss:.4f}", "")
        
        # Current validation loss
        if self.validation_losses:
            current_val = self.validation_losses[-1]
            trend = self.create_simple_trend(self.val_losses_by_epoch)
            table.add_row("Current Val", f"{current_val:.4f}", trend)
            
        # Validation improvement
        if len(self.validation_losses) > 1:
            prev_val = self.validation_losses[-2]
            current_val = self.validation_losses[-1]
            improvement = prev_val - current_val
            imp_text = f"{improvement:+.4f}"
            imp_style = "green" if improvement > 0 else "red"
            table.add_row("Improvement", f"[{imp_style}]{imp_text}[/{imp_style}]", "")
            
        return table
        
    def get_status_panel(self):
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"
        
        status_text = Text()
        status_text.append("RoPen Transformer Training\n", style="bold yellow")
        status_text.append(f"Epoch: {self.current_epoch} | ", style="bold")
        status_text.append(f"Step: {self.current_step:,} | ", style="bold")
        status_text.append(f"Time: {elapsed_str}\n", style="bold")
        status_text.append(f"Best Val Loss: {self.best_val_loss:.4f} | ", style="bold green")
        status_text.append(f"Current Loss: {self.current_loss:.4f}", style="bold blue")
        
        return Panel(status_text, title="Status", border_style="green", title_align="left")
        
    def get_progress_info(self):
        info_text = Text()
        info_text.append("Training Progress\n", style="bold cyan")
        
        if self.training_losses:
            recent_losses = list(self.training_losses)[-10:]
            if len(recent_losses) >= 2:

                trend = recent_losses[-1] - recent_losses[0]
                trend_text = "↓ Improving" if trend < 0 else "↑ Degrading" if trend > 0 else "→ Stable"
                trend_style = "green" if trend < 0 else "red" if trend > 0 else "yellow"
                info_text.append(f"Trend: [{trend_style}]{trend_text}[/{trend_style}]\n")
                
        if self.validation_losses and len(self.validation_losses) >= 2:
            val_improvement = self.validation_losses[-2] - self.validation_losses[-1]
            imp_text = f"Val Δ: {val_improvement:+.4f}"
            imp_style = "green" if val_improvement > 0 else "red"
            info_text.append(f"[{imp_style}]{imp_text}[/{imp_style}]\n")
            
        return Panel(info_text, title="Info", border_style="cyan", title_align="left")
        
    def display(self):
        layout = Layout()
        
        layout.split_column(
            Layout(self.get_status_panel(), size=5, name="status"),
            Layout(name="main"),
            Layout(self.get_training_table(), size=8, name="training"),
            Layout(self.get_validation_table(), size=8, name="validation")
        )
        
        # Split main area for additional info
        layout["main"].split_row(
            Layout(self.get_progress_info(), name="info")
        )
        
        return layout

    def start_monitor(self):
        """Dummy method for compatibility"""
        pass