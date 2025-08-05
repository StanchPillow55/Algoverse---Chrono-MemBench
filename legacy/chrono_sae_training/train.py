#!/usr/bin/env python3
"""
Chrono-SAE Training Script

Command-line interface for training the Chrono-SAE model with temporal dropout
and memory feature analysis capabilities.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

from .model import ChronoSAE, ChronoSAEConfig, create_chrono_sae

app = typer.Typer(help="Chrono-SAE Training CLI")
console = Console()


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup rich logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    return logging.getLogger("chrono_sae")


class ChronoSAETrainer:
    """Trainer class for ChronoSAE model."""
    
    def __init__(self, 
                 model: ChronoSAE,
                 optimizer: torch.optim.Optimizer,
                 device: str,
                 use_amp: bool = False,
                 checkpoint_dir: Optional[Path] = None,
                 log_dir: Optional[Path] = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Mixed precision setup
        if use_amp:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
            
        # Tensorboard writer
        if log_dir:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            
        self.step = 0
        self.epoch = 0
        
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        if self.use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(batch, compute_loss=True)
            loss = outputs['loss']
        else:
            outputs = self.model(batch, compute_loss=True)
            loss = outputs['loss']
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        # Update decoder bias periodically
        if self.step % 100 == 0:
            self.model.update_decoder_bias(batch)
        
        # Collect metrics
        with torch.no_grad():
            loss_components = outputs['loss_components']
            sparsity_metrics = self.model.get_sparsity_metrics(outputs['z'])
            
            metrics = {
                'loss/total': loss.item(),
                'loss/mse': loss_components['mse_loss'].item(),
                'loss/l1': loss_components['l1_loss'].item(),
                'loss/tpg': loss_components['tpg_loss'].item(),
                **{f'sparsity/{k}': v for k, v in sparsity_metrics.items()}
            }
        
        self.step += 1
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = self.model(batch, compute_loss=True)
                else:
                    outputs = self.model(batch, compute_loss=True)
                
                # Accumulate metrics
                loss_components = outputs['loss_components']
                sparsity_metrics = self.model.get_sparsity_metrics(outputs['z'])
                
                batch_metrics = {
                    'val_loss/total': outputs['loss'].item(),
                    'val_loss/mse': loss_components['mse_loss'].item(),
                    'val_loss/l1': loss_components['l1_loss'].item(),
                    'val_loss/tpg': loss_components['tpg_loss'].item(),
                    **{f'val_sparsity/{k}': v for k, v in sparsity_metrics.items()}
                }
                
                for key, value in batch_metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0) + value
                
                num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to tensorboard and console."""
        if step is None:
            step = self.step
            
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def save_checkpoint(self, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
            
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.model.config),
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        if metrics:
            checkpoint['metrics'] = metrics
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        console.log(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        console.log(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        console.log(f"✅ Checkpoint loaded (epoch {self.epoch}, step {self.step})")


def create_dummy_dataset(num_samples: int = 1000, 
                        d_model: int = 768, 
                        seq_len: int = 32) -> TensorDataset:
    """Create dummy dataset for testing/demonstration."""
    # Generate random activations that simulate transformer hidden states
    activations = torch.randn(num_samples, seq_len, d_model)
    
    # Add some structure to make it more realistic
    # Simulate attention patterns
    for i in range(num_samples):
        # Random attention weights
        attn_weights = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
        activations[i] = torch.matmul(attn_weights, activations[i])
    
    return TensorDataset(activations)


@app.command()
def train(
    checkpoint_dir: Path = typer.Option("./outputs/checkpoints", help="Directory to save checkpoints"),
    log_dir: Path = typer.Option("./outputs/logs", help="Directory for tensorboard logs"),
    epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    d_model: int = typer.Option(768, help="Model dimension"),
    d_sae: int = typer.Option(2048, help="SAE hidden dimension"),
    temporal_dropout_p: float = typer.Option(0.1, help="Temporal dropout probability"),
    lambda_sparsity: float = typer.Option(1e-4, help="L1 sparsity coefficient"),
    beta_tpg: float = typer.Option(1e-3, help="TPG loss coefficient"),
    val_split: float = typer.Option(0.2, help="Validation split"),
    val_interval: int = typer.Option(5, help="Validation interval (epochs)"),
    save_interval: int = typer.Option(5, help="Checkpoint save interval (epochs)"),
    resume: Optional[Path] = typer.Option(None, help="Resume from checkpoint"),
    dummy_data: bool = typer.Option(True, help="Use dummy dataset for testing"),
    num_samples: int = typer.Option(1000, help="Number of dummy samples"),
    seq_len: int = typer.Option(32, help="Sequence length for dummy data"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """Train ChronoSAE model."""
    
    # Setup logging
    logger = setup_logging(log_level)
    
    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
        use_amp = True  # Use mixed precision on GPU
        console.log("🚀 Using GPU with mixed precision")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        use_amp = False  # MPS doesn't support AMP yet
        console.log("🍎 Using Apple Silicon MPS")
    else:
        device = "cpu"
        use_amp = False
        console.log("💻 Using CPU")
    
    # Model configuration
    config = ChronoSAEConfig(
        d_model=d_model,
        d_sae=d_sae,
        temporal_dropout_p=temporal_dropout_p,
        lambda_sparsity=lambda_sparsity,
        beta_tpg=beta_tpg,
        device=device,
    )
    
    # Create model
    model = create_chrono_sae(config)
    console.log(f"🧠 Created ChronoSAE model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer
    trainer = ChronoSAETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )
    
    # Resume from checkpoint if specified
    if resume:
        trainer.load_checkpoint(resume)
    
    # Create dataset
    if dummy_data:
        console.log(f"📊 Creating dummy dataset: {num_samples} samples, seq_len={seq_len}")
        dataset = create_dummy_dataset(num_samples, d_model, seq_len)
    else:
        raise NotImplementedError("Real data loading not implemented yet")
    
    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    console.log(f"📈 Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Training loop
    console.log("🚀 Starting training...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("Training", total=epochs)
        
        for epoch in range(trainer.epoch, epochs):
            trainer.epoch = epoch
            
            # Training phase
            epoch_metrics = {}
            num_batches = 0
            
            for batch_idx, (batch,) in enumerate(train_loader):
                metrics = trainer.train_step(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key] = epoch_metrics.get(key, 0) + value
                num_batches += 1
                
                # Log batch metrics occasionally
                if batch_idx % 10 == 0:
                    trainer.log_metrics(metrics)
            
            # Average epoch metrics
            avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
            
            # Validation
            if epoch % val_interval == 0:
                val_metrics = trainer.validate(val_loader)
                avg_metrics.update(val_metrics)
            
            # Log epoch metrics
            trainer.log_metrics(avg_metrics, step=epoch)
            
            # Console output
            progress.update(task, advance=1)
            if epoch % 1 == 0:  # Log every epoch
                table = Table(title=f"Epoch {epoch}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                for key, value in avg_metrics.items():
                    if 'loss' in key:
                        table.add_row(key, f"{value:.6f}")
                    else:
                        table.add_row(key, f"{value:.4f}")
                
                console.print(table)
            
            # Save checkpoint
            if epoch % save_interval == 0:
                trainer.save_checkpoint(epoch, avg_metrics)
    
    # Final checkpoint
    trainer.save_checkpoint(epochs, avg_metrics)
    
    if trainer.writer:
        trainer.writer.close()
    
    console.log("✅ Training completed!")


@app.command()
def evaluate(
    checkpoint_path: Path = typer.Argument(..., help="Path to model checkpoint"),
    output_path: Path = typer.Option("./outputs/evaluation.json", help="Output path for results"),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    num_samples: int = typer.Option(100, help="Number of samples for evaluation"),
):
    """Evaluate trained ChronoSAE model."""
    console.log(f"📊 Evaluating model: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint['config']
    
    # Recreate config
    config = ChronoSAEConfig(**config_dict)
    
    # Create model
    model = create_chrono_sae(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create evaluation dataset  
    dataset = create_dummy_dataset(num_samples, config.d_model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluation
    total_metrics = {}
    num_batches = 0
    
    console.log("🔍 Running evaluation...")
    
    with torch.no_grad():
        for batch, in loader:
            batch = batch.to(config.device)
            outputs = model(batch, compute_loss=True)
            
            # Collect metrics
            loss_components = outputs['loss_components']
            sparsity_metrics = model.get_sparsity_metrics(outputs['z'])
            
            batch_metrics = {
                'loss/total': outputs['loss'].item(),
                'loss/mse': loss_components['mse_loss'].item(),
                'loss/l1': loss_components['l1_loss'].item(),
                'loss/tpg': loss_components['tpg_loss'].item(),
                **{f'sparsity/{k}': v for k, v in sparsity_metrics.items()}
            }
            
            for key, value in batch_metrics.items():
                total_metrics[key] = total_metrics.get(key, 0) + value
            
            num_batches += 1
    
    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in avg_metrics.items():
        if 'loss' in key:
            table.add_row(key, f"{value:.6f}")
        else:
            table.add_row(key, f"{value:.4f}")
    
    console.print(table)
    console.log(f"💾 Results saved to: {output_path}")


if __name__ == "__main__":
    app()
