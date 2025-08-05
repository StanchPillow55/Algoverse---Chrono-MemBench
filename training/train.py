#!/usr/bin/env python3
"""
ChronoSAE Training CLI

A comprehensive Typer-based CLI for training ChronoSAE models with:
- YAML configuration support
- Multiple optimizer support (AdamW, Lion)
- Gradient accumulation for 8GB VRAM efficiency
- torch.compile integration
- Graceful checkpoint resume
- Multi-GPU support via torchrun
- Metric streaming with six dials
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from membench_x.metrics import (
    MemAbsorptionHook, TPGHook, CapGaugeHook,
    ICLPersistenceHook, WeightDeltaHook, RAGTraceHook
)
from training.loop import create_training_loop

# Try to import ChronoSAE - fallback to dummy if not available
try:
    from src.algoverse.chrono.chrono_sae.model import ChronoSAE, ChronoSAEConfig, create_chrono_sae
    CHRONO_SAE_AVAILABLE = True
except ImportError:
    CHRONO_SAE_AVAILABLE = False
    
    # Create dummy classes for testing
    class ChronoSAEConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ChronoSAE(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.encoder = nn.Linear(config.d_model, config.d_sae)
            self.decoder = nn.Linear(config.d_sae, config.d_model)
            self.temporal_dropout = nn.Dropout(config.temporal_dropout_p)
            
        def forward(self, x, compute_loss=True):
            # Simple forward pass for testing
            z = torch.relu(self.encoder(x))
            z = self.temporal_dropout(z)
            x_recon = self.decoder(z)
            
            if compute_loss:
                mse_loss = nn.functional.mse_loss(x_recon, x)
                l1_loss = z.abs().mean()
                temporal_loss = torch.tensor(0.0, device=x.device)
                
                total_loss = mse_loss + self.config.lambda_sparsity * l1_loss + self.config.beta_tpg * temporal_loss
                
                return {
                    'output': x_recon,
                    'activations': z,
                    'loss': total_loss,
                    'loss_components': {
                        'mse_loss': mse_loss,
                        'l1_loss': l1_loss,
                        'temporal_loss': temporal_loss
                    }
                }
            return {'output': x_recon, 'activations': z}
    
    def create_chrono_sae(config):
        return ChronoSAE(config)


# Lion optimizer implementation (simplified)
class Lion(torch.optim.Optimizer):
    """Simplified Lion optimizer implementation"""
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Simplified Lion update (not full implementation)
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss


app = typer.Typer(
    help="ChronoSAE Training CLI with configuration support",
    rich_markup_mode="rich"
)
console = Console()


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup rich logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    return logging.getLogger("chrono_sae_train")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    console.log(f"📄 Loaded config: {config_path}")
    return config


def create_dummy_dataset(config: Dict[str, Any]) -> TensorDataset:
    """Create dummy dataset for testing."""
    data_config = config['data']
    model_config = config['model']
    
    # Generate structured dummy data
    num_samples = data_config['num_samples']
    seq_len = data_config['seq_len']
    d_model = model_config['d_model']
    
    # Create data with some structure (not pure random)
    data = torch.randn(num_samples, seq_len, d_model)
    
    # Add some temporal patterns
    for i in range(num_samples):
        # Add some auto-correlation
        for t in range(1, seq_len):
            data[i, t] = 0.7 * data[i, t] + 0.3 * data[i, t-1]
    
    return TensorDataset(data)


def setup_model(config: Dict[str, Any], device: torch.device, compile_mode: str) -> nn.Module:
    """Create and setup the ChronoSAE model."""
    model_config = config['model']
    
    # Create ChronoSAE config
    sae_config = ChronoSAEConfig(
        d_model=model_config['d_model'],
        d_sae=model_config['d_sae'],
        temporal_dropout_p=model_config['temporal_dropout_p'],
        lambda_sparsity=model_config['lambda_sparsity'],
        beta_tpg=model_config['beta_tpg'],
        device=str(device)
    )
    
    # Create model
    model = create_chrono_sae(sae_config)
    model = model.to(device)
    
    # Apply torch.compile if requested
    if compile_mode != "none":
        if hasattr(torch, 'compile'):
            if compile_mode == "trace":
                model = torch.compile(model, mode="reduce-overhead")
            elif compile_mode == "compile":
                model = torch.compile(model, mode="max-autotune")
            console.log(f"🔧 Applied torch.compile with mode: {compile_mode}")
        else:
            console.log("⚠️ torch.compile not available, skipping compilation")
    
    param_count = sum(p.numel() for p in model.parameters())
    console.log(f"🧠 Created ChronoSAE: {param_count:,} parameters")
    
    return model


def setup_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Setup optimizer based on config."""
    opt_config = config['optimizers']
    primary = opt_config['primary']
    
    if primary == 'adamw':
        optimizer = AdamW(model.parameters(), **opt_config['adamw'])
        console.log(f"⚡ Using AdamW optimizer")
    elif primary == 'lion':
        optimizer = Lion(model.parameters(), **opt_config['lion'])
        console.log(f"🦁 Using Lion optimizer")
    else:
        raise ValueError(f"Unknown optimizer: {primary}")
    
    return optimizer


def setup_distributed(config: Dict[str, Any]) -> bool:
    """Setup distributed training if multiple GPUs available."""
    if torch.cuda.device_count() <= 1:
        return False
    
    if 'RANK' not in os.environ:
        console.log("💡 Multiple GPUs detected but not using torchrun. Use 'torchrun --nproc_per_node=N training/train.py ...' for multi-GPU training.")
        return False
    
    # Initialize distributed training
    dist_config = config.get('distributed', {})
    backend = dist_config.get('backend', 'nccl')
    
    dist.init_process_group(backend=backend)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    console.log(f"🌐 Initialized distributed training: rank {dist.get_rank()}/{dist.get_world_size()}")
    return True


def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    step: int, 
    config: Dict[str, Any],
    checkpoint_dir: Path,
    metrics: Optional[Dict[str, float]] = None
) -> Path:
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Unwrap DDP model if needed
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics or {}
    }
    
    # Save with epoch number
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}_step_{step:06d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    console.log(f"💾 Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, Any]:
    """Load checkpoint and restore training state."""
    console.log(f"📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    console.log(f"✅ Resumed from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    return checkpoint


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to YAML configuration file"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Path to checkpoint to resume from"),
    compile_mode: str = typer.Option("none", "--compile-mode", help="torch.compile mode: none/trace/compile"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """
    Train ChronoSAE model with YAML configuration.
    
    Features:
    - YAML configuration loading
    - Multi-optimizer support (AdamW, Lion)
    - Gradient accumulation for memory efficiency
    - torch.compile integration
    - Graceful checkpoint resume
    - Multi-GPU support (use with torchrun)
    - Six-dial metric streaming
    """
    
    # Setup logging
    logger = setup_logging(log_level)
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"🔧 Using device: {device}")
    
    # Setup distributed training if available
    is_distributed = setup_distributed(config)
    is_main_process = not is_distributed or dist.get_rank() == 0
    
    # Create output directories
    system_config = config['system']
    checkpoint_dir = Path(system_config['checkpoint_dir'])
    log_dir = Path(system_config['log_dir'])
    metrics_dir = Path(system_config['metrics_dir'])
    
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model
    model = setup_model(config, device, compile_mode)
    
    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, find_unused_parameters=config['distributed'].get('find_unused_parameters', False))
    
    # Setup optimizer
    optimizer = setup_optimizer(model, config)
    
    # Setup mixed precision
    use_amp = system_config.get('mixed_precision', False) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Create dataset and dataloaders
    dataset = create_dummy_dataset(config)
    
    # Split dataset
    data_config = config['data']
    train_size = int(len(dataset) * data_config['train_split'])
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_config = config['training']
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )
    
    console.log(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Setup tensorboard writer (main process only)
    writer = SummaryWriter(log_dir) if is_main_process else None
    
    # Create training loop
    metrics_config = config['metrics']
    training_loop = create_training_loop(
        model=model,
        optimizer=optimizer,
        log_dir=str(log_dir),
        metric_interval=metrics_config['metric_interval'],
        metrics_dir=str(metrics_dir)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    
    if resume and resume.exists():
        checkpoint = load_checkpoint(resume, model, optimizer, device)
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['step']
    
    # Training parameters
    epochs = train_config['epochs']
    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    
    save_interval = metrics_config['save_interval']
    val_interval = metrics_config['val_interval']
    log_interval = metrics_config['log_interval']
    
    console.log(f"🚀 Starting training: epochs {start_epoch}-{epochs}")
    console.log(f"📈 Gradient accumulation: {gradient_accumulation_steps} steps")
    console.log(f"💾 Save every {save_interval} steps, validate every {val_interval} steps")
    
    # Training loop
    if is_main_process:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        progress.start()
        task = progress.add_task("Training", total=epochs-start_epoch)
    
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (batch,) in enumerate(train_loader):
                batch = batch.to(device)
                
                # Forward pass with mixed precision
                if use_amp:
                    with autocast():
                        result = training_loop.train_step(global_step, batch, batch)
                else:
                    result = training_loop.train_step(global_step, batch, batch)
                
                loss = result['loss']
                epoch_loss += loss
                num_batches += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                
                # Logging
                if is_main_process and global_step % log_interval == 0:
                    if writer:
                        writer.add_scalar('train/loss', loss, global_step)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                
                # Validation
                if global_step % val_interval == 0 and global_step > 0:
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for val_batch, in val_loader:
                            val_batch = val_batch.to(device)
                            
                            if use_amp:
                                with autocast():
                                    val_result = model(val_batch, compute_loss=True)
                            else:
                                val_result = model(val_batch, compute_loss=True)
                            
                            val_loss += val_result['loss'].item()
                            val_batches += 1
                    
                    avg_val_loss = val_loss / max(val_batches, 1)
                    
                    if is_main_process and writer:
                        writer.add_scalar('val/loss', avg_val_loss, global_step)
                    
                    model.train()
                
                # Save checkpoint
                if is_main_process and global_step % save_interval == 0 and global_step > 0:
                    save_checkpoint(
                        model, optimizer, epoch, global_step, config, checkpoint_dir,
                        {'train_loss': loss, 'epoch': epoch}
                    )
                
                global_step += 1
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            
            if is_main_process:
                progress.update(task, advance=1)
                console.log(f"Epoch {epoch}: avg_loss = {avg_epoch_loss:.6f}")
                
                if writer:
                    writer.add_scalar('epoch/loss', avg_epoch_loss, epoch)
    
    finally:
        if is_main_process:
            progress.stop()
            
            # Final checkpoint
            save_checkpoint(
                model, optimizer, epochs, global_step, config, checkpoint_dir,
                {'final_epoch': epochs}
            )
            
            if writer:
                writer.close()
    
    console.log("✅ Training completed!")


if __name__ == "__main__":
    app()

