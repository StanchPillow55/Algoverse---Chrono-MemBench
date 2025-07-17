#!/usr/bin/env python3
"""
Chrono-MemBench Training Script
Enhanced training script with temporal dropout regularization and Route-SAE integration.
Based on the existing training infrastructure but designed for the chrono-membench framework.
"""

import os
import sys
import yaml
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import argparse
import time
import wandb
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

# Import from existing modules
from chrono.data_loader import DatasetConfig, create_data_loaders
from chrono.train import ModelManager, TrainingManager, load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChronoConfig:
    """Configuration for chrono-membench specific settings."""
    # Temporal dropout settings
    temporal_dropout_rate: float = 0.2
    temporal_dropout_schedule: str = "linear"  # linear, cosine, constant
    
    # SAE settings
    sae_enabled: bool = True
    sae_latent_dim: int = 4096
    sae_checkpoint_interval: int = 150_000_000  # 150M tokens
    
    # Feature alignment settings
    feature_alignment_enabled: bool = True
    feature_alignment_weight: float = 0.1
    
    # Checkpointing
    checkpoint_every_n_tokens: int = 200_000_000  # 200M tokens
    max_checkpoints: int = 20
    
    # Metrics
    compute_temporal_purity: bool = True
    compute_feature_birth_curves: bool = True
    
    # Monitoring
    wandb_project: str = "chrono-membench"
    dashboard_enabled: bool = True
    dashboard_update_interval: int = 10000  # steps


class TemporalDropoutScheduler:
    """Scheduler for temporal dropout rate during training."""
    
    def __init__(self, config: ChronoConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        self.initial_rate = config.temporal_dropout_rate
        
    def get_dropout_rate(self, step: int) -> float:
        """Get dropout rate for current step."""
        if self.config.temporal_dropout_schedule == "constant":
            return self.initial_rate
        elif self.config.temporal_dropout_schedule == "linear":
            # Linear decay from initial_rate to 0
            return self.initial_rate * (1.0 - step / self.total_steps)
        elif self.config.temporal_dropout_schedule == "cosine":
            # Cosine decay
            return self.initial_rate * 0.5 * (1 + np.cos(np.pi * step / self.total_steps))
        else:
            return self.initial_rate


class RouteDAE(nn.Module):
    """Route-SAE (Sparse Autoencoder) for feature extraction."""
    
    def __init__(self, input_dim: int, latent_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through SAE."""
        # Encode
        z = self.encoder(x)
        
        # Decode
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed, z
    
    def get_sparsity_loss(self, z: torch.Tensor, sparsity_weight: float = 0.1) -> torch.Tensor:
        """Compute sparsity loss for latent representations."""
        # L1 sparsity loss
        return sparsity_weight * torch.mean(torch.abs(z))


class ChronoTrainer(Trainer):
    """Enhanced Trainer with chrono-membench functionality."""
    
    def __init__(self, chrono_config: ChronoConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chrono_config = chrono_config
        self.temporal_dropout_scheduler = TemporalDropoutScheduler(
            chrono_config, self.args.max_steps
        )
        
        # Initialize SAE if enabled
        if chrono_config.sae_enabled:
            # Get hidden size from model
            hidden_size = self.model.config.hidden_size
            self.sae = RouteDAE(
                input_dim=hidden_size,
                latent_dim=chrono_config.sae_latent_dim,
                dropout_rate=chrono_config.temporal_dropout_rate
            )
            self.sae.to(self.model.device)
            
            # Add SAE parameters to optimizer
            self.sae_optimizer = torch.optim.AdamW(
                self.sae.parameters(),
                lr=self.args.learning_rate * 0.1  # Lower learning rate for SAE
            )
        
        # Initialize metrics tracking
        self.temporal_purity_scores = []
        self.feature_birth_events = []
        self.checkpoint_features = {}
        
        # Initialize dashboard if enabled
        if chrono_config.dashboard_enabled:
            self.setup_dashboard()
    
    def setup_dashboard(self):
        """Setup monitoring dashboard."""
        if self.chrono_config.wandb_project:
            wandb.init(
                project=self.chrono_config.wandb_project,
                config=self.chrono_config.__dict__
            )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Enhanced loss computation with SAE and feature alignment."""
        # Standard language modeling loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add SAE loss if enabled
        if self.chrono_config.sae_enabled and hasattr(self, 'sae'):
            # Extract hidden states (assuming last layer)
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
            
            if hidden_states is not None:
                # Reshape for SAE
                batch_size, seq_len, hidden_size = hidden_states.shape
                hidden_flat = hidden_states.view(-1, hidden_size)
                
                # Pass through SAE
                reconstructed, latent = self.sae(hidden_flat)
                
                # Reconstruction loss
                reconstruction_loss = F.mse_loss(reconstructed, hidden_flat)
                
                # Sparsity loss
                sparsity_loss = self.sae.get_sparsity_loss(latent)
                
                # Add to total loss
                sae_loss = reconstruction_loss + sparsity_loss
                loss += 0.1 * sae_loss  # Weight SAE loss
                
                # Log SAE metrics
                if self.state.global_step % 100 == 0:
                    self.log({
                        "sae_reconstruction_loss": reconstruction_loss.item(),
                        "sae_sparsity_loss": sparsity_loss.item(),
                        "sae_total_loss": sae_loss.item()
                    })
        
        # Feature alignment loss
        if self.chrono_config.feature_alignment_enabled:
            alignment_loss = self.compute_feature_alignment_loss(outputs)
            loss += self.chrono_config.feature_alignment_weight * alignment_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_feature_alignment_loss(self, outputs) -> torch.Tensor:
        """Compute feature alignment loss to maintain interpretability."""
        # Simple L2 regularization on attention weights
        # More sophisticated alignment can be added here
        total_loss = 0.0
        
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            for attention in outputs.attentions:
                # Encourage attention diversity
                attention_mean = attention.mean(dim=-1, keepdim=True)
                diversity_loss = F.mse_loss(attention, attention_mean)
                total_loss += diversity_loss
        
        return total_loss
    
    def training_step(self, model, inputs):
        """Enhanced training step with temporal dropout."""
        # Update temporal dropout rate
        current_step = self.state.global_step
        dropout_rate = self.temporal_dropout_scheduler.get_dropout_rate(current_step)
        
        # Apply temporal dropout to SAE if enabled
        if self.chrono_config.sae_enabled and hasattr(self, 'sae'):
            self.sae.dropout_rate = dropout_rate
            # Update dropout layers
            for module in self.sae.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate
        
        # Standard training step
        return super().training_step(model, inputs)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with chrono-membench metrics."""
        # Standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Compute temporal purity if enabled
        if self.chrono_config.compute_temporal_purity:
            purity_score = self.compute_temporal_purity()
            eval_results[f"{metric_key_prefix}_temporal_purity"] = purity_score
            self.temporal_purity_scores.append(purity_score)
        
        # Log to dashboard
        if self.chrono_config.dashboard_enabled:
            wandb.log(eval_results)
        
        return eval_results
    
    def compute_temporal_purity(self) -> float:
        """Compute temporal purity metric."""
        # Simplified temporal purity computation
        # In practice, this would analyze feature consistency across checkpoints
        if not hasattr(self, 'sae') or not self.chrono_config.sae_enabled:
            return 0.0
        
        # Extract current feature representations
        current_features = self.extract_features()
        
        # Compare with previous checkpoint if available
        if self.checkpoint_features:
            previous_features = list(self.checkpoint_features.values())[-1]
            # Compute cosine similarity between feature sets
            similarity = F.cosine_similarity(
                current_features.mean(dim=0, keepdim=True),
                previous_features.mean(dim=0, keepdim=True)
            )
            purity = similarity.item()
        else:
            purity = 0.0
        
        return purity
    
    def extract_features(self) -> torch.Tensor:
        """Extract current feature representations."""
        # Simplified feature extraction
        # In practice, this would run inference on a standard dataset
        if hasattr(self, 'sae'):
            # Get random sample from model
            dummy_input = torch.randn(1, 512, self.model.config.hidden_size).to(self.model.device)
            _, features = self.sae(dummy_input.view(-1, self.model.config.hidden_size))
            return features
        else:
            return torch.tensor([0.0])
    
    def save_checkpoint(self, checkpoint_dir: str, step: int):
        """Save checkpoint with chrono-membench state."""
        # Standard checkpoint saving
        super().save_model(checkpoint_dir)
        
        # Save SAE if enabled
        if self.chrono_config.sae_enabled and hasattr(self, 'sae'):
            sae_path = os.path.join(checkpoint_dir, f"sae_step_{step}.pt")
            torch.save(self.sae.state_dict(), sae_path)
        
        # Save features for temporal analysis
        current_features = self.extract_features()
        self.checkpoint_features[step] = current_features.cpu()
        
        # Save temporal metrics
        metrics_path = os.path.join(checkpoint_dir, f"temporal_metrics_{step}.json")
        metrics = {
            "temporal_purity_scores": self.temporal_purity_scores,
            "feature_birth_events": self.feature_birth_events,
            "step": step
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)


class ChronoMemBenchTrainer:
    """Main trainer class for chrono-membench framework."""
    
    def __init__(self, config_path: str, output_dir: str, chrono_config: ChronoConfig):
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.chrono_config = chrono_config
        
        # Initialize model manager
        self.model_manager = ModelManager(self.config)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for chrono-membench."""
        log_file = self.output_dir / "chrono_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train(self):
        """Execute chrono-membench training."""
        logger.info("Starting chrono-membench training...")
        
        # Load model and tokenizer
        model, tokenizer = self.model_manager.load_model_and_tokenizer()
        model = self.model_manager.setup_lora(model, self.config['training'])
        
        # Create dataset configuration
        dataset_config = DatasetConfig(
            sources=self.config['dataset']['sources'],
            mixing_ratios=self.config['dataset']['mixing_ratios'],
            max_length=self.config['dataset']['max_length'],
            train_split=self.config['dataset']['train_split'],
            val_split=self.config['dataset']['val_split'],
            shuffle=self.config['dataset']['shuffle']
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            dataset_config,
            tokenizer,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['environment']['dataloader_num_workers']
        )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=1,
            max_steps=self.config['training']['max_steps'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=50,
            eval_steps=self.config['training']['eval_steps'],
            save_steps=self.config['training']['save_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.chrono_config.max_checkpoints,
            fp16=self.config['environment']['mixed_precision'] == 'fp16',
            dataloader_num_workers=self.config['environment']['dataloader_num_workers'],
            remove_unused_columns=False,
            report_to="wandb" if self.chrono_config.dashboard_enabled else None,
        )
        
        # Create chrono trainer
        trainer = ChronoTrainer(
            chrono_config=self.chrono_config,
            model=model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("Starting training loop...")
        trainer.train()
        
        # Save final model and metrics
        final_checkpoint = self.output_dir / "final_checkpoint"
        trainer.save_checkpoint(str(final_checkpoint), trainer.state.global_step)
        
        logger.info("Training completed successfully!")
        
        # Generate evaluation report
        self.generate_evaluation_report(trainer)
    
    def generate_evaluation_report(self, trainer: ChronoTrainer):
        """Generate final evaluation report."""
        report = {
            "experiment_info": {
                "model_type": self.config['model']['type'],
                "total_steps": trainer.state.global_step,
                "training_time": time.time() - trainer.state.train_batch_size,
                "datasets_used": list(self.config['dataset']['mixing_ratios'].keys()),
                "dataset_mixing_ratios": self.config['dataset']['mixing_ratios'],
            },
            "chrono_metrics": {
                "temporal_purity_scores": trainer.temporal_purity_scores,
                "feature_birth_events": trainer.feature_birth_events,
                "final_temporal_purity": trainer.temporal_purity_scores[-1] if trainer.temporal_purity_scores else 0.0,
            },
            "training_config": self.chrono_config.__dict__,
        }
        
        # Save report
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")


def main():
    """Main entry point for chrono-membench training."""
    parser = argparse.ArgumentParser(description="Chrono-MemBench Training")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs/chrono", help="Output directory")
    parser.add_argument("--temporal_dropout_rate", type=float, default=0.2, help="Temporal dropout rate")
    parser.add_argument("--sae_enabled", action="store_true", help="Enable SAE")
    parser.add_argument("--feature_alignment_enabled", action="store_true", help="Enable feature alignment")
    parser.add_argument("--wandb_project", type=str, default="chrono-membench", help="WandB project name")
    
    args = parser.parse_args()
    
    # Create chrono configuration
    chrono_config = ChronoConfig(
        temporal_dropout_rate=args.temporal_dropout_rate,
        sae_enabled=args.sae_enabled,
        feature_alignment_enabled=args.feature_alignment_enabled,
        wandb_project=args.wandb_project,
    )
    
    # Initialize trainer
    trainer = ChronoMemBenchTrainer(
        config_path=args.config,
        output_dir=args.output_dir,
        chrono_config=chrono_config
    )
    
    # Set random seed
    set_seed(42)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
