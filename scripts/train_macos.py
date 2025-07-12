#!/usr/bin/env python3
"""
macOS-specific training script for chrono-membench
Optimized for Apple Silicon (M1/M2/M3) with MPS support
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chrono.chrono_train import ChronoMemBenchTrainer, ChronoConfig


def check_macos_setup():
    """Check macOS setup for training."""
    print("🍎 macOS Training Setup Check")
    print("=" * 40)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✅ MPS is available and built")
        print(f"MPS device: {torch.device('mps')}")
    else:
        print("❌ MPS is not available")
        print("Please install PyTorch with MPS support:")
        print("pip install torch torchvision torchaudio")
        return False
    
    # Check system info
    import platform
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    
    # Check memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / (1024**3):.1f} GB")
    
    return True


def optimize_for_macos(config_path: str):
    """Optimize configuration for macOS."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Optimize for macOS/MPS
    config['environment']['pin_memory'] = False
    config['environment']['device'] = 'mps'
    config['environment']['mixed_precision'] = 'fp16'
    
    # Conservative settings for local training
    config['training']['batch_size'] = 2
    config['training']['gradient_accumulation_steps'] = 8
    config['training']['max_steps'] = 1000  # Reasonable for local testing
    
    # Optimize data loading for macOS
    config['environment']['dataloader_num_workers'] = 2  # Conservative for macOS
    
    # Save optimized config
    macos_config_path = config_path.replace('.yaml', '_macos.yaml')
    with open(macos_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Optimized config saved to: {macos_config_path}")
    return macos_config_path


def main():
    """Main training function for macOS."""
    parser = argparse.ArgumentParser(description="macOS Chrono-MemBench Training")
    parser.add_argument("--config", type=str, default="configs/chrono_membench.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs/chrono_macos", 
                       help="Output directory")
    parser.add_argument("--check_only", action="store_true", 
                       help="Only check setup, don't train")
    
    args = parser.parse_args()
    
    # Check macOS setup
    if not check_macos_setup():
        return
    
    if args.check_only:
        print("✅ Setup check complete!")
        return
    
    # Optimize configuration for macOS
    macos_config_path = optimize_for_macos(args.config)
    
    # Create chrono configuration
    chrono_config = ChronoConfig(
        temporal_dropout_rate=0.2,
        sae_enabled=True,
        feature_alignment_enabled=True,
        wandb_project="chrono-membench-macos",
        dashboard_enabled=True
    )
    
    # Initialize trainer
    trainer = ChronoMemBenchTrainer(
        config_path=macos_config_path,
        output_dir=args.output_dir,
        chrono_config=chrono_config
    )
    
    # Start training
    print("🚀 Starting macOS training...")
    trainer.train()
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()
