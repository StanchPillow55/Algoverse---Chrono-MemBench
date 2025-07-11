#!/usr/bin/env python3
"""
Command Line Interface for chrono-membench training.
Provides easy access to training different models.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from chrono.train import main as train_main


def train_command(args):
    """Handle the train command."""
    # Override sys.argv to pass arguments to train_main
    original_argv = sys.argv
    sys.argv = [
        'train.py',
        '--config', args.config,
        '--output_dir', args.output_dir,
        '--base_path', args.base_path or ''
    ]
    
    try:
        train_main()
    finally:
        sys.argv = original_argv


def list_configs(args):
    """List available configurations."""
    configs_dir = Path('configs')
    
    if not configs_dir.exists():
        print("No configs directory found.")
        return
    
    print("Available configurations:")
    print("=" * 40)
    
    for config_file in configs_dir.glob('*.yaml'):
        if config_file.stem != 'training_base':
            print(f"• {config_file.stem}")
            print(f"  Path: {config_file}")
            print()


def quick_train(args):
    """Quick training with predefined settings."""
    model_type = args.model_type
    config_file = f"configs/{model_type}.yaml"
    
    if not Path(config_file).exists():
        print(f"Configuration file not found: {config_file}")
        print("Available models: gemma-2b, llama-3-8b, llava-1.6-7b")
        return
    
    print(f"🚀 Starting training for {model_type}")
    print(f"📝 Using config: {config_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print()
    
    # Call train command
    train_args = argparse.Namespace(
        config=config_file,
        output_dir=args.output_dir,
        base_path=args.base_path
    )
    
    train_command(train_args)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chrono-membench: Train language models with ease",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with Gemma-2B
  python src/cli.py quick gemma-2b
  
  # Training with custom config
  python src/cli.py train --config configs/gemma_2b.yaml
  
  # List available configurations
  python src/cli.py list-configs
  
  # Training with custom output directory
  python src/cli.py quick llama-3-8b --output_dir ./my_models
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Quick training command
    quick_parser = subparsers.add_parser(
        'quick',
        help='Quick training with predefined configurations'
    )
    quick_parser.add_argument(
        'model_type',
        choices=['gemma-2b', 'llama-3-8b', 'llava-1.6-7b'],
        help='Model type to train'
    )
    quick_parser.add_argument(
        '--output_dir',
        default='outputs',
        help='Output directory for trained model'
    )
    quick_parser.add_argument(
        '--base_path',
        default='',
        help='Base path for data files'
    )
    quick_parser.set_defaults(func=quick_train)
    
    # Full training command
    train_parser = subparsers.add_parser(
        'train',
        help='Train with custom configuration'
    )
    train_parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration file'
    )
    train_parser.add_argument(
        '--output_dir',
        default='outputs',
        help='Output directory for trained model'
    )
    train_parser.add_argument(
        '--base_path',
        default='',
        help='Base path for data files'
    )
    train_parser.set_defaults(func=train_command)
    
    # List configurations command
    list_parser = subparsers.add_parser(
        'list-configs',
        help='List available configurations'
    )
    list_parser.set_defaults(func=list_configs)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the chosen command
    args.func(args)


if __name__ == '__main__':
    main()
