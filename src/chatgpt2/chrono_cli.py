#!/usr/bin/env python3
"""
Enhanced CLI for Chrono-MemBench Training
Command line interface with chrono-membench specific functionality.
"""

import argparse
import sys
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from chrono.chrono_train import main as chrono_train_main, ChronoConfig
from chrono.train import main as standard_train_main


def load_chrono_config(config_path: str) -> Dict[str, Any]:
    """Load chrono-membench configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def chrono_train_command(args):
    """Handle chrono-membench training command."""
    print("🚀 Starting Chrono-MemBench Training")
    print("=" * 50)
    
    # Load configuration
    config = load_chrono_config(args.config)
    
    # Display configuration summary
    print(f"📝 Configuration: {args.config}")
    print(f"🤖 Model: {config['model']['type']}")
    print(f"📊 Datasets: {list(config['dataset']['mixing_ratios'].keys())}")
    print(f"🎯 Max steps: {config['training']['max_steps']}")
    print(f"💾 Output: {args.output_dir}")
    
    # Show chrono-specific settings
    if 'chrono' in config:
        chrono_config = config['chrono']
        print(f"⏱️  Temporal dropout: {chrono_config.get('temporal_dropout', {}).get('enabled', False)}")
        print(f"🔄 Route-SAE: {chrono_config.get('route_sae', {}).get('enabled', False)}")
        print(f"🔗 Feature alignment: {chrono_config.get('feature_alignment', {}).get('enabled', False)}")
        print(f"📈 Dashboard: {chrono_config.get('monitoring', {}).get('dashboard_enabled', False)}")
    
    print("=" * 50)
    
    # Override sys.argv for chrono training
    original_argv = sys.argv
    sys.argv = [
        'chrono_train.py',
        '--config', args.config,
        '--output_dir', args.output_dir,
    ]
    
    # Add optional flags
    if args.sae_enabled:
        sys.argv.extend(['--sae_enabled'])
    if args.feature_alignment_enabled:
        sys.argv.extend(['--feature_alignment_enabled'])
    if args.temporal_dropout_rate:
        sys.argv.extend(['--temporal_dropout_rate', str(args.temporal_dropout_rate)])
    if args.wandb_project:
        sys.argv.extend(['--wandb_project', args.wandb_project])
    
    try:
        chrono_train_main()
    finally:
        sys.argv = original_argv


def standard_train_command(args):
    """Handle standard training command."""
    print("🚀 Starting Standard Training")
    print("=" * 50)
    
    # Override sys.argv for standard training
    original_argv = sys.argv
    sys.argv = [
        'train.py',
        '--config', args.config,
        '--output_dir', args.output_dir,
        '--base_path', args.base_path or ''
    ]
    
    try:
        standard_train_main()
    finally:
        sys.argv = original_argv


def quick_chrono_train(args):
    """Quick chrono-membench training with predefined settings."""
    model_type = args.model_type
    
    # Use chrono-membench config
    config_file = "configs/chrono_membench.yaml"
    
    if not Path(config_file).exists():
        print(f"❌ Configuration file not found: {config_file}")
        print("Please run: python src/chrono_cli.py create-config")
        return
    
    # Load and modify config for the specified model
    config = load_chrono_config(config_file)
    config['model']['type'] = model_type
    
    # Create temporary config file
    temp_config_path = f"configs/temp_{model_type}_chrono.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"🎯 Quick chrono training for {model_type}")
    print(f"📝 Using config: {temp_config_path}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Call chrono train command
    chrono_args = argparse.Namespace(
        config=temp_config_path,
        output_dir=args.output_dir,
        sae_enabled=True,
        feature_alignment_enabled=True,
        temporal_dropout_rate=0.2,
        wandb_project="chrono-membench"
    )
    
    chrono_train_command(chrono_args)
    
    # Clean up temporary config
    os.remove(temp_config_path)


def list_configs(args):
    """List available configurations."""
    configs_dir = Path('configs')
    
    if not configs_dir.exists():
        print("❌ No configs directory found.")
        return
    
    print("📋 Available configurations:")
    print("=" * 50)
    
    # Standard configs
    print("🔧 Standard Training Configs:")
    for config_file in configs_dir.glob('*.yaml'):
        if config_file.stem not in ['training_base', 'chrono_membench']:
            print(f"  • {config_file.stem}")
            print(f"    Path: {config_file}")
    
    # Chrono configs
    print("\n⏱️  Chrono-MemBench Configs:")
    chrono_configs = ['chrono_membench']
    for config_name in chrono_configs:
        config_file = configs_dir / f"{config_name}.yaml"
        if config_file.exists():
            print(f"  • {config_name}")
            print(f"    Path: {config_file}")


def create_config(args):
    """Create a new chrono-membench configuration."""
    config_name = args.name
    config_path = Path(f"configs/{config_name}.yaml")
    
    if config_path.exists() and not args.force:
        print(f"❌ Configuration '{config_name}' already exists. Use --force to overwrite.")
        return
    
    # Load base chrono config
    base_config_path = Path("configs/chrono_membench.yaml")
    if not base_config_path.exists():
        print("❌ Base chrono configuration not found. Creating default...")
        # Create default config here if needed
        return
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config based on arguments
    if args.model_type:
        config['model']['type'] = args.model_type
    
    if args.temporal_dropout_rate:
        config['chrono']['temporal_dropout']['initial_rate'] = args.temporal_dropout_rate
    
    if args.sae_latent_dim:
        config['chrono']['route_sae']['latent_dim'] = args.sae_latent_dim
    
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    
    # Update experiment info
    config['experiment']['name'] = config_name
    config['experiment']['description'] = f"Custom chrono-membench configuration: {config_name}"
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created configuration: {config_path}")


def analyze_results(args):
    """Analyze training results."""
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print(f"📊 Analyzing results in: {results_dir}")
    print("=" * 50)
    
    # Look for evaluation reports
    eval_reports = list(results_dir.glob("**/evaluation_report.json"))
    
    if not eval_reports:
        print("❌ No evaluation reports found.")
        return
    
    for report_path in eval_reports:
        print(f"\n📄 Report: {report_path}")
        
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            # Display key metrics
            if 'experiment_info' in report:
                exp_info = report['experiment_info']
                print(f"  🤖 Model: {exp_info.get('model_type', 'unknown')}")
                print(f"  📈 Steps: {exp_info.get('total_steps', 0)}")
                print(f"  📊 Datasets: {exp_info.get('datasets_used', [])}")
            
            if 'chrono_metrics' in report:
                chrono_metrics = report['chrono_metrics']
                final_purity = chrono_metrics.get('final_temporal_purity', 0.0)
                print(f"  ⏱️  Final temporal purity: {final_purity:.4f}")
                
                purity_scores = chrono_metrics.get('temporal_purity_scores', [])
                if purity_scores:
                    print(f"  📈 Purity progression: {purity_scores[0]:.4f} → {purity_scores[-1]:.4f}")
        
        except Exception as e:
            print(f"  ❌ Error reading report: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chrono-MemBench: Enhanced Training with Temporal Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick chrono training
  python src/chrono_cli.py quick-chrono gemma-2b
  
  # Full chrono training with custom config
  python src/chrono_cli.py chrono-train --config configs/chrono_membench.yaml
  
  # Standard training (fallback)
  python src/chrono_cli.py standard-train --config configs/gemma_2b.yaml
  
  # Create custom configuration
  python src/chrono_cli.py create-config my_experiment --model_type llama-3-8b
  
  # Analyze results
  python src/chrono_cli.py analyze-results --results_dir outputs/chrono
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Quick chrono training
    quick_parser = subparsers.add_parser(
        'quick-chrono',
        help='Quick chrono-membench training with predefined settings'
    )
    quick_parser.add_argument(
        'model_type',
        choices=['gemma-2b', 'llama-3-8b'],
        help='Model type to train'
    )
    quick_parser.add_argument(
        '--output_dir',
        default='outputs/chrono',
        help='Output directory for trained model'
    )
    quick_parser.set_defaults(func=quick_chrono_train)
    
    # Full chrono training
    chrono_parser = subparsers.add_parser(
        'chrono-train',
        help='Full chrono-membench training with custom configuration'
    )
    chrono_parser.add_argument(
        '--config',
        required=True,
        help='Path to chrono-membench configuration file'
    )
    chrono_parser.add_argument(
        '--output_dir',
        default='outputs/chrono',
        help='Output directory for trained model'
    )
    chrono_parser.add_argument(
        '--sae_enabled',
        action='store_true',
        help='Enable Route-SAE'
    )
    chrono_parser.add_argument(
        '--feature_alignment_enabled',
        action='store_true',
        help='Enable feature alignment'
    )
    chrono_parser.add_argument(
        '--temporal_dropout_rate',
        type=float,
        help='Temporal dropout rate (overrides config)'
    )
    chrono_parser.add_argument(
        '--wandb_project',
        default='chrono-membench',
        help='WandB project name'
    )
    chrono_parser.set_defaults(func=chrono_train_command)
    
    # Standard training (fallback)
    standard_parser = subparsers.add_parser(
        'standard-train',
        help='Standard training without chrono-membench features'
    )
    standard_parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration file'
    )
    standard_parser.add_argument(
        '--output_dir',
        default='outputs',
        help='Output directory for trained model'
    )
    standard_parser.add_argument(
        '--base_path',
        default='',
        help='Base path for data files'
    )
    standard_parser.set_defaults(func=standard_train_command)
    
    # Create configuration
    create_parser = subparsers.add_parser(
        'create-config',
        help='Create a new chrono-membench configuration'
    )
    create_parser.add_argument(
        'name',
        help='Name for the new configuration'
    )
    create_parser.add_argument(
        '--model_type',
        choices=['gemma-2b', 'llama-3-8b'],
        help='Model type for the configuration'
    )
    create_parser.add_argument(
        '--temporal_dropout_rate',
        type=float,
        help='Temporal dropout rate'
    )
    create_parser.add_argument(
        '--sae_latent_dim',
        type=int,
        help='SAE latent dimension'
    )
    create_parser.add_argument(
        '--max_steps',
        type=int,
        help='Maximum training steps'
    )
    create_parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing configuration'
    )
    create_parser.set_defaults(func=create_config)
    
    # List configurations
    list_parser = subparsers.add_parser(
        'list-configs',
        help='List available configurations'
    )
    list_parser.set_defaults(func=list_configs)
    
    # Analyze results
    analyze_parser = subparsers.add_parser(
        'analyze-results',
        help='Analyze training results'
    )
    analyze_parser.add_argument(
        '--results_dir',
        default='outputs/chrono',
        help='Directory containing training results'
    )
    analyze_parser.set_defaults(func=analyze_results)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the chosen command
    args.func(args)


if __name__ == '__main__':
    main()
