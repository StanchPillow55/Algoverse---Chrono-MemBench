# Chrono-MemBench Training Scripts

This directory contains enhanced training scripts for the chrono-membench framework, which implements temporal dropout regularization and Route-SAE integration for studying memory formation in language models.

## Overview

The chrono-membench framework extends the existing training infrastructure with:

- **Temporal Dropout Regularization**: Randomizes encoder weights across training steps to encourage sharper feature identities
- **Route-SAE Integration**: Sparse Autoencoder for feature extraction and analysis
- **Feature Alignment Loss**: Maintains interpretability throughout training
- **Temporal Purity Metrics**: Measures feature consistency across checkpoints
- **Enhanced Monitoring**: Real-time dashboard with WandB integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_training.txt
pip install wandb  # For dashboard monitoring
```

### 2. Quick Training

```bash
# Quick chrono training with Gemma-2B
python src/chrono_cli.py quick-chrono gemma-2b

# Quick chrono training with Llama-3-8B  
python src/chrono_cli.py quick-chrono llama-3-8b --output_dir outputs/llama_chrono
```

### 3. Full Configuration Training

```bash
# Full chrono training with custom configuration
python src/chrono_cli.py chrono-train --config configs/chrono_membench.yaml

# Enable all features
python src/chrono_cli.py chrono-train \
    --config configs/chrono_membench.yaml \
    --sae_enabled \
    --feature_alignment_enabled \
    --temporal_dropout_rate 0.2 \
    --wandb_project my-chrono-experiment
```

## Available Commands

### Training Commands

```bash
# Quick chrono training
python src/chrono_cli.py quick-chrono {gemma-2b|llama-3-8b}

# Full chrono training
python src/chrono_cli.py chrono-train --config CONFIG_FILE

# Standard training (fallback)
python src/chrono_cli.py standard-train --config CONFIG_FILE
```

### Configuration Management

```bash
# List available configurations
python src/chrono_cli.py list-configs

# Create new configuration
python src/chrono_cli.py create-config my_experiment \
    --model_type gemma-2b \
    --temporal_dropout_rate 0.3 \
    --max_steps 10000

# Create configuration with custom SAE settings
python src/chrono_cli.py create-config large_sae_experiment \
    --model_type llama-3-8b \
    --sae_latent_dim 4096 \
    --max_steps 20000
```

### Results Analysis

```bash
# Analyze training results
python src/chrono_cli.py analyze-results --results_dir outputs/chrono

# Analyze specific experiment
python src/chrono_cli.py analyze-results --results_dir outputs/my_experiment
```

## Dataset Configuration

The training scripts use the existing datasets in your project:

- **FineWeb-Edu** (40%): High-quality educational web content
- **WikiText-103** (30%): Clean Wikipedia articles  
- **Orca Math** (20%): Mathematical reasoning problems
- **BookCorpus** (10%): Book text for language modeling

### Custom Dataset Mixing

You can modify the dataset ratios in the configuration files:

```yaml
dataset:
  mixing_ratios:
    fineweb_edu: 0.5      # 50% educational content
    wikitext: 0.25        # 25% Wikipedia
    orca_math: 0.15       # 15% math problems
    bookcorpus: 0.1       # 10% books
```

## Configuration Files

### Main Configuration: `configs/chrono_membench.yaml`

The primary configuration file with all chrono-membench settings:

```yaml
# Chrono-specific settings
chrono:
  temporal_dropout:
    enabled: true
    initial_rate: 0.2
    schedule: "cosine"
    
  route_sae:
    enabled: true
    latent_dim: 2048
    sparsity_weight: 0.1
    
  feature_alignment:
    enabled: true
    weight: 0.05
    
  monitoring:
    wandb_project: "chrono-membench"
    dashboard_enabled: true
```

### Key Configuration Sections

1. **Model Configuration**: Specifies model type and source
2. **Dataset Configuration**: Dataset sources and mixing ratios
3. **Training Configuration**: Standard training parameters
4. **Chrono Configuration**: Temporal dropout, SAE, and feature alignment settings
5. **Monitoring Configuration**: Dashboard and logging settings

## Features

### Temporal Dropout

Implements temporal dropout regularization that varies throughout training:

- **Schedule Options**: `linear`, `cosine`, `constant`
- **Configurable Rates**: Initial rate, minimum rate, decay schedule
- **Dynamic Application**: Applied to Route-SAE encoder weights

### Route-SAE (Sparse Autoencoder)

Implements sparse autoencoder for feature extraction:

- **Configurable Latent Dimension**: Adjustable based on model size
- **Sparsity Loss**: L1 regularization for sparse representations
- **Reconstruction Loss**: MSE loss for faithful reconstruction
- **Checkpointing**: Save SAE state at regular intervals

### Feature Alignment

Maintains interpretability throughout training:

- **Attention Diversity**: Encourages diverse attention patterns
- **Configurable Weight**: Balance between alignment and performance
- **Multiple Alignment Types**: Extensible framework for different alignment methods

### Temporal Purity Metrics

Measures feature consistency across training:

- **Cosine Similarity**: Between feature representations across checkpoints
- **Configurable Window**: Number of checkpoints to compare
- **Threshold-based**: Configurable similarity threshold for purity
- **Birth Curve Analysis**: Track emergence of new features

### Enhanced Monitoring

Real-time monitoring and analysis:

- **WandB Integration**: Automatic logging of all metrics
- **Custom Metrics**: Temporal purity, feature birth rates, SAE losses
- **Checkpoint Analysis**: Automatic feature extraction and comparison
- **Evaluation Reports**: Comprehensive JSON reports with all metrics

## Output Structure

Training produces the following output structure:

```
outputs/chrono/
├── final_checkpoint/           # Final model checkpoint
├── checkpoint-500/             # Intermediate checkpoints
├── checkpoint-1000/
├── logs/                       # Training logs
├── evaluation_report.json      # Final evaluation report
└── chrono_training.log         # Chrono-specific logs
```

### Evaluation Report

The evaluation report contains:

```json
{
  "experiment_info": {
    "model_type": "gemma-2b",
    "total_steps": 5000,
    "datasets_used": ["fineweb_edu", "wikitext", "orca_math", "bookcorpus"],
    "dataset_mixing_ratios": {...}
  },
  "chrono_metrics": {
    "temporal_purity_scores": [0.1, 0.3, 0.5, 0.7],
    "final_temporal_purity": 0.7,
    "feature_birth_events": [...]
  },
  "training_config": {...}
}
```

## Monitoring with WandB

The training scripts integrate with Weights & Biases for real-time monitoring:

1. **Setup WandB**:
   ```bash
   wandb login
   ```

2. **Enable Dashboard**:
   ```yaml
   chrono:
     monitoring:
       wandb_project: "my-chrono-experiment"
       dashboard_enabled: true
   ```

3. **Tracked Metrics**:
   - Training loss and perplexity
   - Temporal purity scores
   - SAE reconstruction and sparsity losses
   - Feature birth rates
   - Attention diversity metrics

## Advanced Usage

### Custom Temporal Dropout Schedules

You can implement custom dropout schedules by modifying the `TemporalDropoutScheduler` class:

```python
def get_dropout_rate(self, step: int) -> float:
    # Custom schedule implementation
    if step < 1000:
        return 0.3
    elif step < 3000:
        return 0.2
    else:
        return 0.1
```

### Custom SAE Architectures

Extend the `RouteDAE` class for custom SAE architectures:

```python
class CustomSAE(RouteDAE):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(input_dim, latent_dim, **kwargs)
        # Add custom layers
        self.custom_layer = nn.Linear(latent_dim, latent_dim)
```

### Custom Feature Alignment

Implement custom feature alignment methods:

```python
def compute_custom_alignment_loss(self, outputs):
    # Custom alignment loss computation
    return custom_loss
```

## Performance Considerations

### Memory Usage

- **SAE Overhead**: Route-SAE adds ~20% memory overhead
- **Checkpoint Storage**: Feature representations require additional storage
- **Monitoring**: Dashboard logging adds minimal overhead

### Training Speed

- **SAE Training**: Adds ~10-15% training time
- **Temporal Purity**: Computed efficiently every N steps
- **Feature Extraction**: Minimal impact on training speed

### Recommended Settings

For different model sizes:

```yaml
# Gemma-2B
chrono:
  route_sae:
    latent_dim: 2048
  
# Llama-3-8B  
chrono:
  route_sae:
    latent_dim: 4096
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or SAE latent dimension
2. **SAE Not Converging**: Lower SAE learning rate or increase warmup steps
3. **Low Temporal Purity**: Reduce temporal dropout rate or increase alignment weight
4. **WandB Issues**: Check API key and project settings

### Debug Mode

Enable debug logging:

```bash
python src/chrono_cli.py chrono-train --config configs/chrono_membench.yaml \
    --debug
```

## Experimental Results

Expected results from the chrono-membench framework:

- **Temporal Purity**: Should increase from ~0.3 to ~0.7 over training
- **Feature Birth**: Clear feature emergence patterns
- **SAE Reconstruction**: Low reconstruction loss with high sparsity
- **Performance**: <1% perplexity degradation vs baseline

## Next Steps

1. **Experiment with different temporal dropout schedules**
2. **Try different SAE architectures**
3. **Implement custom feature alignment methods**
4. **Scale to larger models (Llama-3-70B)**
5. **Add multimodal capabilities (LLaVA)**

## Support

For issues or questions:
1. Check the logs in `outputs/chrono/chrono_training.log`
2. Review the evaluation report for metrics
3. Use the analysis tools to understand results
4. Modify configurations based on your specific needs

Happy training with chrono-membench! 🚀
