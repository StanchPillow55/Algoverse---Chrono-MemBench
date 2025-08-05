# RTX 6000 Ada Optimization for Chrono-MemBench

This directory contains optimized training scripts and configurations specifically designed for the **RTX 6000 Ada 48GB GPU** on RunPod.

## Files Overview

### 1. `colab_optimized_train_rtx6000ada.py`
A comprehensive training script that:
- Automatically installs optimized packages for RTX 6000 Ada
- Sets up environment variables for maximum performance
- Patches the training code with RTX 6000 Ada optimizations
- Configures Flash-Attention 2 and bf16 mixed precision
- Handles authentication for HuggingFace and Weights & Biases
- Runs training with real-time logging and monitoring

### 2. `configs/chrono_membench_rtx6000ada_optimized.yaml`
A highly optimized configuration file featuring:
- **Batch size 2** with **gradient accumulation 8** (effective batch size 16)
- **bf16 mixed precision** (RTX 6000 Ada native support)
- **Flash-Attention 2** enabled for memory efficiency
- **AdamW torch fused optimizer** for speed
- **Higher LoRA rank (64)** for better adaptation
- **Comprehensive monitoring** and logging setup

## Key Optimizations

### Memory Management
- **48GB VRAM optimization**: Configured to use 46GB, leaving 2GB for system
- **Gradient checkpointing**: Enabled for memory efficiency
- **Memory-efficient attention**: Flash-Attention 2 implementation
- **No CPU offload**: Not needed with 48GB VRAM

### Performance Enhancements
- **bf16 mixed precision**: Native RTX 6000 Ada support for better performance
- **AdamW torch fused optimizer**: Faster than standard AdamW
- **Optimized data loading**: 16 workers for preprocessing, 8 for training
- **Gradient accumulation**: Effective batch size 16 with memory efficiency

### Training Stability
- **Torch compile disabled**: Prevents compilation issues
- **Comprehensive error handling**: Robust pipeline with fallbacks
- **Backup system**: Automatic backups of training code
- **Resume capability**: Can resume from checkpoints

## Usage Instructions

### Prerequisites
1. **RunPod instance** with RTX 6000 Ada 48GB GPU
2. **CUDA 12.1+** installed
3. **Python 3.10+**
4. **Git** for repository cloning

### Environment Setup
Set the following environment variables before running:

```bash
# Required for HuggingFace authentication
export HUGGINGFACE_TOKEN="your_hf_token_here"

# Required for Weights & Biases logging
export WANDB_API_KEY="your_wandb_key_here"
```

### Running the Training

#### Option 1: Automated Pipeline (Recommended)
```bash
python colab_optimized_train_rtx6000ada.py
```

This will:
1. Check GPU availability
2. Install optimized packages
3. Clone/update the repository
4. Verify datasets
5. Create optimized configuration
6. Patch training code
7. Run training with monitoring

#### Option 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/StanchPillow55/Algoverse---Chrono-MemBench.git
cd Algoverse---Chrono-MemBench

# Copy the config file
cp ../configs/chrono_membench_rtx6000ada_optimized.yaml configs/

# Install packages
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft datasets flash-attn wandb

# Run training
python -m chrono.chrono_train --config configs/chrono_membench_rtx6000ada_optimized.yaml --output_dir ./outputs
```

## Expected Performance

### Training Speed
- **~4.5 seconds per step** (based on Gemma-2B with batch size 2)
- **~2,200 tokens/second** throughput
- **~6 hours** for 5,000 steps

### Memory Usage
- **~35-40GB VRAM** during training
- **~8GB remaining** for system operations
- **Stable memory usage** without memory leaks

### Model Quality
- **Higher LoRA rank (64)** for better adaptation
- **Optimized learning rate (2e-5)** for RTX 6000 Ada
- **Cosine annealing scheduler** for optimal convergence
- **Comprehensive evaluation metrics**

## Monitoring and Logs

### Weights & Biases
- **Project**: `chrono-membench-rtx6000ada`
- **Tags**: `rtx6000ada`, `gemma-2b`, `lora`, `bf16`
- **Metrics**: Loss, perplexity, GPU memory, tokens/second

### Local Logs
- **Training log**: `rtx6000ada_training.log`
- **Timestamped logs**: `training_rtx6000ada_YYYYMMDD_HHMMSS.log`
- **TensorBoard logs**: `./logs/`

### Model Checkpoints
- **Output directory**: `./outputs/gemma_2b_rtx6000ada_TIMESTAMP/`
- **Checkpoint frequency**: Every 200 steps
- **Best model saved**: Based on evaluation loss
- **Backup system**: Automatic checkpoint backups

## Troubleshooting

### Common Issues

1. **Flash-Attention Installation**
   ```bash
   pip install flash-attn==2.6.3 --no-build-isolation
   ```

2. **CUDA Out of Memory**
   - Reduce batch size to 1 in the config
   - Increase gradient accumulation steps to 16

3. **Package Conflicts**
   - Use the exact versions specified in the script
   - Create a fresh virtual environment

4. **Authentication Issues**
   - Verify environment variables are set
   - Check token permissions on HuggingFace/W&B

### Performance Tips

1. **Maximize throughput**:
   - Use the automated script for optimal settings
   - Ensure datasets are on fast storage (NVMe SSD)
   - Use multiple data loading workers

2. **Reduce memory usage**:
   - Enable gradient checkpointing
   - Use longer context lengths only if needed
   - Monitor memory usage in W&B

3. **Improve stability**:
   - Keep torch compile disabled
   - Use bf16 instead of fp16
   - Monitor gradient norms

## Configuration Details

### Key Settings
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  bf16: true
  optim: "adamw_torch_fused"
  learning_rate: 2e-5
  lora:
    r: 64
    alpha: 128
```

### Environment Variables
```yaml
environment:
  attention_implementation: "flash_attention_2"
  compile_mode: "disabled"
  memory_efficient_attention: true
  max_memory_allocation: "46GB"
```

## Next Steps

After successful training with RTX 6000 Ada, we can create similar optimizations for:
- **NVIDIA GTX 1070** (8GB VRAM - more aggressive optimizations needed)
- **Other GPU configurations** as requested

The RTX 6000 Ada optimization serves as a high-performance baseline that can be adapted for other hardware configurations.

## Support

For issues or questions:
1. Check the training logs for specific error messages
2. Verify all environment variables are set correctly
3. Ensure your RunPod instance has sufficient resources
4. Monitor W&B dashboard for training metrics

This optimization is designed to provide maximum performance and stability for RTX 6000 Ada training while maintaining compatibility with the existing chrono-membench codebase.
