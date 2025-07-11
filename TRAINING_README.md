# Language Model Training Guide

This guide explains how to train Gemma-2B, Llama-3-8B, and LLaVA-1.6 models using your custom datasets.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Open `notebooks/training_colab_template.ipynb` in Google Colab
2. Follow the step-by-step instructions
3. Your trained model will be saved to Google Drive

### Option 2: Command Line Interface
```bash
# Quick training with Gemma-2B
python src/cli.py quick gemma-2b

# List available configurations
python src/cli.py list-configs

# Training with custom config
python src/cli.py train --config configs/gemma_2b.yaml
```

### Option 3: Direct Training Script
```bash
python src/chrono/train.py --config configs/gemma_2b.yaml --output_dir outputs/gemma_2b
```

## 📊 Available Datasets

Your training setup includes these datasets (110,000 samples total):

| Dataset | Samples | Size | Description |
|---------|---------|------|-------------|
| **FineWeb-Edu** | 50,000 | 253.8 MB | High-quality educational web content |
| **WikiText-103** | 20,000 | 5.9 MB | Clean Wikipedia articles |
| **Orca Math** | 25,000 | 23.9 MB | Mathematical reasoning problems |
| **BookCorpus** | 15,000 | 1.0 MB | Book text for language modeling |

## 🤖 Supported Models

### Gemma-2B
- **Configuration**: `configs/gemma_2b.yaml`
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Memory**: ~6-8GB GPU memory
- **Training Time**: ~2-3 hours on V100
- **Best For**: Quick experimentation, educational content

### Llama-3-8B
- **Configuration**: `configs/llama_3_8b.yaml`
- **Training Method**: LoRA
- **Memory**: ~16-24GB GPU memory
- **Training Time**: ~4-6 hours on V100
- **Best For**: Production applications, comprehensive reasoning

### LLaVA-1.6-7B
- **Configuration**: `configs/llava_1_6_7b.yaml`
- **Training Method**: LoRA
- **Memory**: ~16-20GB GPU memory
- **Training Time**: ~4-5 hours on V100
- **Best For**: Vision-language tasks (requires additional vision data)

## 🔧 Configuration Options

### Model Source
Choose between local models or HuggingFace streaming:

```yaml
model:
  source: \"local\"        # Use downloaded models
  # source: \"huggingface\"  # Stream from HuggingFace Hub
```

### Training Parameters
Key parameters you can adjust:

```yaml
training:
  batch_size: 4                    # Adjust based on GPU memory
  gradient_accumulation_steps: 8   # Effective batch size = batch_size * this
  learning_rate: 5e-5             # Learning rate
  max_steps: 1000                 # Total training steps
  training_type: \"lora\"           # \"lora\", \"fine_tuning\", \"full_training\"
```

### Dataset Mixing
Customize dataset ratios:

```yaml
dataset:
  mixing_ratios:
    fineweb_edu: 0.5    # 50% educational content
    wikitext: 0.2       # 20% Wikipedia
    orca_math: 0.2      # 20% math problems
    bookcorpus: 0.1     # 10% books
```

## 🏗️ Project Structure

```
chrono-membench/
├── configs/                 # Training configurations
│   ├── training_base.yaml  # Base configuration
│   ├── gemma_2b.yaml       # Gemma-2B specific
│   ├── llama_3_8b.yaml     # Llama-3-8B specific
│   └── llava_1_6_7b.yaml   # LLaVA-1.6 specific
├── src/
│   ├── chrono/
│   │   ├── train.py        # Main training script
│   │   └── data_loader.py  # Dataset handling
│   └── cli.py              # Command line interface
├── data/
│   └── raw/                # Your datasets (managed by DVC)
├── notebooks/
│   └── training_colab_template.ipynb  # Colab notebook
└── requirements_training.txt  # Training dependencies
```

## 📋 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_training.txt
```

### 2. HuggingFace Authentication (if using HuggingFace models)
```bash
huggingface-cli login
```

### 3. Verify Data Access
```bash
# Check datasets are available
ls -la data/raw/*.jsonl

# If using DVC
dvc pull
```

### 4. Choose Your Training Method

#### Method A: Google Colab (Easiest)
1. Upload your project to Google Drive
2. Open `notebooks/training_colab_template.ipynb`
3. Follow the notebook instructions

#### Method B: CLI (Local/Remote)
```bash
# Quick start
python src/cli.py quick gemma-2b

# Custom training
python src/cli.py train --config configs/gemma_2b.yaml --output_dir my_models
```

#### Method C: Direct Script
```bash
python src/chrono/train.py --config configs/gemma_2b.yaml --output_dir outputs
```

## 🔍 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir outputs/logs
```

### Training Logs
- **Console**: Real-time training progress
- **TensorBoard**: Metrics visualization
- **Checkpoints**: Model saved every N steps

### Sample Output
During training, you'll see:
```
Loading gemma-2b from google/gemma-2b (source: huggingface)
Loading 4 datasets...
Loaded 50000 samples from fineweb_edu
Created mixed dataset with 80000 total samples
Split dataset: 72000 train, 8000 validation
Starting training...
```

## 🎯 Model Testing

After training, test your model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model
tokenizer = AutoTokenizer.from_pretrained(\"outputs/gemma_2b\")
model = AutoModelForCausalLM.from_pretrained(\"outputs/gemma_2b\")

# Generate text
prompt = \"The future of AI is\"
inputs = tokenizer(prompt, return_tensors=\"pt\")
outputs = model.generate(inputs.input_ids, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```yaml
# Reduce batch size
training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

**Model Loading Issues**
```bash
# Check HuggingFace authentication
huggingface-cli whoami

# Verify model paths
ls -la data/models/
```

**Dataset Loading Issues**
```bash
# Check dataset files
ls -la data/raw/*.jsonl

# Test dataset loading
python -c "from src.chrono.data_loader import *; print('Data loader works!')"
```

### Memory Requirements

| Model | GPU Memory | Batch Size | Context Length |
|-------|------------|------------|----------------|
| Gemma-2B | 6-8GB | 4-8 | 2048 |
| Llama-3-8B | 16-24GB | 1-2 | 4096 |
| LLaVA-1.6-7B | 16-20GB | 1-2 | 2048 |

### Performance Tips

1. **Use LoRA**: Reduces memory usage by 60-80%
2. **Mixed Precision**: Use `fp16` or `bf16`
3. **Gradient Accumulation**: Simulate larger batch sizes
4. **Gradient Checkpointing**: Trade compute for memory

## 📊 Expected Results

### Training Metrics
- **Perplexity**: Should decrease over time
- **Loss**: Should steadily decline
- **Learning Rate**: Follows cosine schedule

### Performance Benchmarks
- **Gemma-2B**: ~10-15 tokens/sec on V100
- **Llama-3-8B**: ~3-5 tokens/sec on V100
- **LLaVA-1.6**: ~2-4 tokens/sec on V100

## 🔗 Next Steps

1. **Evaluate**: Test model on your specific tasks
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Deploy**: Use trained models in your applications
4. **Iterate**: Experiment with different dataset mixtures

## 📧 Support

- Check logs for detailed error messages
- Use the Colab notebook for the most reliable experience
- Adjust configurations based on your hardware
- Start with smaller models (Gemma-2B) before moving to larger ones

Happy training! 🚀
