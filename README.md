# Chrono-MemBench

🧠 **A Two-Phase Framework for Tracking and Steering Memory Features in Large Language Models**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StanchPillow55/Algoverse---Chrono-MemBench/blob/main/notebooks/chrono_membench_colab_universal.ipynb)

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
```bash
# Open the notebook in Colab using the badge above
# or manually: notebooks/chrono_membench_colab_universal.ipynb
```

### Option 2: Local Training
```bash
# Clone repository
git clone https://github.com/StanchPillow55/Algoverse---Chrono-MemBench.git
cd Algoverse---Chrono-MemBench

# Install dependencies
pip install -r requirements.txt

# Quick training
python src/chrono_cli.py quick-chrono gemma-2b

# macOS (Apple Silicon)
python scripts/train_macos.py --config configs/chrono_membench.yaml
```

## 🎯 Features

- **🔄 Temporal Dropout Regularization** - Encourages sharper feature identities
- **🧠 Route-SAE Integration** - Sparse Autoencoder for feature extraction 
- **📊 Feature Alignment Loss** - Maintains interpretability during training
- **📈 Temporal Purity Metrics** - Measures feature consistency across checkpoints
- **🌐 Real-time Monitoring** - WandB dashboard integration
- **💻 Multi-Platform** - Supports macOS (MPS), Linux (CUDA), and Google Colab

## 📂 Project Structure

```
chrono-membench/
├── 📖 docs/                    # Documentation and guides
├── 📓 notebooks/               # Jupyter notebooks
│   ├── chrono_membench_colab_universal.ipynb  # Main Colab notebook
│   └── standard_training_colab.ipynb          # Standard training
├── ⚙️  configs/                # Training configurations
├── 🧠 src/                     # Source code
│   ├── chrono/                # Core chrono-membench modules
│   └── chrono_cli.py          # Command-line interface
├── 🔧 scripts/                # Utility scripts
├── 📊 data/                   # Datasets and models (DVC managed)
└── 📋 requirements.txt        # Dependencies
```

## 💾 Datasets

| Dataset | Size | Samples | Description |
|---------|------|---------|-------------|
| FineWeb-Edu | 254 MB | 50,000 | Educational web content |
| WikiText-103 | 5.9 MB | 20,000 | Wikipedia articles |
| Orca Math | 24 MB | 25,000 | Mathematical reasoning |
| BookCorpus | 1.0 MB | 15,000 | Book text |
| **Total** | **285 MB** | **110,000** | Mixed educational datasets |

## 🤖 Supported Models

- **Gemma-2B** - Fast training, good for experimentation
- **Llama-3-8B** - Production-quality results
- **Apple Silicon** - Optimized for M1/M2/M3 chips with MPS

## 📚 Documentation

All documentation is in the [`docs/`](docs/) directory:

- **[Setup Guide](docs/TRAINING_README.md)** - Complete installation and setup
- **[Chrono Training](docs/CHRONO_TRAINING_README.md)** - Advanced features and usage
- **[Dataset Analysis](docs/DATASET_WANDB_ANALYSIS.md)** - Data and WandB requirements
- **[Colab Setup](docs/COLAB_LINKS.md)** - Google Colab instructions

## 🛠️ Development

```bash
# Run tests
python scripts/test_config_fix.py

# Check macOS setup
python scripts/train_macos.py --check_only

# List available configs
python src/chrono_cli.py list-configs
```

## 📊 Monitoring

- **WandB Dashboard** - Real-time training metrics
- **Temporal Purity Tracking** - Feature consistency analysis
- **Free Tier Compatible** - Works with WandB free plan

## 🤝 Contributing

We welcome contributions! Please:
1. Check existing issues and documentation
2. Test your changes with the provided test scripts
3. Update documentation in `docs/` directory
4. Follow the existing code style

## 📄 License

This project is licensed under the MIT License.
