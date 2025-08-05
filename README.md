# Chrono-MemBench

🧠 **A Two-Phase Framework for Tracking and Steering Memory Features in Large Language Models**

![Version](https://img.shields.io/badge/version-0.2.0--chrono-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bradleyharaguchi/Chrono-MemBench/blob/main/notebooks/01_quickstart.ipynb)

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
```bash
# Open the notebook in Colab using the badge above
# or manually: notebooks/01_quickstart.ipynb
```

### Option 2: Local Training
```bash
# Clone repository
git clone https://github.com/bradleyharaguchi/Chrono-MemBench.git
cd Chrono-MemBench

# Install dependencies
pip install -r requirements.txt

# Quick training
python training/train.py --config configs/chrono_membench.yaml

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
│   └── 01_quickstart.ipynb   # Main quickstart notebook
├── ⚙️  configs/                # Training configurations
├── 🧠 src/algoverse/chrono/    # Source code
│   ├── chrono_sae/            # ChronoSAE implementation
│   ├── membench_x/            # Memory benchmarking
│   └── training/              # Training utilities
├── 🚂 training/                # Main training scripts
│   ├── train.py               # Primary training script
│   └── loop.py                # Training loop implementation
├── 🧪 tests/                   # Test suite
│   ├── smoke/                 # Smoke tests
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── 🔧 scripts/                # Utility scripts
├── 📦 legacy/                 # Legacy code (archived)
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
# Run all tests
pytest -q

# Run smoke tests only
pytest tests/smoke/

# Run unit tests only
pytest tests/unit/

# Check macOS setup
python scripts/train_macos.py --check_only

# Find orphaned files
python scripts/find_orphans.py

# Test configuration files
python scripts/test_config_fix.py
```

## 📊 Monitoring

- **WandB Dashboard** - Real-time training metrics
- **Temporal Purity Tracking** - Feature consistency analysis
- **Free Tier Compatible** - Works with WandB free plan

## 🤝 Contributing

We welcome contributions! Please follow this workflow:

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch from `main`
3. **Make Changes**: Implement your changes following existing patterns
4. **Test**: Run the full test suite:
   ```bash
   pytest -q                    # All tests
   pytest tests/smoke/          # Quick smoke tests
   pytest tests/unit/           # Unit tests
   ```
5. **Documentation**: Update relevant documentation in `docs/`
6. **Commit**: Use clear, descriptive commit messages
7. **Push**: Push to your fork and create a pull request

### Development Guidelines
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation for user-facing changes
- Check for orphaned files with `python scripts/find_orphans.py`

## 📄 License

This project is licensed under the MIT License.
