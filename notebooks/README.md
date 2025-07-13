# Chrono-MemBench Notebooks

This directory contains Jupyter notebooks for training chrono-membench models.

## 📓 Available Notebooks

### 🚀 **chrono_membench_colab_universal.ipynb** 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StanchPillow55/Algoverse---Chrono-MemBench/blob/main/notebooks/chrono_membench_colab_universal.ipynb)

**The main notebook for chrono-membench training.**

**Features:**
- ✅ Temporal dropout regularization
- ✅ Route-SAE integration  
- ✅ Feature alignment loss
- ✅ WandB monitoring
- ✅ Multi-platform support (macOS MPS, CUDA, Colab)
- ✅ Local and HuggingFace model support

**Use this for:** Advanced chrono-membench experiments with temporal analysis

---

### 🔧 **standard_training_colab.ipynb**
**Standard model training without chrono-membench features.**

**Features:**
- ✅ Basic LoRA training
- ✅ Multiple model support (Gemma-2B, Llama-3-8B, LLaVA)
- ✅ Colab optimization
- ✅ Simple setup

**Use this for:** Basic model fine-tuning without temporal analysis

## 🎯 Quick Start

### Option 1: Direct Colab (Recommended)
Click the Colab badge above to open the main notebook directly in Google Colab.

### Option 2: Local Jupyter
```bash
# Clone the repository
git clone https://github.com/StanchPillow55/Algoverse---Chrono-MemBench.git
cd Algoverse---Chrono-MemBench

# Install dependencies
pip install -r requirements.txt
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/
```

## 🔍 Choosing the Right Notebook

| Goal | Notebook | Features |
|------|----------|----------|
| **Research & Experiments** | `chrono_membench_colab_universal.ipynb` | Full chrono features |
| **Basic Fine-tuning** | `standard_training_colab.ipynb` | Simple training |
| **Temporal Analysis** | `chrono_membench_colab_universal.ipynb` | Temporal purity metrics |
| **Multi-model Support** | Both | Gemma-2B, Llama-3-8B |

## 🛠️ Setup Requirements

Both notebooks require:
- Google Colab account (for Colab usage)
- WandB account (optional, for monitoring)
- Google Drive (for saving outputs)

## 📚 Documentation

For detailed setup and usage instructions, see:
- **[Setup Guide](../docs/TRAINING_README.md)**
- **[Chrono Training Guide](../docs/CHRONO_TRAINING_README.md)**
- **[Colab Setup](../docs/COLAB_LINKS.md)**
