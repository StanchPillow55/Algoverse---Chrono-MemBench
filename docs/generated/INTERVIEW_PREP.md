# 🎯 Interview Prep - Chrono-MemBench

**Last Updated:** 2026-04-29

---

## 60-90 Second STAR Pitch

### Situation
Sparse Autoencoders (SAEs) are used to extract interpretable features from LLMs, but feature identities can drift across training checkpoints, making temporal analysis difficult.

### Task
Build a framework for tracking and steering memory features in LLMs across training checkpoints, ensuring feature consistency and interpretability.

### Action
- Designed **Chrono-SAE** with temporal dropout regularization for sharper features
- Implemented **Route-SAE integration** for sparse feature extraction
- Created **feature alignment loss** to maintain interpretability during training
- Built **temporal purity metrics** to measure feature consistency
- Added **multi-platform support** (macOS MPS, Linux CUDA, Google Colab)
- Integrated **WandB monitoring** for real-time training dashboards

### Result

| Feature | Status | Evidence |
|---------|--------|----------|
| Temporal dropout | **Confirmed** | `README.md:37` |
| Route-SAE integration | **Confirmed** | `README.md:38` |
| Temporal purity metrics | **Confirmed** | `README.md:40` |
| Multi-platform | **Confirmed** | macOS, Linux, Colab |
| WandB monitoring | **Confirmed** | `README.md:41` |

---

## Technical Deep Dive

### Architecture
- **Framework:** PyTorch 2.0+ with custom training loop
- **SAE:** Sparse Autoencoder with temporal regularization
- **Data:** FineWeb-Edu, WikiText-103, Orca Math, BookCorpus (285MB total)
- **Models:** Gemma-2B, Llama-3-8B
- **Monitoring:** WandB for metrics, feature tracking

### Key Concepts
- **Temporal Dropout:** Randomly drops features to encourage independence
- **Feature Alignment Loss:** Maintains feature-to-concept mappings
- **Temporal Purity:** Measures consistency across checkpoints

---

## Drill-Down Q&A

### Q1: "What is temporal dropout regularization?"

**Answer (Confirmed):** Randomly drops features during training to encourage sharper, more independent feature identities. Prevents features from co-adapting.

### Q2: "How do you measure temporal purity?"

**Answer (Confirmed):** Compares feature activations across training checkpoints. High purity = consistent feature-to-concept mapping over time.

### Q3: "What datasets do you use?"

**Answer (Confirmed):** Mixed educational datasets: FineWeb-Edu (50K), WikiText-103 (20K), Orca Math (25K), BookCorpus (15K). Total 110K samples, 285MB.

### Q4: "Why multi-platform support?"

**Answer (Confirmed):** Research accessibility. Supports Apple Silicon (MPS), NVIDIA GPUs (CUDA), and Google Colab (free tier compatible).

---

## Quick Reference

| Topic | Evidence |
|-------|----------|
| Project structure | `README.md:44-65` |
| Datasets | `README.md:68-76` |
| Models | `README.md:78-83` |
| Testing | `pytest -q` |
