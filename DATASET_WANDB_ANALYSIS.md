# Dataset Sizes and WandB Requirements Analysis

## Dataset Overview

### Dataset Sizes
| Dataset | File Size | Sample Count | Description |
|---------|-----------|--------------|-------------|
| **FineWeb-Edu** | 254 MB | 50,000 | High-quality educational web content |
| **Orca Math** | 24 MB | 25,000 | Mathematical reasoning problems |
| **WikiText-103** | 5.9 MB | 20,000 | Wikipedia articles for language modeling |
| **BookCorpus** | 1.0 MB | 15,000 | Book text for language modeling |
| **TOTAL** | **285 MB** | **110,000 samples** | Mixed educational datasets |

### Dataset Distribution (Mixing Ratios)
- **FineWeb-Edu**: 40% (44,000 samples)
- **WikiText-103**: 30% (33,000 samples)  
- **Orca Math**: 20% (22,000 samples)
- **BookCorpus**: 10% (11,000 samples)

## Training Configuration Impact

### Training Parameters
- **Max Steps**: 5,000 steps
- **Eval Steps**: Every 250 steps (20 evaluations)
- **Save Steps**: Every 500 steps (10 checkpoints)
- **Batch Size**: 2 (with 16 gradient accumulation = effective batch size 32)
- **Max Length**: 2,048 tokens per sample

### Estimated Training Data Usage
- **Total Samples**: ~110,000 samples
- **Training Split**: 85% = ~93,500 samples
- **Validation Split**: 15% = ~16,500 samples
- **Tokens per Sample**: ~2,048 tokens
- **Total Training Tokens**: ~191M tokens
- **Training Duration**: ~3-6 hours (depending on hardware)

## WandB Usage Analysis

### Logging Frequency
- **Training Metrics**: Every 50 steps (default) = 100 log points
- **Evaluation Metrics**: Every 250 steps = 20 evaluation runs
- **Custom Metrics**: Every 100 steps = 50 custom metric logs
- **Checkpoints**: 10 checkpoints with feature representations

### Metrics Being Logged

#### Standard Metrics (per step)
- Training loss
- Learning rate
- Gradient norm
- Training perplexity

#### Evaluation Metrics (every 250 steps)
- Validation loss
- Validation perplexity
- Temporal purity scores

#### Chrono-Specific Metrics (every 100 steps)
- SAE reconstruction loss
- SAE sparsity loss
- Temporal dropout rate
- Feature alignment loss
- Attention diversity scores

#### Large Data Objects
- **Feature representations**: Saved every 500 steps (10 times)
- **SAE state dictionaries**: Saved every 500 steps (10 times)
- **Temporal purity matrices**: Computed every 250 steps (20 times)

### WandB Storage Requirements

#### For Single Training Run (5,000 steps)
- **Scalar Metrics**: ~500 data points × 10 metrics = ~5,000 data points
- **Evaluation Data**: 20 evaluation runs × 5 metrics = ~100 evaluation points
- **Feature Representations**: 10 checkpoints × ~50MB = ~500MB
- **SAE States**: 10 checkpoints × ~100MB = ~1GB
- **Temporal Analysis Data**: 20 evaluations × ~10MB = ~200MB
- **Total per run**: ~1.7GB storage + 5,100 metric data points

#### For Full Experiment (with ablations)
If running ablation studies (4 configurations):
- **Total Storage**: ~6.8GB
- **Total Metrics**: ~20,400 data points
- **Total Runs**: 4 runs

## WandB Membership Requirements

### Free Tier Limitations
- **Storage**: 100GB total
- **Private Projects**: 1 private project
- **Team Members**: Personal use only
- **Run History**: Unlimited scalar metrics
- **Artifacts**: 100GB total

### Personal/Starter ($20/month)
- **Storage**: 100GB 
- **Private Projects**: Unlimited
- **Team Members**: 1 user
- **Features**: All core features
- **Artifacts**: 100GB

### Team ($50/month)
- **Storage**: 1TB
- **Private Projects**: Unlimited  
- **Team Members**: Up to 5 users
- **Advanced Features**: Advanced visualizations
- **Artifacts**: 1TB

### Enterprise (Custom pricing)
- **Storage**: Unlimited
- **Users**: Unlimited
- **Advanced Security**: SSO, audit logs
- **Support**: Priority support

## Recommendation for Your Use Case

### **Free Tier is Sufficient** ✅

Your chrono-membench experiment will use:
- **~1.7GB per training run** (well under 100GB limit)
- **Personal research project** (1 private project is enough)
- **Standard metrics logging** (covered by free tier)

### Optimization Tips for Free Tier

1. **Reduce Feature Storage Frequency**:
   ```yaml
   checkpointing:
     checkpoint_every_n_steps: 1000  # Instead of 500
     save_features: true
     save_sae_state: false  # Save only when needed
   ```

2. **Compress Feature Representations**:
   ```python
   # In your training script
   torch.save(features, path, _use_new_zipfile_serialization=True)
   ```

3. **Use Artifacts Efficiently**:
   ```python
   # Log large objects as artifacts, not files
   wandb.log_artifact(feature_artifact, type="features")
   ```

4. **Clean Up Old Runs**:
   - Delete experimental runs that aren't needed
   - Archive completed experiments

### When to Upgrade

**Upgrade to Personal ($20/month) if**:
- You want multiple private projects
- You're collaborating with others
- You need priority support

**Upgrade to Team ($50/month) if**:
- Working with a team (2-5 people)
- Need advanced visualizations
- Running multiple concurrent experiments
- Need more storage (approaching 100GB)

## Cost-Effective Training Strategy

### Phase 1: Free Tier Development
1. **Short runs** (1,000 steps) for debugging
2. **Minimal feature saving** for proof of concept
3. **Single model type** (Gemma-2B)

### Phase 2: Full Experiments
1. **Full 5,000 step runs** once setup is validated
2. **Complete feature logging** for analysis
3. **Ablation studies** if needed

### Phase 3: Scale Up
1. **Multiple model types** (Llama-3-8B)
2. **Longer training runs** (20,000+ steps)
3. **Team collaboration** (upgrade to Team tier)

## Monitoring and Alerts

Free tier includes:
- **Real-time monitoring** of all metrics
- **Email alerts** for run completion/failure
- **Mobile app** for monitoring on-the-go
- **Public sharing** of results

## Conclusion

**Your chrono-membench project is well-suited for WandB's free tier.**

- ✅ Datasets are manageable size (285MB)
- ✅ Training runs are reasonable length (5,000 steps)
- ✅ Storage requirements fit comfortably in 100GB limit
- ✅ Personal research project structure
- ✅ All core monitoring features available

Start with the **free tier** and upgrade only if you need team collaboration or exceed storage limits.
