# Google Colab Notebook Links

## Direct Access Links

Click these links to open the notebooks directly in Google Colab:

### 🚀 Main Training Notebooks

1. **Chrono-MemBench Colab (Primary)**
   - **Link**: https://colab.research.google.com/github/StanchPillow55/Algoverse---Chrono-MemBench/blob/main/notebooks/chrono_membench_colab.ipynb
   - **Description**: Full chrono-membench training with all features
   - **Recommended for**: Complete training runs with temporal dropout and Route-SAE

2. **Chrono Colab Template (Updated with Flash-Attention Fix)**
   - **Link**: https://colab.research.google.com/github/StanchPillow55/Algoverse---Chrono-MemBench/blob/main/notebooks/chrono_colab_template.ipynb
   - **Description**: Clean template with Flash-Attention fix included
   - **Recommended for**: Starting fresh or troubleshooting

3. **Training Colab Template (Standard)**
   - **Link**: https://colab.research.google.com/github/StanchPillow55/Algoverse---Chrono-MemBench/blob/main/notebooks/training_colab_template.ipynb
   - **Description**: Standard training without chrono-specific features
   - **Recommended for**: Basic model training

### 📚 Demo Notebooks

4. **01 Colab Demo**
   - **Link**: https://colab.research.google.com/github/StanchPillow55/Algoverse---Chrono-MemBench/blob/main/notebooks/01_colab_demo.ipynb
   - **Description**: Basic demo notebook
   - **Status**: Currently empty/placeholder

## Manual Method (Alternative)

If you prefer to navigate manually:

1. Go to https://colab.research.google.com/
2. Click "GitHub" tab
3. Enter repository: `StanchPillow55/Algoverse---Chrono-MemBench`
4. Browse to `notebooks/` folder
5. Select your desired notebook

## Recommended Workflow

### For First-Time Users:
1. Start with **Chrono Colab Template** - it includes the Flash-Attention fix
2. Follow the step-by-step instructions in the notebook
3. The fix script will automatically resolve compatibility issues

### For Experienced Users:
1. Use **Chrono-MemBench Colab** for full feature training
2. Modify parameters as needed for your specific use case

## Important Notes

- **Always start with the Flash-Attention fix** if you encounter import errors
- **Restart your runtime** after running the fix script
- **Save your work frequently** to Google Drive to avoid data loss
- **Monitor GPU usage** - Colab has usage limits

## Repository Structure

```
notebooks/
├── chrono_membench_colab.ipynb        # Main training notebook
├── chrono_colab_template.ipynb        # Template with fixes
├── training_colab_template.ipynb      # Standard training
└── 01_colab_demo.ipynb               # Demo notebook
```

## Support

If you encounter issues:
1. Check `COLAB_FLASH_ATTENTION_FIX.md` for Flash-Attention problems
2. Refer to `CHRONO_TRAINING_README.md` for detailed training instructions
3. Check `TRAINING_README.md` for general training guidance
