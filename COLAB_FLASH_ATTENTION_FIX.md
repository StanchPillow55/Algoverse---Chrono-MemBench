# Flash-Attention Fix for Google Colab

## Problem
Your Colab notebook is experiencing this error:
```
ImportError: /usr/local/lib/python3.11/dist-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

## Quick Fix for Your Current Notebook

Add this cell **immediately after cloning the repository** and **before installing dependencies**:

```python
# Fix Flash-Attention import issues
!python fix_flash_attention.py
```

## Step-by-Step Instructions

1. **In your current notebook**, after the cell where you clone the repository:
   ```python
   !git clone https://github.com/StanchPillow55/Algoverse---Chrono-MemBench.git
   %cd Algoverse---Chrono-MemBench
   ```

2. **Add a new cell** with:
   ```python
   # Fix Flash-Attention import issues
   !python fix_flash_attention.py
   ```

3. **Run this cell** - it will automatically:
   - Try to install a compatible pre-built Flash-Attention wheel
   - If that fails, try to compile from source
   - If that fails, disable Flash-Attention entirely

4. **After the fix completes successfully**, restart your kernel:
   - Runtime → Restart Runtime

5. **Then continue** with your normal dependency installation:
   ```python
   !pip install -r requirements_training.txt
   !pip install wandb accelerate
   ```

## What the Script Does

The `fix_flash_attention.py` script tries three approaches:

1. **Option A**: Install matching pre-built wheel for your PyTorch/CUDA versions
2. **Option B**: Compile Flash-Attention from source (if no matching wheel exists)
3. **Option C**: Disable Flash-Attention entirely (fallback option)

## For Future Notebooks

The updated template in `notebooks/chrono_colab_template.ipynb` now includes this fix automatically. When you clone the repository in future notebooks, the fix will be included.

## Verification

After running the fix script, you should see a message like:
```
🎉 Success! [Option X] worked.

📝 Next steps:
1. Restart your kernel/runtime
2. Re-run your chrono-membench training
```

If you see this message, the fix worked and you can proceed with your training.
