#!/usr/bin/env python3
"""
Fix Flash-Attention import error in Google Colab
This script implements the fixes for the Flash-Attention ImportError with undefined symbols.
"""

import subprocess
import sys
import os
import torch

def check_current_versions():
    """Check current PyTorch and CUDA versions."""
    print("🔍 Checking current versions...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print("-" * 50)

def fix_flash_attention_option_a():
    """Option A: Install matching pre-built wheel"""
    print("🔧 Option A: Installing matching pre-built wheel...")
    
    # Uninstall existing flash-attn
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "flash-attn"], 
                   capture_output=True)
    
    # Install compatible version
    # For PyTorch 2.3/2.4 with CUDA 12.1
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "flash-attn==2.5.1", 
        "--no-build-isolation",
        "--extra-index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Flash-Attention installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install flash-attn wheel: {e}")
        print("Stderr:", e.stderr)
        return False

def fix_flash_attention_option_b():
    """Option B: Re-compile Flash-Attention from source"""
    print("🔧 Option B: Re-compiling Flash-Attention from source...")
    
    # Uninstall existing flash-attn
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "flash-attn"], 
                   capture_output=True)
    
    # Install dependencies for compilation
    subprocess.run([sys.executable, "-m", "pip", "install", "ninja"], 
                   capture_output=True)
    
    # Clone and compile
    commands = [
        ["git", "clone", "https://github.com/Dao-AILab/flash-attention.git"],
        ["pip", "install", "./flash-attention", "--no-build-isolation"]
    ]
    
    try:
        for cmd in commands:
            subprocess.run(cmd, check=True, cwd="/tmp")
        print("✅ Flash-Attention compiled successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to compile flash-attn: {e}")
        return False

def fix_flash_attention_option_c():
    """Option C: Disable Flash-Attention"""
    print("🔧 Option C: Disabling Flash-Attention...")
    
    # Set environment variable to disable flash attention
    os.environ['DISABLE_FLASH_ATTN'] = '1'
    os.environ['FLASH_ATTN_DISABLE'] = '1'
    
    # Uninstall flash-attn to prevent import issues
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "flash-attn"], 
                   capture_output=True)
    
    print("✅ Flash-Attention disabled")
    print("Environment variables set:")
    print(f"  DISABLE_FLASH_ATTN={os.environ.get('DISABLE_FLASH_ATTN')}")
    print(f"  FLASH_ATTN_DISABLE={os.environ.get('FLASH_ATTN_DISABLE')}")
    return True

def test_transformers_import():
    """Test if transformers can be imported without flash-attn issues."""
    print("🧪 Testing transformers import...")
    try:
        import transformers
        print(f"✅ transformers imported successfully (version: {transformers.__version__})")
        return True
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False

def main():
    """Main function to fix Flash-Attention issues."""
    print("🚀 Flash-Attention Fix Script")
    print("=" * 50)
    
    check_current_versions()
    
    # Try fixes in order of preference
    fixes = [
        ("Option A: Install matching pre-built wheel", fix_flash_attention_option_a),
        ("Option B: Re-compile from source", fix_flash_attention_option_b),
        ("Option C: Disable Flash-Attention", fix_flash_attention_option_c)
    ]
    
    for fix_name, fix_func in fixes:
        print(f"\n🔄 Trying {fix_name}...")
        if fix_func():
            if test_transformers_import():
                print(f"\n🎉 Success! {fix_name} worked.")
                print("\n📝 Next steps:")
                print("1. Restart your kernel/runtime")
                print("2. Re-run your chrono-membench training")
                break
            else:
                print(f"❌ {fix_name} didn't resolve the import issue")
        else:
            print(f"❌ {fix_name} failed")
    else:
        print("\n❌ All fixes failed. Please check your environment manually.")

if __name__ == "__main__":
    main()
