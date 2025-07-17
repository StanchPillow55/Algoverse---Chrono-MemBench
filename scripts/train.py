#!/usr/bin/env python3
"""
Main training script for ChatGPT 2-0 / Chrono-MemBench
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatgpt2.chrono.chrono_train import main

if __name__ == "__main__":
    main()
