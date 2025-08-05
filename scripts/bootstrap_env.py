#!/usr/bin/env python3
"""
Bootstrap script for Chrono-MemBench environment setup.

Automatically detects the platform and creates the appropriate conda environment:
- GPU environment for Windows/Linux with CUDA support
- CPU environment for macOS or systems without suitable GPU

Includes GPU capability and VRAM checks for optimal setup.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple

try:
    import rich
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    # Fallback if rich is not available
    rich = None
    Console = None

# Initialize console
console = Console() if Console else None


def print_msg(message: str, style: str = "info") -> None:
    """Print a message with optional styling."""
    if console:
        if style == "error":
            console.print(f"❌ {message}", style="bold red")
        elif style == "warning":
            console.print(f"⚠️  {message}", style="bold yellow")
        elif style == "success":
            console.print(f"✅ {message}", style="bold green")
        elif style == "info":
            console.print(f"ℹ️  {message}", style="bold blue")
        else:
            console.print(message)
    else:
        print(f"[{style.upper()}] {message}")


def run_command(cmd: list, capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=capture_output, 
            text=True, 
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def check_conda_available() -> bool:
    """Check if conda is available."""
    code, _, _ = run_command(["conda", "--version"])
    return code == 0


def detect_gpu_info() -> Tuple[bool, Optional[str], Optional[float], Optional[float]]:
    """
    Detect GPU information.
    
    Returns:
        (has_cuda, gpu_name, compute_capability, vram_gb)
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, None, None, None
        
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        compute_capability = props.major + props.minor / 10.0
        vram_gb = props.total_memory / (1024**3)
        
        return True, gpu_name, compute_capability, vram_gb
        
    except ImportError:
        # PyTorch not available, try nvidia-smi
        code, stdout, _ = run_command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        
        if code == 0 and stdout.strip():
            lines = stdout.strip().split('\n')
            if lines and lines[0]:
                parts = lines[0].split(', ')
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    try:
                        vram_mb = float(parts[1].strip())
                        vram_gb = vram_mb / 1024
                        
                        # Estimate compute capability based on GPU name
                        compute_capability = estimate_compute_capability(gpu_name)
                        
                        return True, gpu_name, compute_capability, vram_gb
                    except ValueError:
                        pass
        
        return False, None, None, None


def estimate_compute_capability(gpu_name: str) -> float:
    """Estimate compute capability based on GPU name."""
    gpu_name_lower = gpu_name.lower()
    
    # GTX 1070 is compute capability 6.1
    if "gtx 1070" in gpu_name_lower:
        return 6.1
    elif "gtx 1080" in gpu_name_lower:
        return 6.1
    elif "gtx 1060" in gpu_name_lower:
        return 6.1
    elif "rtx 20" in gpu_name_lower:
        return 7.5
    elif "rtx 30" in gpu_name_lower:
        return 8.6
    elif "rtx 40" in gpu_name_lower:
        return 8.9
    elif "tesla" in gpu_name_lower:
        return 7.0  # Conservative estimate
    else:
        return 6.0  # Conservative fallback


def determine_environment_type() -> str:
    """Determine which environment to create based on platform and GPU."""
    system = platform.system()
    
    print_msg(f"Detected platform: {system} {platform.machine()}")
    
    # Check for GPU
    has_cuda, gpu_name, compute_capability, vram_gb = detect_gpu_info()
    
    if has_cuda and gpu_name:
        print_msg(f"GPU detected: {gpu_name}")
        print_msg(f"Compute capability: {compute_capability:.1f}")
        print_msg(f"VRAM: {vram_gb:.1f} GB")
        
        # Check compute capability
        if compute_capability and compute_capability < 6.1:
            print_msg(
                f"Warning: GPU compute capability {compute_capability:.1f} is below recommended 6.1. "
                "Some optimizations may not work.", 
                "warning"
            )
        
        # Check VRAM
        if vram_gb and vram_gb < 7.0:
            print_msg(
                f"Error: GPU VRAM {vram_gb:.1f} GB is below minimum requirement of 7 GB. "
                "GPU training will likely fail due to memory constraints.",
                "error"
            )
            print_msg("Consider using the CPU environment instead.", "info")
            
            response = input("Continue with GPU environment anyway? (y/N): ").lower().strip()
            if response not in ['y', 'yes']:
                print_msg("Exiting due to insufficient VRAM.", "error")
                sys.exit(1)
        
        # Use GPU environment for Windows/Linux with sufficient GPU
        if system in ["Windows", "Linux"] and compute_capability and compute_capability >= 6.1:
            return "gpu"
    
    # Fallback to CPU environment
    if system == "Darwin":
        print_msg("macOS detected - using CPU environment (MPS acceleration available)")
    else:
        print_msg("No suitable GPU detected - using CPU environment")
    
    return "cpu"


def create_conda_environment(env_type: str) -> bool:
    """Create the conda environment."""
    env_name = f"chrono_membench_{env_type}"
    
    # Choose the correct environment file
    if env_type == "gpu":
        env_file = Path(__file__).parent.parent / "environment-gpu.yml"
    else:
        env_file = Path(__file__).parent.parent / "environment.yml"
    
    if not env_file.exists():
        print_msg(f"Environment file not found: {env_file}", "error")
        return False
    
    print_msg(f"Creating conda environment: {env_name}")
    
    # Check if environment already exists
    code, stdout, _ = run_command(["conda", "env", "list"])
    if code == 0 and env_name in stdout:
        print_msg(f"Environment {env_name} already exists", "warning")
        response = input("Remove and recreate? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            print_msg(f"Removing existing environment: {env_name}")
            code, _, stderr = run_command(["conda", "env", "remove", "-n", env_name, "-y"])
            if code != 0:
                print_msg(f"Failed to remove environment: {stderr}", "error")
                return False
        else:
            print_msg("Using existing environment", "info")
            return True
    
    # Create environment
    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Creating conda environment...", total=None)
            
            code, stdout, stderr = run_command([
                "conda", "env", "create", 
                "-f", str(env_file),
                "-n", env_name
            ], capture_output=True)
    else:
        print("Creating conda environment (this may take several minutes)...")
        code, stdout, stderr = run_command([
            "conda", "env", "create", 
            "-f", str(env_file),
            "-n", env_name
        ], capture_output=False)
    
    if code != 0:
        print_msg(f"Failed to create environment: {stderr}", "error")
        return False
    
    print_msg(f"Successfully created environment: {env_name}", "success")
    return True


def main():
    """Main bootstrap function."""
    if console:
        console.print(Panel.fit(
            Text("Chrono-MemBench Environment Bootstrap", style="bold magenta"),
            border_style="blue"
        ))
    else:
        print("=== Chrono-MemBench Environment Bootstrap ===")
    
    # Check conda availability
    if not check_conda_available():
        print_msg("Conda not found. Please install Anaconda or Miniconda first.", "error")
        print_msg("Download from: https://docs.conda.io/en/latest/miniconda.html", "info")
        sys.exit(1)
    
    # Determine environment type
    env_type = determine_environment_type()
    print_msg(f"Selected environment type: {env_type}")
    
    # Create environment
    if create_conda_environment(env_type):
        env_name = f"chrono_membench_{env_type}"
        print_msg("Environment setup complete!", "success")
        print_msg(f"To activate: conda activate {env_name}", "info")
        
        # Additional setup instructions
        if console:
            console.print(Panel(
                f"Next steps:\n"
                f"1. conda activate {env_name}\n"
                f"2. python -m pip install -e .\n"
                f"3. pre-commit install\n"
                f"4. pytest tests/smoke/",
                title="Setup Complete",
                border_style="green"
            ))
        else:
            print("\nNext steps:")
            print(f"1. conda activate {env_name}")
            print("2. python -m pip install -e .")
            print("3. pre-commit install")
            print("4. pytest tests/smoke/")
    else:
        print_msg("Environment setup failed!", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()
