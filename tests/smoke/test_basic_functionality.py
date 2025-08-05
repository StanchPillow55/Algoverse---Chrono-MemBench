"""
Smoke tests for basic functionality.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_imports():
    """Test that basic imports work."""
    try:
        import torch
        import transformers
        import yaml
        assert True
    except ImportError as e:
        pytest.fail(f"Basic import failed: {e}")


def test_torch_available():
    """Test that PyTorch is available and working."""
    import torch
    
    # Test basic tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(3, 2)
    z = torch.mm(x, y)
    
    assert z.shape == (2, 2)
    assert torch.is_tensor(z)


def test_config_loading():
    """Test that configuration files can be loaded."""
    import yaml
    
    configs_dir = Path(__file__).parent.parent.parent / "configs"
    
    # Test loading at least one config file
    config_files = list(configs_dir.glob("*.yaml"))
    assert len(config_files) > 0, "No config files found"
    
    # Try to load the first config
    with open(config_files[0], 'r') as f:
        config = yaml.safe_load(f)
    
    assert isinstance(config, dict), "Config should be a dictionary"


def test_algoverse_chrono_imports():
    """Test that algoverse.chrono modules can be imported."""
    try:
        from algoverse.chrono import chrono_sae
        from algoverse.chrono import membench_x
        from algoverse.chrono import training
        
        # Test chrono_sae imports
        assert chrono_sae.__version__ == "0.1.0"
        assert hasattr(chrono_sae, 'models')
        assert hasattr(chrono_sae, 'training')
        assert hasattr(chrono_sae, 'analysis')
        assert hasattr(chrono_sae, 'utils')
        assert hasattr(chrono_sae, 'config')
        
        # Test other modules exist
        assert membench_x.__version__ == "0.1.0"
        assert training.__version__ == "0.1.0"
        
    except ImportError as e:
        pytest.fail(f"Algoverse chrono imports failed: {e}")


def test_gpu_detection():
    """Test GPU detection (should work on both CPU and GPU systems)."""
    import torch
    
    # This should not raise an error
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    
    # Should be boolean
    assert isinstance(cuda_available, bool)
    # Should be non-negative integer
    assert isinstance(device_count, int) and device_count >= 0
    
    # On macOS, we expect CUDA to be unavailable
    import platform
    if platform.system() == "Darwin":
        assert not cuda_available
        assert device_count == 0
