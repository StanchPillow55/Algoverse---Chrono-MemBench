"""
Basic integration tests for the Chrono-MemBench pipeline.
"""
import pytest
import sys
import torch
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestBasicPipeline:
    """Test basic pipeline integration."""
    
    def test_chrono_sae_import(self):
        """Test that chrono_sae package can be imported."""
        try:
            import chrono_sae
            assert chrono_sae.__version__ == "0.1.0"
            assert hasattr(chrono_sae, 'models')
            assert hasattr(chrono_sae, 'training')
            assert hasattr(chrono_sae, 'analysis')
            assert hasattr(chrono_sae, 'utils')
            assert hasattr(chrono_sae, 'config')
        except ImportError as e:
            pytest.fail(f"Failed to import chrono_sae: {e}")
    
    def test_platform_detection(self):
        """Test platform-specific setup detection."""
        import platform
        
        system = platform.system()
        assert system in ["Darwin", "Windows", "Linux"], f"Unsupported system: {system}"
        
        if system == "Darwin":
            # macOS - should not have CUDA
            assert not torch.cuda.is_available()
            # May have MPS on M1/M2 Macs
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            # Either MPS or CPU should be available
            assert mps_available or torch.get_num_threads() > 0
        
        elif system in ["Windows", "Linux"]:
            # May have CUDA available - test if present
            if torch.cuda.is_available():
                assert torch.cuda.device_count() > 0
                # Test basic CUDA operations
                device = torch.device("cuda:0")
                x = torch.randn(2, 3, device=device)
                y = torch.randn(3, 2, device=device)
                z = torch.mm(x, y)
                assert z.device == device
    
    def test_config_platform_compatibility(self):
        """Test that platform configurations are compatible."""
        import yaml
        
        platform_config_path = Path(__file__).parent.parent.parent / "configs" / "platform_config.yaml"
        
        if not platform_config_path.exists():
            pytest.skip("platform_config.yaml not found")
        
        with open(platform_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check that platform configs exist
        assert 'macos' in config, "macOS configuration missing"
        assert 'windows_gtx1070' in config, "Windows GTX 1070 configuration missing"
        
        # Validate macOS config
        macos_config = config['macos']
        assert macos_config['device'] in ['mps', 'cpu'], "Invalid macOS device"
        assert not macos_config['use_flash_attention'], "Flash attention should be disabled on macOS"
        assert not macos_config['use_triton'], "Triton should be disabled on macOS"
        
        # Validate Windows config
        windows_config = config['windows_gtx1070']
        assert windows_config['device'] == 'cuda', "Windows config should use CUDA"
        assert windows_config['use_flash_attention'], "Flash attention should be enabled on Windows"
        assert windows_config['gradient_checkpointing'], "Gradient checkpointing should be enabled for GTX 1070"
    
    @pytest.mark.slow
    def test_minimal_training_setup(self):
        """Test that minimal training setup works without errors."""
        # This is a placeholder for future training integration tests
        # For now, just test that we can set up the basic components
        
        import platform
        system = platform.system()
        
        if system == "Darwin":
            # macOS - use MPS or CPU
            if torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            # Windows/Linux - use CUDA if available, else CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test basic tensor operations on the selected device
        x = torch.randn(10, 768, device=device)  # Typical hidden dim
        
        # Simple linear layer (SAE-like)
        linear = torch.nn.Linear(768, 768).to(device)
        output = linear(x)
        
        assert output.shape == x.shape
        assert output.device.type == device.type
        
        # Test that we can compute gradients
        loss = torch.nn.functional.mse_loss(output, x)
        loss.backward()
        
        # Check that gradients exist
        assert linear.weight.grad is not None
        assert linear.bias.grad is not None
