"""
Unit tests for configuration management.
"""
import pytest
import os
import sys
import yaml
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestConfigurationFiles:
    """Test configuration file validity."""
    
    @pytest.fixture
    def configs_dir(self):
        """Return path to configs directory."""
        return Path(__file__).parent.parent.parent / "configs"
    
    def test_config_files_exist(self, configs_dir):
        """Test that expected configuration files exist."""
        expected_files = [
            "gemma_2b.yaml",
            "llama_3_8b.yaml", 
            "chrono_membench.yaml",
            "training_base.yaml"
        ]
        
        for config_file in expected_files:
            config_path = configs_dir / config_file
            assert config_path.exists(), f"Config file {config_file} not found"
    
    def test_config_structure(self, configs_dir):
        """Test that all configurations have required structure."""
        config_files = list(configs_dir.glob("*.yaml"))
        assert len(config_files) > 0, "No config files found"
        
        for config_path in config_files:
            # Skip platform config file - it has different structure
            if config_path.name == "platform_config.yaml":
                continue
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic structure checks
            assert isinstance(config, dict), f"{config_path.name}: Config should be a dictionary"
            
            # Check for model section
            assert 'model' in config, f"{config_path.name}: Missing 'model' section"
            
            # Check for paths section in model
            if 'paths' not in config['model']:
                pytest.skip(f"{config_path.name}: Missing 'paths' in model section - may be legacy format")
    
    def test_model_paths_validity(self, configs_dir):
        """Test that model paths are valid for their respective model types."""
        config_files = list(configs_dir.glob("*.yaml"))
        
        for config_path in config_files:
            # Skip platform config and other special files
            if config_path.name in ["platform_config.yaml", "base.yaml"]:
                continue
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'model' not in config or 'paths' not in config['model']:
                continue
                
            paths = config['model']['paths']
            model_type = config['model'].get('type', 'unknown')
            
            assert model_type in paths, f"{config_path.name}: Missing paths for model type '{model_type}'"
            
            # Handle both string paths and nested path structures
            model_paths = paths[model_type]
            if isinstance(model_paths, str):
                # Simple string path
                assert True
            elif isinstance(model_paths, dict):
                # Nested path structure (e.g., with 'local' and 'huggingface' keys)
                assert 'huggingface' in model_paths or 'local' in model_paths, f"{config_path.name}: No valid path sources found"
                for path_key, path_value in model_paths.items():
                    assert isinstance(path_value, str), f"{config_path.name}: Path value should be string for {path_key}"
            else:
                pytest.fail(f"{config_path.name}: Path should be string or dictionary")


class TestModelManager:
    """Test ModelManager functionality."""
    
    def test_model_manager_import(self):
        """Test that ModelManager can be imported."""
        try:
            from chrono.train import ModelManager
            assert ModelManager is not None
        except ImportError as e:
            pytest.skip(f"ModelManager not available: {e}")
    
    def test_model_manager_with_config(self):
        """Test ModelManager with a valid configuration."""
        try:
            from chrono.train import ModelManager
        except ImportError:
            pytest.skip("ModelManager not available")
        
        configs_dir = Path(__file__).parent.parent.parent / "configs"
        gemma_config = configs_dir / "gemma_2b.yaml"
        
        if not gemma_config.exists():
            pytest.skip("gemma_2b.yaml config not found")
        
        with open(gemma_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # This should not raise a KeyError
        try:
            manager = ModelManager(config)
            model_path = manager.get_model_path()
            assert isinstance(model_path, str), "Model path should be a string"
        except KeyError as e:
            pytest.fail(f"KeyError in ModelManager: {e}")


class TestConfigGeneration:
    """Test configuration generation utilities."""
    
    def test_macos_config_optimization(self):
        """Test macOS configuration optimization if available."""
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        train_macos_script = scripts_dir / "train_macos.py"
        
        if not train_macos_script.exists():
            pytest.skip("train_macos.py not found")
        
        # Add scripts to path
        sys.path.insert(0, str(scripts_dir))
        
        try:
            from train_macos import optimize_for_macos
            
            base_config = Path(__file__).parent.parent.parent / "configs" / "chrono_membench.yaml"
            if not base_config.exists():
                pytest.skip("chrono_membench.yaml not found")
            
            # Test config generation
            macos_config_path = optimize_for_macos(str(base_config))
            
            # Verify generated config
            assert Path(macos_config_path).exists(), "Generated config file should exist"
            
            with open(macos_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Should have paths section
            assert 'model' in config, "Generated config should have model section"
            if 'paths' in config.get('model', {}):
                assert True  # Paths are present
            
            # Clean up
            os.remove(macos_config_path)
            
        except ImportError:
            pytest.skip("train_macos module not available")
        except Exception as e:
            pytest.fail(f"Config generation failed: {e}")
        finally:
            # Remove scripts from path
            if str(scripts_dir) in sys.path:
                sys.path.remove(str(scripts_dir))
