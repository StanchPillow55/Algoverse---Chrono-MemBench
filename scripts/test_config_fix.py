#!/usr/bin/env python3
"""
Test script to verify that the configuration KeyError: 'paths' fix works.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_config_paths():
    """Test that all configurations have the required paths section."""
    configs_dir = Path("configs")
    test_results = []
    
    print("🧪 Testing Configuration Files for KeyError: 'paths' Fix")
    print("=" * 60)
    
    config_files = [
        "gemma_2b.yaml",
        "llama_3_8b.yaml", 
        "chrono_membench.yaml",
        "training_base.yaml"
    ]
    
    for config_file in config_files:
        config_path = configs_dir / config_file
        if not config_path.exists():
            print(f"❌ {config_file}: File not found")
            test_results.append(False)
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if model section exists
            if 'model' not in config:
                print(f"❌ {config_file}: Missing 'model' section")
                test_results.append(False)
                continue
            
            # Check if paths section exists
            if 'paths' not in config['model']:
                print(f"❌ {config_file}: Missing 'paths' in model section")
                test_results.append(False)
                continue
            
            # Check if required model types are in paths
            paths = config['model']['paths']
            model_type = config['model'].get('type', 'unknown')
            
            if model_type in paths:
                print(f"✅ {config_file}: Has paths for '{model_type}'")
                test_results.append(True)
            else:
                print(f"❌ {config_file}: Missing paths for model type '{model_type}'")
                test_results.append(False)
                
        except Exception as e:
            print(f"❌ {config_file}: Error loading - {e}")
            test_results.append(False)
    
    return all(test_results)

def test_model_manager():
    """Test that ModelManager can load configurations without KeyError."""
    print("\n🔧 Testing ModelManager with Fixed Configurations")
    print("=" * 60)
    
    try:
        from chrono.train import ModelManager
        
        # Test with gemma_2b config
        print("Testing ModelManager with gemma_2b config...")
        with open('configs/gemma_2b.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        manager = ModelManager(config)
        model_path = manager.get_model_path()
        print(f"✅ ModelManager.get_model_path() = {model_path}")
        
        return True
        
    except KeyError as e:
        print(f"❌ KeyError still occurs: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_config_generation():
    """Test the config generation functions."""
    print("\n⚙️  Testing Configuration Generation")
    print("=" * 60)
    
    try:
        # Test macOS config generation
        sys.path.insert(0, str(Path(__file__).parent))
        from train_macos import optimize_for_macos
        
        print("Testing macOS config optimization...")
        macos_config = optimize_for_macos('configs/chrono_membench.yaml')
        
        # Verify generated config
        with open(macos_config, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'paths' in config.get('model', {}):
            print("✅ macOS config generation includes paths")
            # Clean up
            os.remove(macos_config)
            return True
        else:
            print("❌ macOS config generation missing paths")
            return False
            
    except Exception as e:
        print(f"❌ Config generation error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running Configuration Fix Tests\n")
    
    # Run all tests
    tests = [
        ("Configuration Files", test_config_paths),
        ("ModelManager Loading", test_model_manager), 
        ("Config Generation", test_config_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    all_passed = all(results)
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} {test_name}")
    
    if all_passed:
        print("\n🎉 All tests passed! The KeyError: 'paths' fix is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. The fix may need additional work.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
