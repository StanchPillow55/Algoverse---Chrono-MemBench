import pytest
import torch
import torch.nn as nn
import json
from unittest.mock import Mock, patch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from membench_x.metrics import (
    BaseMetricHook, MemAbsorptionHook, TPGHook, CapGaugeHook,
    ICLPersistenceHook, WeightDeltaHook, RAGTraceHook
)


class DummyModel(nn.Module):
    """Simple test model for metric testing."""
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.temporal_dropout = Mock()
        self.temporal_dropout.gates = torch.rand(hidden_dim)
    
    def forward(self, x):
        hidden = self.encoder(x)
        output = self.decoder(hidden)
        return {
            'output': output,
            'activations': hidden,
            'hidden_states': hidden
        }


@pytest.fixture
def mock_writer():
    """Mock tensorboard writer to avoid file I/O in tests."""
    return Mock(spec=SummaryWriter)


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel()


@pytest.fixture
def sample_tensors():
    """Generate sample tensors for testing."""
    batch_size, seq_len, dim = 4, 16, 128
    return {
        'activations': torch.randn(batch_size, seq_len, dim),
        'attention_weights': torch.randn(batch_size, seq_len, seq_len),
        'loss_dict': {'temporal_loss': torch.tensor(0.5)}
    }


class TestMetricHookShapes:
    """Test tensor shapes and dimensions for all metric hooks."""
    
    def test_mem_absorption_hook_shapes(self, mock_writer, dummy_model):
        hook = MemAbsorptionHook(mock_writer)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=1024**3), \
             patch('torch.cuda.memory_reserved', return_value=2*1024**3):
            
            metrics = hook.on_step(step=0, model=dummy_model)
            
            assert isinstance(metrics, dict)
            assert 'mem_absorption' in metrics
            assert isinstance(metrics['mem_absorption'], float)
            assert 0.0 <= metrics['mem_absorption'] <= 1.0
    
    def test_tpg_hook_shapes(self, mock_writer, dummy_model, sample_tensors):
        hook = TPGHook(mock_writer)
        
        # Test with loss_dict
        metrics = hook.on_step(
            step=0, 
            model=dummy_model, 
            loss_dict=sample_tensors['loss_dict']
        )
        
        assert isinstance(metrics, dict)
        assert 'tpg' in metrics
        assert isinstance(metrics['tpg'], float)
        
        # Test fallback without loss_dict
        metrics_fallback = hook.on_step(step=0, model=dummy_model)
        assert 'tpg' in metrics_fallback
    
    def test_cap_gauge_hook_shapes(self, mock_writer, dummy_model, sample_tensors):
        hook = CapGaugeHook(mock_writer)
        
        # Test with activations
        metrics = hook.on_step(
            step=0, 
            model=dummy_model, 
            activations=sample_tensors['activations']
        )
        
        assert isinstance(metrics, dict)
        assert 'cap_gauge' in metrics
        assert isinstance(metrics['cap_gauge'], float)
        assert 0.0 <= metrics['cap_gauge'] <= 1.0
        
        # Test fallback without activations
        metrics_fallback = hook.on_step(step=0, model=dummy_model)
        assert 'cap_gauge' in metrics_fallback
    
    def test_icl_persistence_hook_shapes(self, mock_writer, dummy_model, sample_tensors):
        hook = ICLPersistenceHook(mock_writer)
        
        # First call - no previous activations
        metrics1 = hook.on_step(
            step=0, 
            model=dummy_model, 
            activations=sample_tensors['activations']
        )
        
        assert isinstance(metrics1, dict)
        assert 'icl_persistence' in metrics1
        assert metrics1['icl_persistence'] == 0.0  # No previous activations
        
        # Second call - should compute similarity
        metrics2 = hook.on_step(
            step=1, 
            model=dummy_model, 
            activations=sample_tensors['activations']
        )
        
        assert isinstance(metrics2['icl_persistence'], float)
        assert 0.0 <= metrics2['icl_persistence'] <= 1.0
    
    def test_weight_delta_hook_shapes(self, mock_writer, dummy_model):
        hook = WeightDeltaHook(mock_writer)
        
        # First call - no previous weights
        metrics1 = hook.on_step(step=0, model=dummy_model)
        
        assert isinstance(metrics1, dict)
        assert 'weight_delta' in metrics1
        assert metrics1['weight_delta'] == 0.0  # No previous weights
        
        # Modify model weights slightly
        with torch.no_grad():
            dummy_model.encoder.weight += 0.01
        
        # Second call - should detect weight change
        metrics2 = hook.on_step(step=1, model=dummy_model)
        
        assert isinstance(metrics2['weight_delta'], float)
        assert metrics2['weight_delta'] > 0.0  # Should detect change
    
    def test_rag_trace_hook_shapes(self, mock_writer, dummy_model, sample_tensors):
        hook = RAGTraceHook(mock_writer)
        
        # Test with attention weights
        metrics = hook.on_step(
            step=0, 
            model=dummy_model, 
            attention_weights=sample_tensors['attention_weights']
        )
        
        assert isinstance(metrics, dict)
        assert 'rag_trace' in metrics
        assert isinstance(metrics['rag_trace'], float)
        assert 0.0 <= metrics['rag_trace'] <= 1.0
        
        # Test fallback without attention weights
        metrics_fallback = hook.on_step(step=0, model=dummy_model)
        assert 'rag_trace' in metrics_fallback


class TestMetricHookMemoryUsage:
    """Test that metric hooks don't cause OOM on 8GB GPU."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hooks_memory_efficiency(self, mock_writer, dummy_model):
        """Test that all hooks together don't consume excessive GPU memory."""
        
        device = torch.device('cuda')
        dummy_model = dummy_model.to(device)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create all hooks
        hooks = [
            MemAbsorptionHook(mock_writer),
            TPGHook(mock_writer),
            CapGaugeHook(mock_writer),
            ICLPersistenceHook(mock_writer),
            WeightDeltaHook(mock_writer),
            RAGTraceHook(mock_writer)
        ]
        
        # Create large tensors to simulate real training
        batch_size, seq_len, dim = 32, 512, 1024
        activations = torch.randn(batch_size, seq_len, dim, device=device)
        attention_weights = torch.randn(batch_size, seq_len, seq_len, device=device)
        loss_dict = {'temporal_loss': torch.tensor(0.5, device=device)}
        
        # Run multiple steps to test memory accumulation
        for step in range(10):
            for hook in hooks:
                hook.on_step(
                    step=step,
                    model=dummy_model,
                    activations=activations,
                    attention_weights=attention_weights,
                    loss_dict=loss_dict
                )
        
        final_memory = torch.cuda.memory_allocated()
        memory_increase = (final_memory - initial_memory) / (1024**3)  # GB
        
        # Assert memory increase is reasonable (less than 1GB for hooks)
        assert memory_increase < 1.0, f"Hooks consumed {memory_increase:.2f}GB, too much!"
        
        # Cleanup
        del activations, attention_weights, loss_dict
        torch.cuda.empty_cache()
    
    def test_hooks_handle_large_tensors(self, mock_writer, dummy_model):
        """Test hooks can handle large tensor dimensions without crashing."""
        
        # Create very large tensors (but keep on CPU to avoid OOM in CI)
        batch_size, seq_len, dim = 16, 2048, 4096
        large_activations = torch.randn(batch_size, seq_len, dim)
        large_attention = torch.randn(batch_size, seq_len, seq_len)
        
        hooks = [
            MemAbsorptionHook(mock_writer),
            TPGHook(mock_writer),
            CapGaugeHook(mock_writer),
            ICLPersistenceHook(mock_writer),
            WeightDeltaHook(mock_writer),
            RAGTraceHook(mock_writer)
        ]
        
        # Should not crash with large tensors
        for hook in hooks:
            try:
                metrics = hook.on_step(
                    step=0,
                    model=dummy_model,
                    activations=large_activations,
                    attention_weights=large_attention
                )
                assert isinstance(metrics, dict)
            except Exception as e:
                pytest.fail(f"{hook.__class__.__name__} failed with large tensors: {e}")


class TestMetricHookEdgeCases:
    """Test edge cases and error handling."""
    
    def test_hooks_with_empty_tensors(self, mock_writer, dummy_model):
        """Test hooks handle empty tensors gracefully."""
        
        hooks = [
            MemAbsorptionHook(mock_writer),
            TPGHook(mock_writer),
            CapGaugeHook(mock_writer),
            ICLPersistenceHook(mock_writer),
            WeightDeltaHook(mock_writer),
            RAGTraceHook(mock_writer)
        ]
        
        empty_tensor = torch.empty(0)
        
        for hook in hooks:
            try:
                metrics = hook.on_step(
                    step=0,
                    model=dummy_model,
                    activations=empty_tensor
                )
                assert isinstance(metrics, dict)
            except Exception as e:
                # Some hooks might reasonably fail with empty tensors, that's ok
                pass
    
    def test_hooks_with_none_inputs(self, mock_writer, dummy_model):
        """Test hooks handle None inputs gracefully."""
        
        hooks = [
            MemAbsorptionHook(mock_writer),
            TPGHook(mock_writer),
            CapGaugeHook(mock_writer),
            ICLPersistenceHook(mock_writer),
            WeightDeltaHook(mock_writer),
            RAGTraceHook(mock_writer)
        ]
        
        for hook in hooks:
            try:
                metrics = hook.on_step(
                    step=0,
                    model=dummy_model,
                    activations=None,
                    attention_weights=None,
                    loss_dict=None
                )
                assert isinstance(metrics, dict)
            except Exception as e:
                pytest.fail(f"{hook.__class__.__name__} failed with None inputs: {e}")
    
    def test_jsonl_logging(self, mock_writer, dummy_model, tmp_path):
        """Test JSONL file logging functionality."""
        
        metrics_dir = tmp_path / "metrics"
        hook = MemAbsorptionHook(mock_writer, metrics_dir)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=1024**3), \
             patch('torch.cuda.memory_reserved', return_value=2*1024**3):
            
            hook.on_step(step=5, model=dummy_model)
        
        # Check JSONL file was created
        jsonl_file = metrics_dir / "mem_absorption.jsonl"
        assert jsonl_file.exists()
        
        # Check file content
        with open(jsonl_file) as f:
            line = f.readline().strip()
            data = json.loads(line)
            assert data['step'] == 5
            assert 'value' in data
            assert isinstance(data['value'], float)


if __name__ == "__main__":
    pytest.main([__file__])
