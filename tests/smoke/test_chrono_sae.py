"""
Smoke tests for ChronoSAE model functionality.

Tests basic forward pass, loss computation, and gradient flow.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from algoverse.chrono.chrono_sae.model import ChronoSAE, ChronoSAEConfig, create_chrono_sae


class TestChronoSAE:
    """Test ChronoSAE model functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ChronoSAEConfig(
            d_model=768,
            d_sae=2048,
            temporal_dropout_p=0.1,
            lambda_sparsity=1e-4,
            beta_tpg=1e-3,
            device="cpu",  # Force CPU for testing
        )
    
    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return create_chrono_sae(config)
    
    def test_model_creation(self, config):
        """Test that model can be created."""
        model = create_chrono_sae(config)
        assert isinstance(model, ChronoSAE)
        assert model.config.d_model == 768
        assert model.config.d_sae == 2048
    
    def test_parameter_count(self, model):
        """Test parameter count is reasonable."""
        param_count = sum(p.numel() for p in model.parameters())
        # Should have encoder + decoder + gate + biases
        # Encoder: 768 * 2048 + 2048 = 1,574,912
        # Decoder: 2048 * 768 + 768 = 1,574,144  
        # Gate: (768 * 192 + 192) + (192 * 768 + 768) = 295,680
        # Decoder bias: 768
        # Total ~3.4M parameters
        assert 3_000_000 < param_count < 4_000_000
    
    def test_forward_pass_2d(self, model):
        """Test forward pass with 2D input (batch, d_model)."""
        batch_size = 4
        d_model = 768
        
        # Create dummy activations
        x = torch.randn(batch_size, d_model)
        
        # Forward pass
        outputs = model(x, compute_loss=True)
        
        # Check output shapes
        assert outputs['x_hat'].shape == (batch_size, d_model)
        assert outputs['z'].shape == (batch_size, 2048)
        assert outputs['gate_values'].shape == (batch_size, d_model)
        
        # Check loss components exist
        assert 'loss' in outputs
        assert 'loss_components' in outputs
        assert outputs['loss'].item() > 0
    
    def test_forward_pass_3d(self, model):
        """Test forward pass with 3D input (batch, seq_len, d_model)."""
        batch_size = 2
        seq_len = 16
        d_model = 768
        
        # Create dummy activations (3 checkpoints as mentioned in task)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        outputs = model(x, compute_loss=True)
        
        # Check output shapes
        assert outputs['x_hat'].shape == (batch_size, seq_len, d_model)
        assert outputs['z'].shape == (batch_size, seq_len, 2048)
        assert outputs['gate_values'].shape == (batch_size, seq_len, d_model)
        
        # Check loss components exist
        assert 'loss' in outputs
        assert 'loss_components' in outputs
        assert outputs['loss'].item() > 0
    
    def test_loss_components(self, model):
        """Test that all loss components are computed correctly."""
        batch_size = 2
        seq_len = 3  # 3 checkpoints as mentioned
        d_model = 768
        
        x = torch.randn(batch_size, seq_len, d_model)
        outputs = model(x, compute_loss=True)
        
        loss_components = outputs['loss_components']
        
        # Check all loss components exist and are positive
        assert 'mse_loss' in loss_components
        assert 'l1_loss' in loss_components
        assert 'tpg_loss' in loss_components
        assert 'total_loss' in loss_components
        
        # All losses should be non-negative
        assert loss_components['mse_loss'].item() >= 0
        assert loss_components['l1_loss'].item() >= 0
        assert loss_components['tpg_loss'].item() >= 0
        assert loss_components['total_loss'].item() >= 0
        
        # Total loss should be sum of components (approximately)
        expected_total = (loss_components['mse_loss'] + 
                         model.config.lambda_sparsity * loss_components['l1_loss'] +
                         model.config.beta_tpg * loss_components['tpg_loss'])
        
        assert torch.allclose(loss_components['total_loss'], expected_total, rtol=1e-5)
    
    def test_backward_pass_and_gradients(self, model):
        """Test backward pass and gradient computation."""
        batch_size = 2
        seq_len = 3  # 3 checkpoints as mentioned
        d_model = 768
        
        # Create dummy activations
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        outputs = model(x, compute_loss=True)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed and not None
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient is None for parameter: {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Gradient is zero for parameter: {name}"
    
    def test_temporal_dropout_effect(self, model):
        """Test that temporal dropout has an effect during training."""
        batch_size = 2
        seq_len = 8
        d_model = 768
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Set model to training mode
        model.train()
        
        # Run multiple forward passes - outputs should vary due to dropout
        outputs1 = model(x, compute_loss=False)
        outputs2 = model(x, compute_loss=False)
        
        gate1 = outputs1['gate_values']
        gate2 = outputs2['gate_values'] 
        
        # Gate values should be different due to temporal dropout
        assert not torch.allclose(gate1, gate2, rtol=1e-3), \
            "Temporal dropout should cause variation in gate values"
        
        # Set model to eval mode - should be deterministic
        model.eval()
        
        outputs3 = model(x, compute_loss=False)
        outputs4 = model(x, compute_loss=False)
        
        gate3 = outputs3['gate_values']
        gate4 = outputs4['gate_values']
        
        # Gate values should be identical in eval mode
        assert torch.allclose(gate3, gate4, rtol=1e-6), \
            "Gate values should be deterministic in eval mode"
    
    def test_sparsity_metrics(self, model):
        """Test sparsity metrics computation."""
        batch_size = 4
        d_model = 768
        
        x = torch.randn(batch_size, d_model)
        outputs = model(x, compute_loss=False)
        z = outputs['z']
        
        # Get sparsity metrics
        metrics = model.get_sparsity_metrics(z)
        
        # Check metrics exist and are reasonable
        assert 'active_fraction' in metrics
        assert 'l0_norm' in metrics
        assert 'l1_norm' in metrics
        assert 'l2_norm' in metrics
        
        # Active fraction should be between 0 and 1
        assert 0 <= metrics['active_fraction'] <= 1
        
        # L0 norm should be positive (number of active neurons)
        assert metrics['l0_norm'] >= 0
        
        # L1 and L2 norms should be positive
        assert metrics['l1_norm'] >= 0
        assert metrics['l2_norm'] >= 0
    
    def test_decoder_bias_update(self, model):
        """Test decoder bias updating mechanism."""
        batch_size = 4
        d_model = 768
        
        x = torch.randn(batch_size, d_model)
        
        # Get initial bias
        initial_bias = model.decoder_bias.clone()
        
        # Update decoder bias
        model.update_decoder_bias(x)
        
        # Bias should have changed
        assert not torch.allclose(initial_bias, model.decoder_bias, rtol=1e-6), \
            "Decoder bias should be updated"
    
    def test_model_modes(self, model):
        """Test model training and evaluation modes."""
        batch_size = 2
        d_model = 768
        
        x = torch.randn(batch_size, d_model)
        
        # Test training mode
        model.train()
        assert model.training
        outputs_train = model(x, compute_loss=False)
        
        # Test eval mode
        model.eval()
        assert not model.training
        outputs_eval = model(x, compute_loss=False)
        
        # Outputs should be different due to temporal dropout in training mode
        # but this test might be flaky due to randomness, so we just check shapes
        assert outputs_train['x_hat'].shape == outputs_eval['x_hat'].shape
        assert outputs_train['z'].shape == outputs_eval['z'].shape
        assert outputs_train['gate_values'].shape == outputs_eval['gate_values'].shape
    
    def test_device_compatibility(self):
        """Test model works on different devices."""
        # Test CPU
        config_cpu = ChronoSAEConfig(d_model=64, d_sae=128, device="cpu")
        model_cpu = create_chrono_sae(config_cpu)
        
        x_cpu = torch.randn(2, 64)
        outputs_cpu = model_cpu(x_cpu, compute_loss=True)
        assert outputs_cpu['x_hat'].device.type == "cpu"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            config_cuda = ChronoSAEConfig(d_model=64, d_sae=128, device="cuda")
            model_cuda = create_chrono_sae(config_cuda)
            
            x_cuda = torch.randn(2, 64, device="cuda")
            outputs_cuda = model_cuda(x_cuda, compute_loss=True)
            assert outputs_cuda['x_hat'].device.type == "cuda"
        
        # Test MPS if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config_mps = ChronoSAEConfig(d_model=64, d_sae=128, device="mps")
            model_mps = create_chrono_sae(config_mps)
            
            x_mps = torch.randn(2, 64, device="mps")
            outputs_mps = model_mps(x_mps, compute_loss=True)
            assert outputs_mps['x_hat'].device.type == "mps"
