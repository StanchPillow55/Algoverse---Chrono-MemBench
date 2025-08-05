"""
ChronoSAE: Sparse Autoencoder with Temporal Dropout for Memory Feature Analysis.

Implements the Chrono-SAE architecture from §3.1 of the paper, including:
- Standard SAE encoder/decoder
- Temporal dropout gate mechanism
- TPG (Temporal Pattern Gauge) loss component
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChronoSAEConfig:
    """Configuration for ChronoSAE model."""
    d_model: int = 768  # Input dimension
    d_sae: int = 2048   # SAE hidden dimension (typically 2-4x d_model)
    temporal_dropout_p: float = 0.1  # Temporal dropout probability
    lambda_sparsity: float = 1e-4    # L1 sparsity coefficient
    beta_tpg: float = 1e-3           # TPG loss coefficient
    device: str = "auto"             # Device selection
    dtype: torch.dtype = torch.float32


class TemporalDropoutGate(nn.Module):
    """
    Temporal dropout gate that selectively masks features across time steps.
    
    Unlike standard dropout which is random, this gate learns to identify
    and suppress temporal patterns based on the temporal context.
    """
    
    def __init__(self, d_model: int, dropout_p: float = 0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, temporal_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal dropout gate.
        
        Args:
            x: Input tensor [batch, seq_len, d_model] or [batch, d_model]
            temporal_context: Optional context for gate computation
            
        Returns:
            gated_x: Gated output
            gate_values: Gate activation values for TPG computation
        """
        if temporal_context is None:
            temporal_context = x
            
        # Compute gate values
        gate_values = self.gate(temporal_context)
        
        # Apply dropout probability during training
        if self.training:
            # Temporal dropout: create structured masks
            dropout_mask = torch.bernoulli(torch.full_like(gate_values, 1 - self.dropout_p))
            gate_values = gate_values * dropout_mask
        
        # Apply gating
        gated_x = x * gate_values
        
        return gated_x, gate_values


class ChronoSAE(nn.Module):
    """
    Chrono-SAE: Sparse Autoencoder with Temporal Dropout.
    
    Architecture:
    1. Temporal dropout gate (optional)
    2. SAE encoder: x -> z (sparse)
    3. SAE decoder: z -> x_hat
    4. Loss: MSE + λ||z||₁ + β·TPG
    """
    
    def __init__(self, config: ChronoSAEConfig):
        super().__init__()
        self.config = config
        
        # Core SAE components
        self.encoder = nn.Linear(config.d_model, config.d_sae)
        self.decoder = nn.Linear(config.d_sae, config.d_model)
        
        # Temporal dropout gate
        self.temporal_gate = TemporalDropoutGate(
            config.d_model, 
            config.temporal_dropout_p
        )
        
        # Bias terms
        self.decoder_bias = nn.Parameter(torch.zeros(config.d_model))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following SAE best practices."""
        # Xavier/Glorot initialization for encoder
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Decoder weights should be transpose of encoder for perfect reconstruction
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
        nn.init.zeros_(self.decoder.bias)
        
        # Decoder bias initialized to mean activation
        # (will be updated during training)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation."""
        # Linear transformation
        z_pre = self.encoder(x)
        
        # ReLU activation for sparsity
        z = F.relu(z_pre)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to input space."""
        x_hat = self.decoder(z) + self.decoder_bias
        return x_hat
    
    def forward(self, x: torch.Tensor, compute_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ChronoSAE.
        
        Args:
            x: Input activations [batch, seq_len, d_model] or [batch, d_model]
            compute_loss: Whether to compute and return loss components
            
        Returns:
            Dictionary containing:
            - x_hat: Reconstructed input
            - z: Sparse latent representation
            - gate_values: Temporal gate activations
            - loss: Total loss (if compute_loss=True)
            - loss_components: Individual loss terms (if compute_loss=True)
        """
        batch_shape = x.shape[:-1]  # Handle both 2D and 3D inputs
        d_model = x.shape[-1]
        
        # Flatten for processing if needed
        x_flat = x.view(-1, d_model)
        
        # Apply temporal dropout gate
        x_gated, gate_values = self.temporal_gate(x_flat)
        
        # SAE forward pass
        z = self.encode(x_gated)
        x_hat = self.decode(z)
        
        # Reshape back to original shape
        x_hat = x_hat.view(*batch_shape, d_model)
        z = z.view(*batch_shape, self.config.d_sae)
        gate_values = gate_values.view(*batch_shape, d_model)
        
        outputs = {
            'x_hat': x_hat,
            'z': z,
            'gate_values': gate_values,
        }
        
        if compute_loss:
            loss_components = self.compute_loss(x, x_hat, z, gate_values)
            outputs['loss'] = loss_components['total_loss']
            outputs['loss_components'] = loss_components
            
        return outputs
    
    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, 
                    z: torch.Tensor, gate_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Chrono-SAE loss components.
        
        Loss = MSE + λ||z||₁ + β·TPG
        
        Where:
        - MSE: Reconstruction loss
        - λ||z||₁: L1 sparsity penalty on latent codes
        - β·TPG: Temporal Pattern Gauge (measures temporal consistency)
        """
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(x_hat, x, reduction='mean')
        
        # Sparsity loss (L1 on latent codes)
        l1_loss = torch.mean(torch.abs(z))
        
        # TPG loss: measures temporal inconsistency in gate values
        tpg_loss = self.compute_tpg_loss(gate_values)
        
        # Total loss
        total_loss = (mse_loss + 
                     self.config.lambda_sparsity * l1_loss + 
                     self.config.beta_tpg * tpg_loss)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'l1_loss': l1_loss,
            'tpg_loss': tpg_loss,
        }
    
    def compute_tpg_loss(self, gate_values: torch.Tensor) -> torch.Tensor:
        """
        Compute Temporal Pattern Gauge (TPG) loss.
        
        TPG measures the temporal consistency of gate activations.
        High TPG indicates the gate is learning meaningful temporal patterns.
        
        Args:
            gate_values: Gate activations [..., d_model]
            
        Returns:
            TPG loss scalar
        """
        if gate_values.dim() < 3:
            # No temporal dimension, return zero TPG
            return torch.tensor(0.0, device=gate_values.device, dtype=gate_values.dtype)
        
        # Compute temporal differences
        # gate_values shape: [batch, seq_len, d_model]
        temporal_diff = gate_values[:, 1:] - gate_values[:, :-1]
        
        # TPG is the variance of temporal differences
        # High variance = inconsistent gating = high TPG loss
        tpg_loss = torch.var(temporal_diff, dim=(1, 2)).mean()
        
        return tpg_loss
    
    def get_sparsity_metrics(self, z: torch.Tensor) -> Dict[str, float]:
        """Compute sparsity metrics for monitoring."""
        with torch.no_grad():
            # Fraction of active neurons
            active_fraction = (z > 1e-6).float().mean().item()
            
            # L0 norm (number of active neurons)
            l0_norm = (z > 1e-6).sum(dim=-1).float().mean().item()
            
            # L1 norm
            l1_norm = torch.abs(z).sum(dim=-1).mean().item()
            
            # L2 norm
            l2_norm = torch.norm(z, dim=-1).mean().item()
            
        return {
            'active_fraction': active_fraction,
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
        }
    
    def update_decoder_bias(self, x: torch.Tensor, momentum: float = 0.99):
        """Update decoder bias to match input mean (running average)."""
        with torch.no_grad():
            batch_mean = x.mean(dim=tuple(range(x.dim()-1)))  # Mean over all but last dim
            if not hasattr(self, '_bias_initialized'):
                self.decoder_bias.copy_(batch_mean)
                self._bias_initialized = True
            else:
                self.decoder_bias.mul_(momentum).add_(batch_mean, alpha=1-momentum)


def create_chrono_sae(config: ChronoSAEConfig) -> ChronoSAE:
    """Factory function to create ChronoSAE model."""
    # Auto-detect device if needed
    if config.device == "auto":
        if torch.cuda.is_available():
            config.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config.device = "mps"
        else:
            config.device = "cpu"
    
    model = ChronoSAE(config)
    model = model.to(config.device)
    
    return model
