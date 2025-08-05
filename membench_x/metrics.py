import torch
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter


class BaseMetricHook(ABC):
    """Base class for metric hooks that stream dial values during training."""
    
    def __init__(self, writer: SummaryWriter, jsonl_dir: Optional[Path] = None):
        self.writer = writer
        self.jsonl_dir = jsonl_dir
        self._device_cache = {}
        
        # Create metrics directory if specified
        if self.jsonl_dir:
            self.jsonl_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def on_step(self, step: int, model: torch.nn.Module, **kwargs) -> Dict[str, float]:
        """Compute metrics and log to tensorboard/jsonl. Returns dict of metric values."""
        pass
    
    def _log_scalar(self, name: str, value: float, step: int):
        """Log scalar to tensorboard and optionally to jsonl."""
        self.writer.add_scalar(name, value, step)
        
        if self.jsonl_dir:
            jsonl_path = self.jsonl_dir / f"{name.lower()}.jsonl"
            entry = {"step": step, "value": value}
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class MemAbsorptionHook(BaseMetricHook):
    """Tracks GPU memory absorption during training."""
    
    def on_step(self, step: int, model: torch.nn.Module, **kwargs) -> Dict[str, float]:
        # Cache memory stats on device to avoid frequent host sync
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            mem_absorption = mem_allocated / max(mem_reserved, 1e-6)   # ratio
        else:
            mem_absorption = 0.0
            
        self._log_scalar('mem_absorption', mem_absorption, step)
        return {'mem_absorption': mem_absorption}


class TPGHook(BaseMetricHook):
    """Tracks Temporal Policy Gradient dial."""
    
    def on_step(self, step: int, model: torch.nn.Module, loss_dict: Dict[str, torch.Tensor] = None, **kwargs) -> Dict[str, float]:
        # Extract TPG loss component if available
        if loss_dict and 'temporal_loss' in loss_dict:
            tpg_value = loss_dict['temporal_loss'].detach().item()
        else:
            # Fallback: compute temporal gating penalty from model state
            tpg_value = 0.0
            if hasattr(model, 'temporal_dropout'):
                # Use gate values as proxy for temporal policy gradient
                gates = getattr(model.temporal_dropout, 'gates', None)
                if gates is not None:
                    tpg_value = (1.0 - gates.mean()).detach().item()
                    
        self._log_scalar('tpg', tpg_value, step)
        return {'tpg': tpg_value}


class CapGaugeHook(BaseMetricHook):
    """Tracks capacity gauge (model utilization)."""
    
    def on_step(self, step: int, model: torch.nn.Module, activations: torch.Tensor = None, **kwargs) -> Dict[str, float]:
        # Measure sparsity as proxy for capacity utilization
        if activations is not None:
            # Cache computation on device
            active_fraction = (activations.abs() > 1e-6).float().mean().detach().item()
            cap_gauge = active_fraction
        else:
            # Fallback: use model parameter utilization
            total_params = 0
            active_params = 0
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        total_params += param.numel()
                        active_params += (param.abs() > 1e-6).sum().item()
            cap_gauge = active_params / max(total_params, 1)
            
        self._log_scalar('cap_gauge', cap_gauge, step)
        return {'cap_gauge': cap_gauge}


class ICLPersistenceHook(BaseMetricHook):
    """Tracks In-Context Learning persistence across sequences."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_activations = None
    
    def on_step(self, step: int, model: torch.nn.Module, activations: torch.Tensor = None, **kwargs) -> Dict[str, float]:
        icl_persistence = 0.0
        
        if activations is not None and self._prev_activations is not None:
            # Compute cosine similarity between consecutive activation patterns
            curr_flat = activations.flatten()
            prev_flat = self._prev_activations.flatten()
            
            # Ensure same size
            min_size = min(curr_flat.size(0), prev_flat.size(0))
            curr_flat = curr_flat[:min_size]
            prev_flat = prev_flat[:min_size]
            
            if min_size > 0:
                cosine_sim = torch.nn.functional.cosine_similarity(
                    curr_flat.unsqueeze(0), prev_flat.unsqueeze(0), dim=1
                ).item()
                icl_persistence = max(0.0, min(1.0, cosine_sim))  # clamp to [0, 1]
        
        # Cache current activations for next step
        if activations is not None:
            self._prev_activations = activations.detach().clone()
            
        self._log_scalar('icl_persistence', icl_persistence, step)
        return {'icl_persistence': icl_persistence}


class WeightDeltaHook(BaseMetricHook):
    """Tracks weight change magnitude over time."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_weights = {}
    
    def on_step(self, step: int, model: torch.nn.Module, **kwargs) -> Dict[str, float]:
        weight_delta = 0.0
        total_params = 0
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in self._prev_weights:
                        delta = (param - self._prev_weights[name]).norm().item()
                        weight_delta += delta
                    total_params += 1
                    self._prev_weights[name] = param.detach().clone()
        
        # Normalize by number of parameters
        if total_params > 0:
            weight_delta /= total_params
            
        self._log_scalar('weight_delta', weight_delta, step)
        return {'weight_delta': weight_delta}


class RAGTraceHook(BaseMetricHook):
    """Tracks Retrieval-Augmented Generation trace patterns."""
    
    def on_step(self, step: int, model: torch.nn.Module, attention_weights: torch.Tensor = None, **kwargs) -> Dict[str, float]:
        rag_trace = 0.0
        
        if attention_weights is not None:
            # Measure attention entropy as proxy for retrieval diversity
            # Higher entropy = more diverse attention = better RAG trace
            with torch.no_grad():
                # Flatten attention weights and compute entropy
                attn_flat = attention_weights.flatten()
                attn_probs = torch.softmax(attn_flat, dim=0)
                entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum().item()
                
                # Normalize entropy to [0, 1] range
                max_entropy = torch.log(torch.tensor(float(attn_flat.size(0))))
                rag_trace = entropy / max_entropy.item() if max_entropy > 0 else 0.0
        else:
            # Fallback: use decoder weight patterns
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'weight'):
                with torch.no_grad():
                    decoder_weights = model.decoder.weight.flatten()
                    weight_std = decoder_weights.std().item()
                    weight_mean = decoder_weights.mean().abs().item()
                    rag_trace = weight_std / max(weight_mean, 1e-6)
                    rag_trace = min(1.0, rag_trace)  # clamp to [0, 1]
                    
        self._log_scalar('rag_trace', rag_trace, step)
        return {'rag_trace': rag_trace}

