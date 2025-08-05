import torch
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

from membench_x.metrics import (
    BaseMetricHook, MemAbsorptionHook, TPGHook, CapGaugeHook,
    ICLPersistenceHook, WeightDeltaHook, RAGTraceHook
)


class TrainingLoop:
    """Training loop with integrated metric streaming for the six dials."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        metric_interval: int = 10,
        metrics_dir: Optional[Path] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.metric_interval = metric_interval
        
        # Initialize metric hooks
        self.metric_hooks: List[BaseMetricHook] = [
            MemAbsorptionHook(writer, metrics_dir),
            TPGHook(writer, metrics_dir),
            CapGaugeHook(writer, metrics_dir),
            ICLPersistenceHook(writer, metrics_dir),
            WeightDeltaHook(writer, metrics_dir),
            RAGTraceHook(writer, metrics_dir)
        ]
    
    def train_step(
        self,
        step: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a single training step with metric streaming."""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss (assuming model returns loss dict or single loss)
        if isinstance(outputs, dict):
            loss = outputs.get('loss', outputs.get('total_loss'))
            loss_dict = outputs
            activations = outputs.get('activations', outputs.get('hidden_states'))
        else:
            # Simple case: model returns raw outputs
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss_dict = {'loss': loss}
            activations = outputs
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Stream metrics every N steps
        metrics_data = {}
        if step % self.metric_interval == 0:
            metrics_data = self._stream_metrics(step, activations, loss_dict, **kwargs)
        
        return {
            'loss': loss.item(),
            'outputs': outputs,
            'metrics': metrics_data
        }
    
    def _stream_metrics(
        self,
        step: int,
        activations: Optional[torch.Tensor] = None,
        loss_dict: Optional[Dict[str, torch.Tensor]] = None,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Stream all six dial metrics."""
        
        all_metrics = {}
        
        for hook in self.metric_hooks:
            try:
                metrics = hook.on_step(
                    step=step,
                    model=self.model,
                    activations=activations,
                    loss_dict=loss_dict,
                    attention_weights=attention_weights,
                    **kwargs
                )
                all_metrics.update(metrics)
            except Exception as e:
                print(f"Warning: Failed to compute metrics for {hook.__class__.__name__}: {e}")
        
        return all_metrics
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
        start_step: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch with metric streaming."""
        
        epoch_metrics = {}
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            step = start_step + batch_idx
            
            # Extract inputs and targets from batch
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get('input_ids', batch.get('inputs'))
                targets = batch.get('labels', batch.get('targets', inputs))
            else:
                inputs = batch
                targets = batch
            
            # Move to device
            device = next(self.model.parameters()).device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            
            # Training step
            step_result = self.train_step(step, inputs, targets)
            
            total_loss += step_result['loss']
            num_batches += 1
            
            # Aggregate metrics
            if step_result['metrics']:
                for key, value in step_result['metrics'].items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
        
        # Average metrics over epoch
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics['avg_loss'] = avg_loss
        
        # Log epoch-level metrics
        self.writer.add_scalar('epoch_loss', avg_loss, epoch)
        for key, value in avg_metrics.items():
            if key.startswith('avg_') and key != 'avg_loss':
                self.writer.add_scalar(f"epoch_{key[4:]}", value, epoch)
        
        return avg_metrics


def create_training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    log_dir: str = "runs",
    metric_interval: int = 10,
    metrics_dir: Optional[str] = None
) -> TrainingLoop:
    """Factory function to create a training loop with tensorboard logging."""
    
    writer = SummaryWriter(log_dir)
    metrics_path = Path(metrics_dir) if metrics_dir else None
    
    return TrainingLoop(
        model=model,
        optimizer=optimizer,
        writer=writer,
        metric_interval=metric_interval,
        metrics_dir=metrics_path
    )
