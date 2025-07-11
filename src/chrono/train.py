"""
Comprehensive training script for Gemma-2B, Llama-3-8B, and LLaVA-1.6.
Supports both local models and HuggingFace Hub streaming.
Optimized for Google Colab and local environments.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

from data_loader import DatasetConfig, create_data_loaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading for different architectures and sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.model_type = self.model_config['type']
        self.model_source = self.model_config['source']
        
    def get_model_path(self) -> str:
        """Get the model path based on configuration."""
        model_paths = self.model_config['paths'][self.model_type]
        return model_paths[self.model_source]
    
    def load_model_and_tokenizer(self) -> tuple:
        """Load model and tokenizer based on configuration."""
        model_path = self.get_model_path()
        
        logger.info(f"Loading {self.model_type} from {model_path} (source: {self.model_source})")
        
        # Load tokenizer
        tokenizer = self._load_tokenizer(model_path)
        
        # Load model
        model = self._load_model(model_path)
        
        return model, tokenizer
    
    def _load_tokenizer(self, model_path: str):
        """Load tokenizer with proper configuration."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self, model_path: str):
        """Load model with proper configuration."""
        # Common model loading arguments
        model_args = {
            'trust_remote_code': True,
            'torch_dtype': torch.float16,
            'device_map': 'auto' if torch.cuda.is_available() else None,
            'low_cpu_mem_usage': True
        }
        
        # Handle different model types
        if self.model_type == 'llava-1.6-7b':
            # Special handling for LLaVA models
            from transformers import LlavaNextForConditionalGeneration
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path, **model_args
            )
        else:
            # Standard causal LM models (Gemma, Llama)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, **model_args
            )
        
        return model
    
    def setup_lora(self, model, training_config: Dict[str, Any]):
        """Setup LoRA configuration for the model."""
        if training_config['training_type'] != 'lora':
            return model
        
        lora_config = training_config['lora']
        
        # Configure LoRA based on model type
        if self.model_type == 'gemma-2b':
            target_modules = lora_config['target_modules']
        elif self.model_type == 'llama-3-8b':
            target_modules = lora_config['target_modules']
        elif self.model_type == 'llava-1.6-7b':
            # LLaVA-specific target modules
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            target_modules=target_modules,
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model


class TrainingManager:
    """Manages the training process."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config['training']
        self.env_config = config['environment']
        self.eval_config = config['evaluation']
        
    def create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create training arguments from configuration."""
        
        # Determine mixed precision
        fp16 = self.env_config['mixed_precision'] == 'fp16'
        bf16 = self.env_config['mixed_precision'] == 'bf16'
        
        return TrainingArguments(
            output_dir=output_dir,
            
            # Training parameters
            num_train_epochs=1,  # We use max_steps instead
            max_steps=self.training_config['max_steps'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config['batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            
            # Optimization
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            max_grad_norm=self.training_config['max_grad_norm'],
            warmup_steps=self.training_config['warmup_steps'],
            lr_scheduler_type=self.training_config['scheduler'],
            
            # Evaluation and saving
            eval_steps=self.training_config['eval_steps'],
            save_steps=self.training_config['save_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=3,
            
            # Logging
            logging_steps=50,
            logging_dir=f"{output_dir}/logs",
            report_to="tensorboard" if self.env_config['logging']['tensorboard'] else None,
            
            # Hardware optimization
            fp16=fp16,
            bf16=bf16,
            dataloader_num_workers=self.env_config['dataloader_num_workers'],
            dataloader_pin_memory=self.env_config['pin_memory'],
            
            # Other settings
            remove_unused_columns=False,
            push_to_hub=False,
            hub_private_repo=True,
        )
    
    def train(
        self,
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        output_dir: str
    ):
        """Execute the training process."""
        
        # Create training arguments
        training_args = self.create_training_arguments(output_dir)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=val_dataloader.dataset,
            tokenizer=tokenizer,
            data_collator=None,  # Use default data collator
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        return trainer


def setup_colab_environment(config: Dict[str, Any]):
    """Setup Google Colab environment."""
    colab_config = config.get('colab', {})
    
    if colab_config.get('mount_drive', False):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully")
        except ImportError:
            logger.warning("Not running in Colab, skipping drive mount")
    
    if colab_config.get('install_requirements', False):
        # Install required packages
        packages = [
            'transformers',
            'peft',
            'datasets',
            'torch',
            'tensorboard',
            'accelerate'
        ]
        
        for package in packages:
            os.system(f'pip install -q {package}')
        
        logger.info("Required packages installed")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and merge base config if specified
    base_config_path = "configs/training_base.yaml"
    if Path(base_config_path).exists():
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configurations (specific config overrides base)
        merged_config = base_config.copy()
        merged_config.update(config)
        config = merged_config
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train language models")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--base_path", 
        type=str, 
        default="", 
        help="Base path for data files (useful for Colab)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    if config['environment']['platform'] == 'colab':
        setup_colab_environment(config)
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model manager
    model_manager = ModelManager(config)
    
    # Load model and tokenizer
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # Setup LoRA if specified
    model = model_manager.setup_lora(model, config['training'])
    
    # Create dataset configuration
    dataset_config = DatasetConfig(
        sources=config['dataset']['sources'],
        mixing_ratios=config['dataset']['mixing_ratios'],
        max_length=config['dataset']['max_length'],
        train_split=config['dataset']['train_split'],
        val_split=config['dataset']['val_split'],
        shuffle=config['dataset']['shuffle']
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_config,
        tokenizer,
        batch_size=config['training']['batch_size'],
        num_workers=config['environment']['dataloader_num_workers'],
        base_path=args.base_path
    )
    
    # Initialize training manager
    training_manager = TrainingManager(config)
    
    # Start training
    trainer = training_manager.train(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=str(output_dir)
    )
    
    logger.info("Training completed successfully!")
    
    # Generate some sample outputs
    logger.info("Generating sample outputs...")
    sample_prompts = [
        "The future of artificial intelligence is",
        "In mathematics, the concept of infinity",
        "Climate change is a global challenge that"
    ]
    
    for prompt in sample_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {generated_text}")
        logger.info("-" * 50)


if __name__ == "__main__":
    main()
