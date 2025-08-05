"""
Data loader for handling multiple JSONL datasets with configurable mixing ratios.
Supports both local and Google Colab environments.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    sources: List[str]
    mixing_ratios: Dict[str, float]
    max_length: int = 2048
    train_split: float = 0.9
    val_split: float = 0.1
    shuffle: bool = True


class MultiDatasetLoader:
    """Loads and mixes multiple JSONL datasets according to specified ratios."""
    
    def __init__(self, config: DatasetConfig, base_path: str = ""):
        self.config = config
        self.base_path = Path(base_path)
        self.datasets = {}
        self.dataset_sizes = {}
        
    def load_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all datasets from JSONL files."""
        logger.info(f"Loading {len(self.config.sources)} datasets...")
        
        for source_path in self.config.sources:
            full_path = self.base_path / source_path
            dataset_name = self._extract_dataset_name(source_path)
            
            logger.info(f"Loading dataset: {dataset_name} from {full_path}")
            
            try:
                data = self._load_jsonl(full_path)
                self.datasets[dataset_name] = data
                self.dataset_sizes[dataset_name] = len(data)
                logger.info(f"Loaded {len(data)} samples from {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                
        return self.datasets
    
    def _extract_dataset_name(self, source_path: str) -> str:
        """Extract dataset name from file path."""
        filename = Path(source_path).stem
        
        # Map filenames to dataset names
        name_mapping = {
            "HuggingFaceFW_fineweb_edu_edu_web_train": "fineweb_edu",
            "wikitext_wiki103_train": "wikitext",
            "microsoft_orca_math_word_problems_200k_math_reasoning_train": "orca_math",
            "bookcorpus_books_train": "bookcorpus"
        }
        
        return name_mapping.get(filename, filename)
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from a JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
        return data
    
    def create_mixed_dataset(self, total_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Create a mixed dataset according to the specified ratios."""
        if not self.datasets:
            self.load_datasets()
        
        # Normalize mixing ratios
        total_ratio = sum(self.config.mixing_ratios.values())
        normalized_ratios = {
            name: ratio / total_ratio 
            for name, ratio in self.config.mixing_ratios.items()
        }
        
        # Determine total samples
        if total_samples is None:
            total_samples = min(self.dataset_sizes.values())
        
        # Calculate samples per dataset
        samples_per_dataset = {}
        for name, ratio in normalized_ratios.items():
            if name in self.datasets:
                samples_per_dataset[name] = int(total_samples * ratio)
        
        # Create mixed dataset
        mixed_data = []
        for dataset_name, num_samples in samples_per_dataset.items():
            dataset = self.datasets[dataset_name]
            
            # Sample with replacement if needed
            if num_samples > len(dataset):
                sampled_data = random.choices(dataset, k=num_samples)
            else:
                sampled_data = random.sample(dataset, num_samples)
            
            mixed_data.extend(sampled_data)
            logger.info(f"Added {num_samples} samples from {dataset_name}")
        
        # Shuffle the mixed dataset
        if self.config.shuffle:
            random.shuffle(mixed_data)
        
        logger.info(f"Created mixed dataset with {len(mixed_data)} total samples")
        return mixed_data
    
    def split_dataset(self, data: List[Dict[str, Any]]) -> tuple:
        """Split dataset into train and validation sets."""
        total_size = len(data)
        train_size = int(total_size * self.config.train_split)
        
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data


class TextDataset(Dataset):
    """PyTorch dataset for text data with tokenization."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        text_column: str = "text"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # Handle different text column names
        self.text_extractors = {
            "text": lambda x: x.get("text", ""),
            "question_answer": lambda x: f"Question: {x.get('question', '')}\nAnswer: {x.get('answer', '')}",
            "auto": self._auto_extract_text
        }
    
    def _auto_extract_text(self, item: Dict[str, Any]) -> str:
        """Automatically extract text from different data formats."""
        # Handle question-answer format (Orca Math)
        if "question" in item and "answer" in item:
            return f"Question: {item['question']}\nAnswer: {item['answer']}"
        
        # Handle text format (most datasets)
        if "text" in item:
            return item["text"]
        
        # Handle other formats
        if "content" in item:
            return item["content"]
        
        # Fallback: convert to string
        return str(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract text
        text = self.text_extractors["auto"](item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()  # For language modeling
        }


def create_data_loaders(
    config: DatasetConfig,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    num_workers: int = 2,
    base_path: str = "",
    total_samples: Optional[int] = None,
    pin_memory: bool = False
) -> tuple:
    """Create training and validation data loaders."""
    
    # Load and mix datasets
    loader = MultiDatasetLoader(config, base_path)
    mixed_data = loader.create_mixed_dataset(total_samples)
    train_data, val_data = loader.split_dataset(mixed_data)
    
    # Create PyTorch datasets
    train_dataset = TextDataset(
        train_data, 
        tokenizer, 
        max_length=config.max_length
    )
    val_dataset = TextDataset(
        val_data, 
        tokenizer, 
        max_length=config.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Test the data loader
    config = DatasetConfig(
        sources=[
            "data/raw/HuggingFaceFW_fineweb_edu_edu_web_train.jsonl",
            "data/raw/wikitext_wiki103_train.jsonl"
        ],
        mixing_ratios={"fineweb_edu": 0.7, "wikitext": 0.3},
        max_length=512
    )
    
    loader = MultiDatasetLoader(config)
    datasets = loader.load_datasets()
    mixed_data = loader.create_mixed_dataset(1000)
    
    print(f"Loaded {len(mixed_data)} mixed samples")
    print(f"Sample: {mixed_data[0]}")
