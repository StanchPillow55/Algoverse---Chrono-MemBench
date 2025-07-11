#!/usr/bin/env python3
"""
Download relevant dataset subsets for training Gemma-2B, Llama-3-8B, and LLaVA-1.6
"""

import os
from datasets import load_dataset
import json
from pathlib import Path

def download_dataset_subset(dataset_name, config_name, split, output_dir, max_samples=10000, subset_name=None):
    """Download a subset of a dataset and save as JSON lines"""
    
    print(f"\n📥 Downloading {dataset_name} ({config_name if config_name else 'default'}) - {split}")
    print(f"   Max samples: {max_samples}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        # Create filename
        safe_name = dataset_name.replace('/', '_').replace('-', '_')
        if subset_name:
            filename = f"{safe_name}_{subset_name}_{split}.jsonl"
        else:
            filename = f"{safe_name}_{split}.jsonl"
        
        output_file = output_path / filename
        
        # Download and save subset
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                if count >= max_samples:
                    break
                
                # Write as JSON lines
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                count += 1
                
                if count % 1000 == 0:
                    print(f"   Downloaded {count} samples...")
        
        print(f"✅ Saved {count} samples to {output_file}")
        
        # Create DVC file
        dvc_file = output_file.with_suffix('.jsonl.dvc')
        os.system(f"cd {output_dir} && dvc add {filename}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return None

def main():
    """Main download function"""
    
    # Configuration
    base_output_dir = "data/raw"
    
    # Datasets to download
    datasets_config = [
        # High-quality educational web text
        {
            'name': 'HuggingFaceFW/fineweb-edu',
            'config': None,
            'split': 'train',
            'max_samples': 50000,  # ~50K samples for educational content
            'subset_name': 'edu_web'
        },
        
        # Wikipedia articles - clean, structured text
        {
            'name': 'wikitext',
            'config': 'wikitext-103-v1',
            'split': 'train',
            'max_samples': 20000,  # Full WikiText-103 training set
            'subset_name': 'wiki103'
        },
        
        # Mathematical reasoning
        {
            'name': 'microsoft/orca-math-word-problems-200k',
            'config': None,
            'split': 'train',
            'max_samples': 25000,  # 25K math problems
            'subset_name': 'math_reasoning'
        },
        
        # Books corpus (if we can access it)
        {
            'name': 'bookcorpus',
            'config': None,
            'split': 'train',
            'max_samples': 15000,  # 15K book samples
            'subset_name': 'books'
        }
    ]
    
    print("🚀 Starting dataset downloads...")
    print(f"📁 Output directory: {base_output_dir}")
    
    downloaded_files = []
    
    for config in datasets_config:
        result = download_dataset_subset(
            dataset_name=config['name'],
            config_name=config['config'],
            split=config['split'],
            output_dir=base_output_dir,
            max_samples=config['max_samples'],
            subset_name=config['subset_name']
        )
        
        if result:
            downloaded_files.append(result)
    
    print(f"\n🎉 Download complete! Downloaded {len(downloaded_files)} datasets:")
    for file in downloaded_files:
        print(f"   • {file}")
    
    # Create a summary file
    summary_file = Path(base_output_dir) / "dataset_summary.json"
    summary = {
        'total_datasets': len(downloaded_files),
        'datasets': [str(f) for f in downloaded_files],
        'description': {
            'fineweb_edu': 'High-quality educational web content',
            'wikitext': 'Wikipedia articles for language modeling',
            'orca_math': 'Mathematical reasoning problems',
            'bookcorpus': 'Book text for language modeling'
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
