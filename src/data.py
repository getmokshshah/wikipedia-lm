import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

def download_and_process_data(config):
    print("Downloading Wikipedia...")
    # dataset = load_dataset("wikipedia", "20220301.en")
    dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)

    
    # Limit for quick start
    max_articles = config['data']['max_articles']
    if max_articles:
        dataset['train'] = dataset['train'].select(range(min(max_articles, len(dataset['train']))))
    
    print("Processing text...")
    processed = []
    
    for example in tqdm(dataset['train']):
        text = clean_text(example['text'])
        
        if (len(text) >= config['data']['min_length'] and 
            len(text) <= config['data']['max_length']):
            processed.append(text)
    
    # Split data
    val_size = int(len(processed) * config['data']['val_split'])
    test_size = int(len(processed) * config['data']['test_split'])
    
    splits = {
        'train': processed[:-val_size-test_size],
        'validation': processed[-val_size-test_size:-test_size] if test_size > 0 else processed[-val_size:],
        'test': processed[-test_size:] if test_size > 0 else []
    }
    
    # Save splits
    os.makedirs('data', exist_ok=True)
    for split_name, split_data in splits.items():
        with open(f'data/{split_name}.txt', 'w', encoding='utf-8') as f:
            for text in split_data:
                f.write(text + '\n')
        print(f"Saved {len(split_data)} {split_name} examples")

def clean_text(text):
    # Remove Wikipedia markup
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Clean whitespace
    text = ' '.join(text.split())
    return text

class WikipediaDataset(Dataset):
    def __init__(self, split, tokenizer, config):
        self.tokenizer = tokenizer
        self.max_length = config['model']['max_length']
        
        # Load data
        with open(f'data/{split}.txt', 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }

def create_dataloader(split, tokenizer, config):
    dataset = WikipediaDataset(split, tokenizer, config)
    
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=2
    )