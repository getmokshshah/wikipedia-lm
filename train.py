import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import yaml
import os

from src.model import WikipediaTransformer
from src.data import create_dataloader

def train():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json",
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        mask_token="<mask>"
    )
    
    # Create model
    model = WikipediaTransformer(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Data loaders
    train_loader = create_dataloader('train', tokenizer, config)
    val_loader = create_dataloader('validation', tokenizer, config)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training
    model.train()
    step = 0
    best_loss = float('inf')
    os.makedirs('models', exist_ok=True)
    
    # Log dataset info
    print(f"\nDataset sizes:")
    print(f"  - Training batches per epoch: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(100):  # Large number, will break on max_steps
        print(f"\n=== Starting Epoch {epoch + 1} ===")
        epoch_start_step = step

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            
            # Evaluation
            if step % config['training']['eval_steps'] == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"Validation loss: {val_loss:.4f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'step': step
                    }, 'models/best_model.pt')
                    print("Saved best model!")
                
                model.train()
            
            if step >= config['training']['max_steps']:
                print(f"\nReached max_steps ({config['training']['max_steps']})")
                return
        
        epoch_steps = step - epoch_start_step
        print(f"Completed Epoch {epoch + 1}: {epoch_steps} steps (total: {step})")

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            total_loss += outputs['loss'].item()
            count += 1
    
    return total_loss / count

if __name__ == "__main__":
    train()