import torch
import math
from transformers import PreTrainedTokenizerFast
import yaml
from src.model import WikipediaTransformer
from src.data import create_dataloader

def evaluate():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = WikipediaTransformer(config)
    checkpoint = torch.load('models/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer  
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
    
    # Test data
    test_loader = create_dataloader('test', tokenizer, config)
    
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            total_loss += outputs['loss'].item()
            count += 1
    
    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    evaluate()