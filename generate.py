import torch
from transformers import PreTrainedTokenizerFast
import yaml
from src.model import WikipediaTransformer

def generate():
    # Load config
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
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json",
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        mask_token="<mask>"
    )
    
    print("Model loaded! Enter prompts (type 'quit' to exit):")
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        if prompt:
            # Generate
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                generated = model.generate(input_ids, max_length=100, temperature=0.8)
            
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Generated: {text}")

if __name__ == "__main__":
    generate()