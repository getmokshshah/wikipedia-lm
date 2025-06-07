import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_tokenizer(config):
    print("Training tokenizer...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Setup trainer
    trainer = trainers.BpeTrainer(
        vocab_size=config['model']['vocab_size'],
        special_tokens=["<pad>", "<eos>", "<unk>", "<mask>"],
        min_frequency=2
    )
    
    # Train on data
    tokenizer.train(['data/train.txt'], trainer)
    
    # Save tokenizer
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved!")