# Wikipedia Language Model

A simple transformer language model trained on Wikipedia text.

## Quick Start

```bash
# 1. Setup environment
git clone https://github.com/getmokshshah/wikipedia-lm.git
cd wikipedia-lm
conda env create -f environment.yml
conda activate wikipedia-lm
pip install -e .

# 2. Download and process data
python -c "from src.data import download_and_process_data; import yaml; download_and_process_data(yaml.safe_load(open('config.yaml')))"

# 3. Train tokenizer
python -c "from src.tokenizer import train_tokenizer; import yaml; train_tokenizer(yaml.safe_load(open('config.yaml')))"

# 4. Train model
python train.py

# 5. Generate text
python generate.py
```

## Project Structure

```
wikipedia-lm/
├── environment.yml          # Conda environment
├── setup.py                 # Package setup  
├── config.yaml              # Model configuration
├── train.py                 # Training script
├── generate.py              # Text generation
├── evaluate.py              # Model evaluation
├── src/
│   ├── __init__.py          # Package marker
│   ├── model.py             # Transformer implementation
│   ├── data.py              # Data processing
│   └── tokenizer.py         # Tokenizer training
├── README.md                # This file
└── .gitignore               # Git ignore rules
```

## Usage

### Training
```bash
python train.py
```
Trains the model on processed Wikipedia data. Saves checkpoints in `models/` directory.

### Text Generation
```bash
python generate.py
```
Interactive text generation. Enter prompts and get completions.

### Evaluation
```bash
python evaluate.py
```
Evaluates the trained model and reports perplexity.

## Configuration

Edit `config.yaml` to modify:

- **Model size**: `d_model`, `n_layers`, `n_heads` (default: 110M parameters)
- **Training**: `batch_size`, `learning_rate`, `max_steps`
- **Data**: `max_articles` (default: 10,000 for quick start)

## Model Architecture

- **Type**: Transformer decoder (GPT-style)
- **Default size**: 6 layers, 512 dimensions, 8 attention heads
- **Tokenizer**: BPE with 32k vocabulary
- **Context length**: 512 tokens
- **Parameters**: ~110M (configurable)

## Requirements

- Python 3.9+
- PyTorch 2.0+
- 8GB+ GPU memory (or CPU)
- ~5GB disk space for data

## Examples

```python
# Training
python train.py  # Trains for 50k steps by default

# Generation
python generate.py
> Prompt: The future of artificial intelligence
> Generated: The future of artificial intelligence will transform...

# Evaluation  
python evaluate.py
> Test Loss: 3.45, Perplexity: 31.5
```

## Customization

### Small Model (faster training)
```yaml
# config.yaml
model:
  d_model: 256
  n_layers: 4
  n_heads: 4
```

### Large Dataset
```yaml
# config.yaml  
data:
  max_articles: 100000  # Use more articles
```

### Longer Training
```yaml
# config.yaml
training:
  max_steps: 200000  # Train longer
```

## Files Generated

During usage, these files will be created:
- `data/` - Processed Wikipedia text
- `models/` - Trained model checkpoints  
- `tokenizer.json` - Trained BPE tokenizer

## License

MIT License