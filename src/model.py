import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class WikipediaTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config['model']
        self.vocab_size = self.config['vocab_size']
        self.d_model = self.config['d_model']
        self.max_length = self.config['max_length']
        
        # Model components
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = self._create_positional_encoding()
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.d_model,
                self.config['n_heads'],
                self.config['d_ff'],
                self.config['dropout']
            ) for _ in range(self.config['n_layers'])
        ])
        
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)
        
        self._init_weights()
    
    def _create_positional_encoding(self):
        pe = torch.zeros(self.max_length, self.d_model)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids, labels=None):
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        # Embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Causal mask
        mask = self._create_causal_mask(seq_len, device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                 shift_labels.view(-1))
        
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0):
        self.eval()
        
        for _ in range(max_length):
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if input_ids.size(1) >= self.max_length:
                break
        
        return input_ids