import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import re
from typing import List, Tuple, Optional

class TinyWordEncoder(nn.Module):
    def __init__(self, vocab_size=128, char_dim=64, num_layers=1, num_heads=2):
        super().__init__()
        self.char_dim = char_dim
        self.char_embedding = nn.Embedding(vocab_size, char_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=char_dim,
            nhead=num_heads,
            dim_feedforward=char_dim * 2,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, char_sequences):
        char_embeds = self.char_embedding(char_sequences)
        encoded = self.transformer(char_embeds)
        return encoded[:, 0, :]  # Return [W] position embedding

class TinyWordBackbone(nn.Module):
    def __init__(self, word_dim=128, num_layers=2, num_heads=2):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=word_dim,
            nhead=num_heads,
            dim_feedforward=word_dim * 2,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
    def forward(self, word_embeddings):
        seq_len = word_embeddings.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        output = self.transformer(
            tgt=word_embeddings,
            memory=word_embeddings,
            tgt_mask=causal_mask
        )
        return output

class TinyWordDecoder(nn.Module):
    def __init__(self, word_dim=128, char_dim=64, vocab_size=128, num_layers=1, num_heads=2):
        super().__init__()
        self.char_dim = char_dim
        self.vocab_size = vocab_size
        
        self.char_embedding = nn.Embedding(vocab_size, char_dim)
        self.word_to_char_proj = nn.Linear(word_dim, char_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=char_dim,
            nhead=num_heads,
            dim_feedforward=char_dim * 2,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(char_dim, vocab_size)
        
    def forward(self, predictive_word_embeddings, target_char_sequences):
        word_context = self.word_to_char_proj(predictive_word_embeddings)
        char_embeds = self.char_embedding(target_char_sequences)
        
        max_chars = char_embeds.size(1)
        memory = word_context.unsqueeze(1).repeat(1, max_chars, 1)
        
        causal_mask = torch.triu(torch.ones(max_chars, max_chars), diagonal=1).bool()
        
        decoded = self.transformer(
            tgt=char_embeds,
            memory=memory,
            tgt_mask=causal_mask
        )
        
        logits = self.output_proj(decoded)
        return logits

class TinyHierarchicalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 128  # ASCII + special tokens
        self.char_dim = 64
        self.word_dim = 128
        
        self.encoder = TinyWordEncoder(self.vocab_size, self.char_dim, 1, 2)
        self.backbone = TinyWordBackbone(self.word_dim, 2, 2)
        self.decoder = TinyWordDecoder(self.word_dim, self.char_dim, self.vocab_size, 1, 2)
        
        self.encoder_to_backbone = nn.Linear(self.char_dim, self.word_dim)
        
        # Special tokens
        self.W_TOKEN = 126  # [W] token
        self.PAD_TOKEN = 127  # Padding
        
    def forward(self, input_words, target_words):
        # Encode words
        batch_size, seq_len, max_word_len = input_words.shape
        input_flat = input_words.view(-1, max_word_len)
        word_embeddings = self.encoder(input_flat)
        word_embeddings = word_embeddings.view(batch_size, seq_len, self.char_dim)
        
        # Project and process through backbone
        backbone_input = self.encoder_to_backbone(word_embeddings)
        predictive_embeddings = self.backbone(backbone_input)
        
        # Decode characters
        target_flat = target_words.view(-1, target_words.size(-1))
        pred_flat = predictive_embeddings.view(-1, self.word_dim)
        logits = self.decoder(pred_flat, target_flat)
        
        return logits.view(batch_size, seq_len, target_words.size(-1), self.vocab_size)

class SimpleTokenizer:
    def __init__(self):
        self.W_TOKEN = 126
        self.PAD_TOKEN = 127
        
    def encode_text(self, text: str, max_words: int = 10, max_word_len: int = 8) -> torch.Tensor:
        """Convert text to tensor format"""
        words = text.lower().split()[:max_words]
        
        encoded_words = []
        for word in words:
            # Add [W] token at start
            chars = [self.W_TOKEN]
            # Add characters (limit to ASCII printable)
            for char in word[:max_word_len-1]:
                char_id = min(ord(char), 125)  # Cap at 125 to leave room for special tokens
                chars.append(char_id)
            
            # Pad to max_word_len
            while len(chars) < max_word_len:
                chars.append(self.PAD_TOKEN)
            
            encoded_words.append(chars[:max_word_len])
        
        # Pad to max_words
        while len(encoded_words) < max_words:
            encoded_words.append([self.PAD_TOKEN] * max_word_len)
        
        return torch.tensor(encoded_words).unsqueeze(0)  # Add batch dimension
    
    def decode_chars(self, char_ids: List[int]) -> str:
        """Convert character IDs back to string"""
        chars = []
        for char_id in char_ids:
            if char_id == self.W_TOKEN:
                continue
            elif char_id == self.PAD_TOKEN:
                break
            elif char_id < 126:
                chars.append(chr(char_id))
        return ''.join(chars)

def generate_next_word(model: TinyHierarchicalTransformer, 
                      tokenizer: SimpleTokenizer,
                      input_text: str,
                      max_len: int = 8) -> str:
    """Generate next word character by character"""
    model.eval()
    
    with torch.no_grad():
        # Encode input
        input_words = tokenizer.encode_text(input_text)
        batch_size, seq_len, max_word_len = input_words.shape
        
        # Get word embeddings and backbone output
        input_flat = input_words.view(-1, max_word_len)
        word_embeddings = model.encoder(input_flat)
        word_embeddings = word_embeddings.view(batch_size, seq_len, model.char_dim)
        
        backbone_input = model.encoder_to_backbone(word_embeddings)
        predictive_embeddings = model.backbone(backbone_input)
        
        # Get last predictive embedding (for next word)
        last_pred = predictive_embeddings[:, -1:, :]  # [1, 1, word_dim]
        
        # Generate characters autoregressively
        generated_chars = [tokenizer.W_TOKEN]  # Start with [W]
        
        for i in range(max_len):
            # Prepare current sequence
            current_seq = generated_chars + [tokenizer.PAD_TOKEN] * (max_len - len(generated_chars))
            current_seq = torch.tensor(current_seq).unsqueeze(0)  # [1, max_len]
            
            # Get logits for next character
            logits = model.decoder(last_pred.squeeze(1), current_seq)  # [1, max_len, vocab_size]
            next_logits = logits[0, len(generated_chars)-1, :]  # Logits for next position
            
            # Sample next character
            probs = F.softmax(next_logits, dim=-1)
            next_char = torch.multinomial(probs, 1).item()
            
            if next_char == tokenizer.W_TOKEN:  # End of word
                break
            
            generated_chars.append(next_char)
        
        return tokenizer.decode_chars(generated_chars[1:])  # Skip initial [W]

def interactive_demo():
    """Interactive demo of the tiny model"""
    print("Initializing Tiny Hierarchical Transformer...")
    
    model = TinyHierarchicalTransformer()
    tokenizer = SimpleTokenizer()
    
    # Initialize with random weights (no training)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n=== Tiny Hierarchical Transformer Demo ===")
    print("This is an untrained model - outputs will be random!")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            input_text = input("Enter text: ").strip()
            
            if input_text.lower() == 'quit':
                break
            
            if not input_text:
                continue
            
            print("Generating next word...")
            next_word = generate_next_word(model, tokenizer, input_text)
            
            print(f"Input: '{input_text}'")
            print(f"Generated next word: '{next_word}'")
            print(f"Full: '{input_text} {next_word}'")
            print("-" * 40)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nDemo ended.")

def training_example():
    """Show how training would work"""
    print("\n=== Training Example ===")
    
    model = TinyHierarchicalTransformer()
    tokenizer = SimpleTokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Sample training data
    texts = [
        "the cat sat on the mat",
        "hello world how are you",
        "python is a programming language",
        "machine learning is fun"
    ]
    
    print("Sample training step...")
    
    for epoch in range(3):  # Just a few steps
        total_loss = 0
        
        for text in texts:
            words = text.split()
            if len(words) < 2:
                continue
                
            # Create input/target pairs
            for i in range(len(words) - 1):
                context = " ".join(words[:i+1])
                target_word = words[i+1]
                
                # Encode
                input_words = tokenizer.encode_text(context, max_words=5)
                target_chars = tokenizer.encode_text(target_word, max_words=1)
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(input_words, target_chars)
                
                # Compute loss
                target_flat = target_chars.view(-1)
                logits_flat = logits.view(-1, model.vocab_size)
                
                # Ignore padding tokens in loss
                mask = target_flat != tokenizer.PAD_TOKEN
                if mask.sum() > 0:
                    loss = F.cross_entropy(logits_flat[mask], target_flat[mask])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(texts):.4f}")
    
    print("Training example complete!")
    
    # Test generation after mini-training
    print("\nTesting generation after mini-training:")
    test_text = "the cat"
    next_word = generate_next_word(model, tokenizer, test_text)
    print(f"'{test_text}' -> '{next_word}'")

if __name__ == "__main__":
    print("Tiny Hierarchical Transformer Demo")
    print("==================================")
    
    # Show model info
    model = TinyHierarchicalTransformer()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params/1000:.1f}K)")
    print(f"Model size estimate: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    choice = input("\n1. Interactive demo\n2. Training example\n3. Both\nChoose (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        interactive_demo()
    
    if choice in ['2', '3']:
        training_example()
