import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class WordEncoder(nn.Module):
    """Encodes character sequences within words to word embeddings"""
    
    def __init__(self, vocab_size: int = 256, char_dim: int = 512, 
                 num_layers: int = 3, num_heads: int = 6):
        super().__init__()
        self.char_dim = char_dim
        
        # Character embedding lookup
        self.char_embedding = nn.Embedding(vocab_size, char_dim)
        
        # Bidirectional transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=char_dim,
            nhead=num_heads,
            dim_feedforward=char_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, char_sequences: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            char_sequences: [batch_size, num_words, max_word_len] 
                          Each word starts with [W] token
            attention_mask: [batch_size, num_words, max_word_len]
        
        Returns:
            word_embeddings: [batch_size, num_words, char_dim]
        """
        batch_size, num_words, max_word_len = char_sequences.shape
        
        # Reshape to process all words together
        chars_flat = char_sequences.view(-1, max_word_len)  # [B*W, L]
        
        # Embed characters
        char_embeds = self.char_embedding(chars_flat)  # [B*W, L, char_dim]
        
        # Create attention mask if not provided
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1, max_word_len)
            # Convert to key padding mask (True = ignore)
            key_padding_mask = ~mask_flat.bool()
        else:
            key_padding_mask = None
        
        # Apply bidirectional transformer
        encoded = self.transformer(char_embeds, src_key_padding_mask=key_padding_mask)
        
        # Extract word embedding from [W] position (first token)
        word_embeddings = encoded[:, 0, :]  # [B*W, char_dim]
        
        # Reshape back to word sequence
        word_embeddings = word_embeddings.view(batch_size, num_words, self.char_dim)
        
        return word_embeddings


class WordBackbone(nn.Module):
    """Processes sequence of word embeddings autoregressively"""
    
    def __init__(self, word_dim: int = 2048, num_layers: int = 24, num_heads: int = 24):
        super().__init__()
        self.word_dim = word_dim
        
        # Causal transformer decoder (autoregressive)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=word_dim,
            nhead=num_heads,
            dim_feedforward=word_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
    def forward(self, word_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            word_embeddings: [batch_size, seq_len, word_dim]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            predictive_embeddings: [batch_size, seq_len, word_dim]
        """
        seq_len = word_embeddings.size(1)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(word_embeddings.device)
        
        # Create key padding mask if attention mask provided
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        # For decoder, we use the same sequence as both input and memory
        # This creates an autoregressive causal transformer
        output = self.transformer(
            tgt=word_embeddings,
            memory=word_embeddings,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask
        )
        
        return output


class WordDecoder(nn.Module):
    """Decodes predictive word embeddings to character sequences"""
    
    def __init__(self, word_dim: int = 2048, char_dim: int = 512, 
                 vocab_size: int = 256, num_layers: int = 3, num_heads: int = 6):
        super().__init__()
        self.char_dim = char_dim
        self.vocab_size = vocab_size
        
        # Character embedding (same as encoder)
        self.char_embedding = nn.Embedding(vocab_size, char_dim)
        
        # Project word dimension to character dimension
        self.word_to_char_proj = nn.Linear(word_dim, char_dim)
        
        # Causal transformer decoder for character generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=char_dim,
            nhead=num_heads,
            dim_feedforward=char_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(char_dim, vocab_size)
        
    def forward(self, predictive_word_embeddings: torch.Tensor,
                target_char_sequences: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictive_word_embeddings: [batch_size, num_words, word_dim]
            target_char_sequences: [batch_size, num_words, max_chars] 
                                 Target characters for next words
            attention_mask: [batch_size, num_words, max_chars]
        
        Returns:
            character_logits: [batch_size, num_words, max_chars, vocab_size]
        """
        batch_size, num_words, word_dim = predictive_word_embeddings.shape
        max_chars = target_char_sequences.size(-1)
        
        # Project word embeddings to character dimension
        word_context = self.word_to_char_proj(predictive_word_embeddings)  # [B, W, char_dim]
        
        # Flatten for processing
        word_context_flat = word_context.view(-1, self.char_dim)  # [B*W, char_dim]
        chars_flat = target_char_sequences.view(-1, max_chars)  # [B*W, max_chars]
        
        # Embed target characters
        char_embeds = self.char_embedding(chars_flat)  # [B*W, max_chars, char_dim]
        
        # Add word context as first token (like memory)
        word_context_expanded = word_context_flat.unsqueeze(1)  # [B*W, 1, char_dim]
        memory = word_context_expanded.repeat(1, max_chars, 1)  # [B*W, max_chars, char_dim]
        
        # Create causal mask for character sequence
        causal_mask = torch.triu(torch.ones(max_chars, max_chars), diagonal=1).bool()
        causal_mask = causal_mask.to(char_embeds.device)
        
        # Create attention masks if provided
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1, max_chars)
            key_padding_mask = ~mask_flat.bool()
        else:
            key_padding_mask = None
        
        # Apply transformer decoder
        decoded = self.transformer(
            tgt=char_embeds,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask
        )
        
        # Project to vocabulary logits
        logits = self.output_proj(decoded)  # [B*W, max_chars, vocab_size]
        
        # Reshape back
        logits = logits.view(batch_size, num_words, max_chars, self.vocab_size)
        
        return logits


class HierarchicalTransformer(nn.Module):
    """Complete hierarchical transformer combining all three components"""
    
    def __init__(self, 
                 vocab_size: int = 256,
                 char_dim: int = 512,
                 word_dim: int = 2048,
                 encoder_layers: int = 3,
                 encoder_heads: int = 6,
                 backbone_layers: int = 24,
                 backbone_heads: int = 24,
                 decoder_layers: int = 3,
                 decoder_heads: int = 6):
        super().__init__()
        
        self.encoder = WordEncoder(vocab_size, char_dim, encoder_layers, encoder_heads)
        self.backbone = WordBackbone(word_dim, backbone_layers, backbone_heads)
        self.decoder = WordDecoder(word_dim, char_dim, vocab_size, decoder_layers, decoder_heads)
        
        # Projection layers between encoder/decoder and backbone
        self.encoder_to_backbone = nn.Linear(char_dim, word_dim)
        self.backbone_to_decoder = nn.Linear(word_dim, char_dim)
        
    def forward(self, 
                input_words: torch.Tensor,
                target_words: torch.Tensor,
                input_mask: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_words: [batch_size, seq_len, max_word_len] - input word character sequences
            target_words: [batch_size, seq_len, max_word_len] - target next word sequences
            input_mask: [batch_size, seq_len, max_word_len] - mask for input words
            target_mask: [batch_size, seq_len, max_word_len] - mask for target words
        
        Returns:
            character_logits: [batch_size, seq_len, max_word_len, vocab_size]
        """
        # Encode words to embeddings
        word_embeddings = self.encoder(input_words, input_mask)  # [B, seq_len, char_dim]
        
        # Project to backbone dimension
        backbone_input = self.encoder_to_backbone(word_embeddings)  # [B, seq_len, word_dim]
        
        # Process through backbone
        predictive_embeddings = self.backbone(backbone_input)  # [B, seq_len, word_dim]
        
        # Decode to character sequences
        character_logits = self.decoder(predictive_embeddings, target_words, target_mask)
        
        return character_logits

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = HierarchicalTransformer(
        vocab_size=256,
        char_dim=512,
        word_dim=2048,
        encoder_layers=3,
        encoder_heads=6,
        backbone_layers=24,
        backbone_heads=24,
        decoder_layers=3,
        decoder_heads=6
    )
    
    # Example input: "My name is"
    # Each word padded to max length, with [W] token (id=256) prepended
    W_TOKEN = 256  # Special [W] token
    
    batch_size, seq_len, max_word_len = 2, 3, 8
    
    # Mock input (normally you'd have proper tokenization)
    input_words = torch.randint(0, 256, (batch_size, seq_len, max_word_len))
    input_words[:, :, 0] = W_TOKEN  # First position is always [W]
    
    target_words = torch.randint(0, 256, (batch_size, seq_len, max_word_len))
    target_words[:, :, 0] = W_TOKEN  # First position is always [W]
    
    # Forward pass
    logits = model(input_words, target_words)
    print(f"Output logits shape: {logits.shape}")  # [2, 3, 8, 256]
    
    # Compute loss (example)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # ignore padding
    loss = loss_fn(logits.view(-1, 256), target_words.view(-1))
    print(f"Loss: {loss.item()}")
