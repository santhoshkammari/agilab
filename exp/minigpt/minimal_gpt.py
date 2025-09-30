import torch
import torch.nn.functional as F

class GPT(torch.nn.Module):
    def __init__(self, vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.token_embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos_embed = torch.nn.Embedding(max_len, d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_out = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.token_embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        return self.head(x)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp = MLP(d_model)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class MLP(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, 4 * d_model)
        self.fc2 = torch.nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class BPETokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {'<|endoftext|>': 0}

    def train(self, text):
        words = text.split()
        word_freqs = {}
        for word in words:
            word_freqs[word] = word_freqs.get(word, 0) + 1

        # Initialize vocab with characters
        vocab = list(set(''.join(words))) + list(self.special_tokens.keys())
        for i, token in enumerate(vocab):
            self.vocab[token] = i

        # Build initial word representations
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)

        # Learn merges
        while len(self.vocab) < self.vocab_size:
            pairs = {}
            for word, word_tokens in splits.items():
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + word_freqs[word]

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges[best_pair] = new_token

            # Update splits
            for word in splits:
                new_word = []
                i = 0
                while i < len(splits[word]):
                    if (i < len(splits[word]) - 1 and
                        (splits[word][i], splits[word][i + 1]) == best_pair):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(splits[word][i])
                        i += 1
                splits[word] = new_word

    def encode(self, text):
        if not hasattr(self, 'vocab') or not self.vocab:
            return [ord(c) % 256 for c in text]

        words = text.split()
        tokens = []
        for word in words:
            word_tokens = list(word)
            while len(word_tokens) > 1:
                pairs = [(word_tokens[i], word_tokens[i + 1]) for i in range(len(word_tokens) - 1)]
                bigram = None
                for pair in pairs:
                    if pair in self.merges:
                        bigram = pair
                        break
                if bigram is None:
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and
                        word_tokens[i] == first and word_tokens[i + 1] == second):
                        new_word.append(self.merges[bigram])
                        i += 2
                    else:
                        new_word.append(word_tokens[i])
                        i += 1
                word_tokens = new_word
            tokens.extend([self.vocab.get(token, 0) for token in word_tokens])
        return tokens

    def decode(self, tokens):
        if not hasattr(self, 'vocab'):
            return ''.join(chr(t) for t in tokens)
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join(inv_vocab.get(token, '') for token in tokens)

def create_dataloader(text, seq_len, batch_size, vocab_size=512):
    tokenizer = BPETokenizer(vocab_size)
    tokenizer.train(text)
    tokens = torch.tensor(tokenizer.encode(text))
    dataset = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        x = tokens[i:i+seq_len]
        y = tokens[i+1:i+seq_len+1]
        dataset.append((x, y))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), tokenizer

def train(model, dataloader, epochs=10, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    tokens = torch.tensor([tokenizer.encode(prompt)])
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) >= model.max_len:
                tokens = tokens[:, -model.max_len+1:]
            logits = model(tokens)
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
    return tokenizer.decode(tokens[0].tolist())

if __name__ == "__main__":
    vocab_size = 512
    model = GPT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=32)
    text = "hello world this is a simple gpt model for educational purposes. the quick brown fox jumps over the lazy dog. artificial intelligence and machine learning are fascinating fields. " * 50
    dataloader, tokenizer = create_dataloader(text, seq_len=16, batch_size=4, vocab_size=vocab_size)
    print("Training...")
    train(model, dataloader, epochs=20)
    print("\nGenerating:")
    print(generate(model, tokenizer, "hello", max_new_tokens=50))