import math
import random

class GPT:
    def __init__(self, vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        # Token embeddings: [vocab_size, d_model]
        self.token_embed = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(vocab_size)]

        # Position embeddings: [max_len, d_model]
        self.pos_embed = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(max_len)]

        # Transformer layers
        self.layers = []
        for _ in range(n_layers):
            layer = {
                'attn_w': [[[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)] for _ in range(3)],  # Q,K,V
                'attn_out': [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)],
                'mlp_w1': [[random.uniform(-0.1, 0.1) for _ in range(d_model*4)] for _ in range(d_model)],
                'mlp_w2': [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model*4)],
                'ln1_w': [1.0] * d_model,
                'ln1_b': [0.0] * d_model,
                'ln2_w': [1.0] * d_model,
                'ln2_b': [0.0] * d_model
            }
            self.layers.append(layer)

        # Output projection
        self.out_w = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(d_model)]

    def layer_norm(self, x, w, b):
        # x: [seq_len, d_model]
        result = []
        for i in range(len(x)):
            row = x[i]
            mean = sum(row) / len(row)
            var = sum((val - mean) ** 2 for val in row) / len(row)
            std = math.sqrt(var + 1e-6)
            norm_row = [w[j] * (row[j] - mean) / std + b[j] for j in range(len(row))]
            result.append(norm_row)
        return result

    def matmul(self, a, b):
        # a: [m, k], b: [k, n] -> [m, n]
        m, k, n = len(a), len(a[0]), len(b[0])
        result = [[0.0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                for p in range(k):
                    result[i][j] += a[i][p] * b[p][j]
        return result

    def softmax(self, x):
        # x: [seq_len, seq_len]
        result = []
        for row in x:
            max_val = max(row)
            exp_row = [math.exp(val - max_val) for val in row]
            sum_exp = sum(exp_row)
            result.append([val / sum_exp for val in exp_row])
        return result

    def attention(self, x, layer):
        seq_len, d_model = len(x), len(x[0])
        head_dim = d_model // self.n_heads

        # Compute Q, K, V
        q = self.matmul(x, layer['attn_w'][0])
        k = self.matmul(x, layer['attn_w'][1])
        v = self.matmul(x, layer['attn_w'][2])

        # Multi-head attention
        out = [[0.0] * d_model for _ in range(seq_len)]

        for h in range(self.n_heads):
            start, end = h * head_dim, (h + 1) * head_dim

            # Extract head
            q_h = [[q[i][j] for j in range(start, end)] for i in range(seq_len)]
            k_h = [[k[i][j] for j in range(start, end)] for i in range(seq_len)]
            v_h = [[v[i][j] for j in range(start, end)] for i in range(seq_len)]

            # Attention scores: [seq_len, seq_len]
            scores = [[0.0] * seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    if j <= i:  # Causal mask
                        scores[i][j] = sum(q_h[i][d] * k_h[j][d] for d in range(head_dim)) / math.sqrt(head_dim)
                    else:
                        scores[i][j] = -float('inf')

            # Apply softmax
            attn = self.softmax(scores)

            # Apply to values
            for i in range(seq_len):
                for d in range(head_dim):
                    val = sum(attn[i][j] * v_h[j][d] for j in range(seq_len))
                    out[i][start + d] += val

        return self.matmul(out, layer['attn_out'])

    def mlp(self, x, layer):
        # x -> Linear -> GELU -> Linear
        hidden = self.matmul(x, layer['mlp_w1'])

        # GELU activation
        for i in range(len(hidden)):
            for j in range(len(hidden[0])):
                val = hidden[i][j]
                hidden[i][j] = 0.5 * val * (1 + math.tanh(math.sqrt(2/math.pi) * (val + 0.044715 * val**3)))

        return self.matmul(hidden, layer['mlp_w2'])

    def forward(self, tokens):
        seq_len = len(tokens)

        # Embeddings: token + position
        x = []
        for i, token in enumerate(tokens):
            row = []
            for j in range(self.d_model):
                row.append(self.token_embed[token][j] + self.pos_embed[i][j])
            x.append(row)

        # Transformer layers
        for layer in self.layers:
            # Self-attention with residual
            attn_out = self.attention(self.layer_norm(x, layer['ln1_w'], layer['ln1_b']), layer)
            for i in range(seq_len):
                for j in range(self.d_model):
                    x[i][j] += attn_out[i][j]

            # MLP with residual
            mlp_out = self.mlp(self.layer_norm(x, layer['ln2_w'], layer['ln2_b']), layer)
            for i in range(seq_len):
                for j in range(self.d_model):
                    x[i][j] += mlp_out[i][j]

        # Output projection: [seq_len, vocab_size]
        return self.matmul(x, self.out_w)

class Tokenizer:
    def encode(self, text):
        return [ord(c) % 256 for c in text]

    def decode(self, tokens):
        return ''.join(chr(t) for t in tokens)

def create_batches(text, seq_len, batch_size):
    tokenizer = Tokenizer()
    tokens = tokenizer.encode(text)

    batches = []
    for i in range(0, len(tokens) - seq_len, batch_size * seq_len):
        batch = []
        for b in range(batch_size):
            start = i + b * seq_len
            if start + seq_len + 1 < len(tokens):
                x = tokens[start:start + seq_len]
                y = tokens[start + 1:start + seq_len + 1]
                batch.append((x, y))
        if batch:
            batches.append(batch)
    return batches

def cross_entropy_loss(logits, targets):
    # logits: [seq_len, vocab_size], targets: [seq_len]
    loss = 0.0
    for i, target in enumerate(targets):
        row = logits[i]
        max_val = max(row)
        exp_sum = sum(math.exp(val - max_val) for val in row)
        log_prob = row[target] - max_val - math.log(exp_sum)
        loss -= log_prob
    return loss / len(targets)

def train_step(model, batch, lr=0.001):
    total_loss = 0.0

    for x, y in batch:
        logits = model.forward(x)
        loss = cross_entropy_loss(logits, y)
        total_loss += loss

        # Simple gradient update (placeholder - real backprop would be complex)
        # This is a simplified version for educational purposes
        for layer in model.layers:
            for param in ['attn_w', 'mlp_w1', 'mlp_w2']:
                if param == 'attn_w':
                    for i in range(3):
                        for j in range(len(layer[param][i])):
                            for k in range(len(layer[param][i][j])):
                                layer[param][i][j][k] += random.uniform(-lr, lr)
                else:
                    for j in range(len(layer[param])):
                        for k in range(len(layer[param][j])):
                            layer[param][j][k] += random.uniform(-lr, lr)

    return total_loss / len(batch)

def generate(model, prompt, max_new_tokens=50):
    tokenizer = Tokenizer()
    tokens = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        if len(tokens) > model.max_len:
            tokens = tokens[-model.max_len:]

        logits = model.forward(tokens)
        last_logits = logits[-1]

        # Simple sampling
        max_val = max(last_logits)
        probs = [math.exp(val - max_val) for val in last_logits]
        prob_sum = sum(probs)
        probs = [p / prob_sum for p in probs]

        # Sample
        r = random.random()
        cumsum = 0.0
        next_token = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                next_token = i
                break

        tokens.append(next_token)

    return tokenizer.decode(tokens)

# Example usage
if __name__ == "__main__":
    # Initialize model
    gpt = GPT(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_len=32)

    # Sample text
    text = "hello world this is a simple gpt model for educational purposes"

    # Create training data
    batches = create_batches(text, seq_len=16, batch_size=2)

    # Train for a few steps
    print("Training...")
    for epoch in range(10):
        total_loss = 0.0
        for batch in batches:
            loss = train_step(gpt, batch, lr=0.001)
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(batches):.4f}")

    # Generate text
    print("\nGenerating text:")
    generated = generate(gpt, "hello", max_new_tokens=30)
    print(f"Generated: {generated}")