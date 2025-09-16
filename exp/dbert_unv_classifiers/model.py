import torch
import torch.nn as nn

class DebateUniversalClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        d: int = 192,
        h: int = 4,
        doc_layers: int = 1,
        lab_layers: int = 1,
        debate_layers: int = 1,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        ff_dim = ff_mult * d
        enc_block = lambda: nn.TransformerEncoderLayer(d, h, ff_dim, batch_first=True)
        self.emb = nn.Embedding(vocab_size, d)
        self.doc_enc = nn.TransformerEncoder(enc_block(), doc_layers)
        self.lab_enc = nn.TransformerEncoder(enc_block(), lab_layers)
        self.cross = nn.MultiheadAttention(d, h, batch_first=True)
        self.debate = nn.TransformerEncoder(enc_block(), debate_layers)
        self.cls = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, doc_ids: torch.Tensor, lab_ids: torch.Tensor) -> torch.Tensor:
        # doc_ids: [B, T], lab_ids: [B, M, t]
        B, M, t = lab_ids.shape
        doc_tokens = self.emb(doc_ids)
        label_tokens = self.emb(lab_ids.view(B * M, t))
        doc_ctx = self.doc_enc(doc_tokens)
        label_ctx = self.lab_enc(label_tokens).mean(1).view(B, M, -1)
        attn_out, _ = self.cross(label_ctx, doc_ctx, doc_ctx)
        debated = self.debate(label_ctx + attn_out)
        return self.cls(debated).squeeze(-1)

if __name__ == "__main__":
    model = DebateUniversalClassifier()
    doc_ids = torch.randint(0, 10, (1, 2048))
    lab_ids = torch.randint(0, 10, (1, 6, 8))
    logits = model(doc_ids, lab_ids)
    print(logits.shape, logits)
