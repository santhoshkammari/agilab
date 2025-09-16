# train.py

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from model import DebateUniversalClassifier  # assumes model.py contains your class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# 1. Load datasets
# ------------------
ds_emotion = load_dataset("dair-ai/emotion", split="train")
ds_ag = load_dataset("ag_news", split="train")
ds_yelp = load_dataset("yelp_polarity", split="train")

emotion_labels = ds_emotion.features["label"].names
ag_labels = ds_ag.features["label"].names
yelp_labels = ["negative", "positive"]

def make_pairs(dataset, label_names, text_field="text", max_samples=None):
    pairs = []
    for i, ex in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        doc = ex[text_field]
        gold = label_names[ex["label"]]
        for lbl in label_names:
            sent = f"This text is about {lbl}."
            y = 1 if lbl == gold else 0
            pairs.append((doc, sent, y))
    return pairs

pairs_emotion = make_pairs(ds_emotion, emotion_labels, max_samples=3000)
pairs_ag = make_pairs(ds_ag, ag_labels, max_samples=3000)
pairs_yelp = make_pairs(ds_yelp, yelp_labels, max_samples=3000)

all_pairs = pairs_emotion + pairs_ag + pairs_yelp

# ------------------
# 2. Split train/val
# ------------------
val_ratio = 0.1
val_size = int(len(all_pairs) * val_ratio)
train_size = len(all_pairs) - val_size
train_pairs, val_pairs = random_split(all_pairs, [train_size, val_size])

# ------------------
# 3. Tokenizer + collator
# ------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def collate(batch):
    docs, labs, ys = zip(*batch)
    doc_enc = tokenizer(list(docs), padding=True, truncation=True, max_length=256, return_tensors="pt")
    lab_enc = tokenizer(list(labs), padding=True, truncation=True, max_length=16, return_tensors="pt")
    B = len(batch)
    return (
        doc_enc["input_ids"].to(DEVICE),
        lab_enc["input_ids"].view(B, 1, -1).to(DEVICE),
        torch.tensor(ys, dtype=torch.float32, device=DEVICE),
    )

train_loader = DataLoader(train_pairs, batch_size=32, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_pairs, batch_size=32, shuffle=False, collate_fn=collate)

# ------------------
# 4. Model, loss, optim
# ------------------
model = DebateUniversalClassifier(vocab_size=tokenizer.vocab_size, d=192, h=4).to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

writer = SummaryWriter(log_dir="runs/universal_classifier")

# ------------------
# 5. Train + Eval Loop
# ------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    total_loss = 0.0
    for doc_ids, lab_ids, y in train_loader:
        opt.zero_grad()
        logits = model(doc_ids, lab_ids).squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_train = total_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for doc_ids, lab_ids, y in val_loader:
            logits = model(doc_ids, lab_ids).squeeze(-1)
            loss = loss_fn(logits, y)
            val_loss += loss.item()
    avg_val = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
    writer.add_scalars("Loss", {"train": avg_train, "val": avg_val}, epoch+1)

# ------------------
# 6. Save model
# ------------------
torch.save(model.state_dict(), "debate_universal_classifier.pt")
writer.close()
print("Training complete, model + logs saved.")

