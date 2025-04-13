import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from src.dataset import MultimodalDataset
from src.model import CaptioningModel
from src.tokenizer import tokenize

# === Config ===
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load dataset ===
dataset = MultimodalDataset("aligned_data.json")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model, loss, optimizer ===
model = CaptioningModel(vocab_size=30522).to(DEVICE)  # 30522 is for BERT
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# === Training loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in loader:
        clip_feats = batch["clip_features"].to(DEVICE)  # (B, 512)
        texts = batch["audio_text"]
        tokens = tokenize(texts)
        input_ids = tokens["input_ids"].to(DEVICE)  # (B, seq_len)

        optimizer.zero_grad()
        logits = model(clip_feats)  # (B, 1, vocab)
        logits = logits.squeeze(1)  # (B, vocab)
        targets = input_ids[:, 0]   # use first token as a quick test (can improve)

        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"🌀 Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# === Save trained model ===
torch.save(model.state_dict(), "captioning_model.pt")
print("✅ Model saved to captioning_model.pt")
