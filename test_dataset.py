from src.dataset import MultimodalDataset
from torch.utils.data import DataLoader

# Load the dataset
dataset = MultimodalDataset("aligned_data.json")

# Wrap in a DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"📦 Total samples: {len(dataset)}")

# Preview one batch
for batch in loader:
    print("🧠 Clip features shape:", batch["clip_features"].shape)
    print("💬 Audio text:", batch["audio_text"])
    print("⏱️ Timestamps:", batch["timestamp"])
    break
