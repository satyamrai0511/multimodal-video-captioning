import json
import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, json_path):
        """
        Args:
            json_path (str): Path to aligned_data.json
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert CLIP features to tensor
        clip_tensor = torch.tensor(item["clip_features"], dtype=torch.float32)

        # Return both tensor and text
        return {
            "clip_features": clip_tensor,
            "audio_text": item["audio_text"],
            "timestamp": item["timestamp"],  # optional for debugging
        }
