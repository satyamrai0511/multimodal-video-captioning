import torch
import torch.nn as nn

class CaptioningModel(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=512, vocab_size=10000):
        super().__init__()
        self.embed = nn.Linear(embed_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, clip_feats, captions=None):
        """
        Args:
            clip_feats: Tensor of shape (batch, 512)
            captions: Tensor of shape (batch, seq_len) [Optional for future use]
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
        """
        x = self.embed(clip_feats).unsqueeze(1)  # shape: (batch, 1, hidden)
        output, _ = self.decoder(x)
        logits = self.output(output)
        return logits
