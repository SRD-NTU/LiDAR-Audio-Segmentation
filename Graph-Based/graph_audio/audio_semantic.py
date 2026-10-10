"""Dataset-adapted audio semantics: annotations are training targets only."""
from torch import nn


class AudioCategoryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.GELU(),
                                 nn.Dropout(.3),nn.Linear(128,16))

    def forward(self, embedding):
        return self.net(embedding)
