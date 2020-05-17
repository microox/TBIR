import torch
import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim, c):
        super().__init__()
        # instantiate basic image projection to shared feature space
        self.img_proj = torch.nn.Sequential(
            torch.nn.Linear(img_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, c),
        )

        # instantiate basic text projection to shared feature space
        self.txt_proj = torch.nn.Sequential(
            torch.nn.Linear(txt_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, c),
        )

    def forward(self, x, y):
        # project image to shared feature space
        F = self.img_proj(x)

        # project description to shared feature space
        G = self.txt_proj(y)

        B = (F + G).sign()
        return F, G, B