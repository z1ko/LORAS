import torch.nn as nn

class GLU(nn.Module):
    """Gated Linear Units used in the S4 paper
    """
    def __init__(
        self,
        features,
        dropout=0.0
    ):
        super().__init__()

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Sequential(
            nn.Linear(features, 2 * features),
            nn.GLU(dim=-1)
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        x = self.output_linear(x)
        return x