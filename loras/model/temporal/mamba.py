import torch
import torch.nn as nn

from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        state_dim,
        layers_count,
        *,
        conv_size,
        expand
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.layers_count = layers_count

        self.input_proj = nn.Linear(self.input_dim, self.state_dim)
        self.output_proj = nn.Linear(self.state_dim, self.output_dim)

        self.layers = nn.ModuleList()
        for _ in range(self.layers_count):
            self.layers.append(Mamba(
                d_model=self.state_dim,
                d_state=self.state_dim,
                d_conv=conv_size,
                expand=expand
            ))

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)