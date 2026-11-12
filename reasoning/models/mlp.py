import torch.nn as nn
import torch

class Block(nn.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.activation = nn.ReLU()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.activation(self.norm(x))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x

class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, num_layers, mlp_ratio):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, mlp_ratio) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat((x, y), dim=-1)
        z = self.input_layer(x)

        for block in self.blocks:
            z = z + block(z)

        z = self.output_layer(z)

        energy = 0.5 * torch.sum(z**2, dim=-1)

        return energy