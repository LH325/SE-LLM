import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim=256,
                 dropout=0.1,
                 activation='tanh'):
        super().__init__()

        # activation
        if activation == 'relu':
            act_layer = nn.ReLU()
        elif activation == 'tanh':
            act_layer = nn.Tanh()
        elif activation == 'gelu':
            act_layer = nn.GELU()
        else:
            raise NotImplementedError(f'Unsupported activation: {activation}')

        # fixed 2-layer MLP
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.network(x)