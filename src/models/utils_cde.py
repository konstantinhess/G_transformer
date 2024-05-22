import torch
from torch import nn
import torchcde


class CDEIntegrand(nn.Module):
    """
    Neural CDE integrand
    """
    def __init__(self, input_size, hidden_size, num_layer=1, dropout_rate=0.0):
        super().__init__()
        # Input layer
        self.cde_layers = [nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Dropout(dropout_rate)]
        # Hidden layers
        for _ in range(num_layer):
            self.cde_layers += [nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(), nn.Dropout(dropout_rate)
                                ]
        # Output layer
        self.cde_layers += [nn.Linear(hidden_size, hidden_size * input_size),
                            nn.Tanh(),
                            nn.Dropout(dropout_rate)]
        self.cde_layers = nn.Sequential(*self.cde_layers)

        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, t, x):
        return self.cde_layers(x).view(-1, self.hidden_size, self.input_size)


class NeuralCDE(nn.Module):
    """
    Neural CDE model
    """
    def __init__(self, input_size, hidden_size, num_layer=1, dropout_rate=0.0):
        super().__init__()

        # Neural CDE integrand
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.f = CDEIntegrand(input_size, hidden_size, num_layer, dropout_rate)

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def forward(self, x, init_states=None):

        if x.size(1) == 1:
            # Riemann-Stiltjes integral over singleton
            return torch.matmul(self.f(0, init_states), x.squeeze(1).unsqueeze(-1)).unsqueeze(1).squeeze(-1)
        else:
            coeffs = torchcde.linear_interpolation_coeffs(x)
            X = torchcde.LinearInterpolation(coeffs)

            if init_states is None:
                init_states = self.input_layer(x[:, 0, :])
            # Decoder receives init_states from encoder

            return torchcde.cdeint(X=X, func=self.f, z0=init_states, method='euler', t=X.grid_points)


