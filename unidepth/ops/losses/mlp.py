import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(
            self,
            layer_sizes: list,                   # e.g. [3, 128, 512, 130816]
            init_weights: float | list = 0.0,    # scalar or list (weights)
            init_std_devs: float | list = 1e-3,  # scalar or list (std devs)
            init_bias: float | list = 0.0        # scalar or list (biases)
    ):
        super().__init__()

        # Validate layer dimensions
        assert len(layer_sizes) >= 2, "Need at least two layer's size"

        # Create layers
        self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.num_layers = len(self.layers)

        # Expand initialization values
        init_weights = self._to_list(init_weights)
        init_std_devs = self._to_list(init_std_devs)
        init_bias = self._to_list(init_bias)

        # Initialize layers
        for i, layer in enumerate(self.layers):
            nn.init.normal_(layer.weight, mean=init_weights[i], std=init_std_devs[i])
            nn.init.constant_(layer.bias, init_bias[i])

    def _to_list(self, val):

        # Check dimensions and return, expand to a list if needed
        if isinstance(val, list):
            assert len(val) == self.num_layers, "Invalid initialization size"
            return val
        else:
            return [val] * self.num_layers
        
    def forward(self, x):
            
        # Process all layers except the final one
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Apply the last layer without ReLU
        x = self.layers[-1](x)
        
        return x
