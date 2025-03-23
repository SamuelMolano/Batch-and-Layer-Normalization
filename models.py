import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    """
    Custom implementation of Batch Normalization.
    We used EMA (Exponential Moving Average) for the inference statistics,
    as the PyTorch implementation does.

    Parameters
    ----------
    num_features : int
        Number of features in the input.
    eps : float, optional
        Small value to avoid division by zero, default is 1e-5.
    momentum : float, optional
        Momentum for the moving average, default is 0.1.

    Attributes
    ----------
    gamma : nn.Parameter
        Learnable scale parameter.
    beta : nn.Parameter
        Learnable shift parameter.
    running_mean : torch.Tensor
        Running mean of the input, used during inference.
    running_var : torch.Tensor
        Running variance of the input, used during inference.

    Methods
    -------
    forward(x)
        Applies batch normalization to the input.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        """
        Applies batch normalization to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        if self.training:
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze(0)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze(0)

            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            return self.gamma * x_hat + self.beta
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            return self.gamma * x_hat + self.beta

class LayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization.

    Parameters
    ----------
    num_features : int
        Number of features in the input.
    eps : float, optional
        Small value to avoid division by zero, default is 1e-5.

    Attributes
    ----------
    beta : nn.Parameter
        Learnable shift parameter.
    gamma : nn.Parameter
        Learnable scale parameter.

    Methods
    -------
    forward(x)
        Applies layer normalization to the input.
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.eps = eps

        self.beta = torch.nn.Parameter(torch.zeros(num_features))
        self.gamma = torch.nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        """
        Applies layer normalization to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        layer_mean = torch.mean(x, dim=-1, keepdim=True)
        layer_var = torch.var(x, dim=-1, keepdim=True, unbiased=False)

        x_hat = (x - layer_mean) / torch.sqrt(layer_var + self.eps)
        return self.gamma * x_hat + self.beta

class Check(nn.Module):
    """
    A simple module to store the input tensor for inspection.

    Attributes
    ----------
    value : torch.Tensor
        The stored input tensor.

    Methods
    -------
    forward(x)
        Stores the input tensor and returns it.
    """
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, x):
        """
        Stores the input tensor and returns it.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The same input tensor.
        """
        self.value = x
        return x

class Model_MNIST(nn.Module):
    """
    A neural network model for MNIST digit classification with optional normalization.

    Parameters
    ----------
    batch_norm_q : bool, optional
        Whether to use batch normalization, default is False.
    layer_norm_q : bool, optional
        Whether to use layer normalization, default is False.

    Attributes
    ----------
    model : nn.Sequential
        The sequential model containing all layers.

    Methods
    -------
    forward(x)
        Performs a forward pass through the network.
    """
    def __init__(self, batch_norm_q=False, layer_norm_q=False):
        super().__init__()

        self.batch_norm_q = batch_norm_q
        self.layer_norm_q = layer_norm_q
        self.check = Check()

        layers = [
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
        ]
        for _ in range(2):
            if batch_norm_q:
                layers.append(BatchNorm(num_features=100))
            elif layer_norm_q:
                layers.append(LayerNorm(num_features=100))
            layers += [
                nn.Sigmoid(),
                nn.Linear(100, 100)
            ]
        if batch_norm_q:
            layers.append(BatchNorm(num_features=100))
        elif layer_norm_q:
            layers.append(LayerNorm(num_features=100))
        layers += [
            self.check,
            nn.Sigmoid(),
            nn.Linear(100, 10),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after the forward pass.
        """
        return self.model(x)
