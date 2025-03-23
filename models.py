import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum = 0.1):
        super().__init__()

        self.eps = eps #paramètre fixé pour éviter la division par 0
        self.momentum = momentum #paramètre de moyenne mobile

        # Création de paramètres qu'on optimisera par la suite
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Création des statistiques globales qu'on utilisera pendant l'inférence
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            #On calcule les statistiques du mini-batch (dim = 0)
            batch_mean = torch.mean(x, dim = 0, keepdim = True)
            batch_var = torch.var(x, dim = 0, keepdim = True, unbiased=False)

            #On actualise les statistiques globales. On utilise la moyenne mobile exponentielle car même si ce n'est
            #l'implémentation dans le papier, il s'agit de l'implémentation dans PyTorch.

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze(0)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze(0)

            #On re-scale nos données
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            return self.gamma * x_hat + self.beta
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            return self.gamma * x_hat + self.beta


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.eps = eps #paramètre fixé pour éviter la division par 0

        # Création de paramètres qu'on optimisera par la suite
        self.beta = torch.nn.Parameter(torch.zeros(num_features))
        self.gamma = torch.nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        #On calcule les statistiques de la couche (dim = -1, i.e. la dernière dimension)
        layer_mean = torch.mean(x, dim = -1, keepdim = True)
        layer_var = torch.var(x, dim = -1, keepdim = True, unbiased=False)
        
        #On re scale nos données
        x_hat = (x - layer_mean) / torch.sqrt(layer_var + self.eps)
        return self.gamma * x_hat + self.beta

class Check(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None
    def forward(self, x):
        self.value = x
        return x

class Model_MNIST(nn.Module):
    def __init__(self, batch_norm_q=False, layer_norm_q=False):
        super().__init__()
        

        self.barch_norm_q = batch_norm_q
        self.layer_norm_q = layer_norm_q
        self.check = Check()

        layers = [
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
        ]
        for _ in range(2):
            if batch_norm_q:
                layers.append(BatchNorm(num_features = 100))
            elif layer_norm_q:
                layers.append(LayerNorm(num_features = 100))
            layers += [
                nn.Sigmoid(),
                nn.Linear(100, 100)
            ]
        if batch_norm_q:
            layers.append(BatchNorm(num_features = 100))
        elif layer_norm_q:
            layers.append(LayerNorm(num_features = 100))
        layers += [
            self.check,
            nn.Sigmoid(),
            nn.Linear(100, 10),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
