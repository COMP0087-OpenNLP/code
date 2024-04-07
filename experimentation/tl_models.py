import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchmetrics

class LinearTransformationModel(torch.nn.Module):
    def __init__(self, embedding_dim: int, embedding_output_dim: int, dropout_rate: float = 0.1):
        super(LinearTransformationModel, self).__init__()
        self.linear = torch.nn.Linear(embedding_dim, embedding_output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.linear(x)
        return x

class ElementwiseProductModel(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(ElementwiseProductModel, self).__init__()
        self.params = nn.Parameter(torch.ones(embedding_dim))

    def weights(self):
        return F.softmax(self.params, dim=0)

    def forward(self, x):
        weights = self.weights()
        return x * weights
    
class StackWiseProductModel(torch.nn.Module):
    def __init__(self, stack_size):
        super(StackWiseProductModel, self).__init__()
        assert stack_size > 1, "Stack size must be greater than 1"
        self.stack_size = stack_size
        self.params = nn.Parameter(torch.ones(stack_size))

    def weights(self):
        return F.softmax(self.params, dim=0)

    def forward(self, x):
        # Make sure they some to one to prevent possible explosion
        weights = self.weights()
        for i in range(self.stack_size):
            j = i * 1024
            x[:, j:(j+1024)] *= weights[i]
        return x