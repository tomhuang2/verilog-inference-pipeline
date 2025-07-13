import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,128)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        return x


    
