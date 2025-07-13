import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from model import MLP

def evaluate(model,test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = x.view(x.size(0),-1)
            logits = model(x)
            preds = torch.argmax(logits,dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Accuracy: {correct/total * 100:.2f}%")

model = MLP()
model.load_state_dict(torch.load("./software/pytorch/saved_weights/MLP_weights.pth"))
test_loader = DataLoader(
    datasets.MNIST(root="./data", train = False, download = True, transform = transforms.ToTensor()),
    batch_size = 64,
    shuffle = False
)

evaluate(model, test_loader)
