import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MLP
import os

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor()),
    batch_size = 64,
    shuffle = True
)

model = MLP()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    total_correct = 0
    number = 0

    for batch_id, (x,y) in enumerate(train_loader):
        x = x.view(x.size(0),-1)
        logits = model(x)
        loss = loss_func(logits,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == y).sum().item()
        number += y.size(0)

        if batch_id % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch [{batch_id}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    accuracy = total_correct/ number
    print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

os.makedirs("./software/pytorch/saved_weights", exist_ok=True)
torch.save(model.state_dict(), "./software/pytorch/saved_weights/MLP_weights.pth")


