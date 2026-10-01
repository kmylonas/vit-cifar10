from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision import datasets, transforms
from model import ViT


num_classes=10
patch_size=4
num_t_blocks=3
embed_dim=192
num_heads=3
num_epochs=1
batch_size=128

to_tensor = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=to_tensor,
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=to_tensor,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)



def train(model, train_loader):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for X,y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optim.zero_grad()
            logits = model(X)
            
            loss = loss_fn(logits, y)
            epoch_loss += loss.item()
            
            
            loss.backward()
            optim.step()

        print(f'epoch: {epoch}\t loss: {epoch_loss / len(train_loader)}')


def evaluate(model, test_loader):
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for X,y in test_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()

        print(f"test loss: {total_loss / len(test_loader)}")



def save_model(model, path="vit_cifar10.pth"):
    torch.save(model.state_dict(), path)


def load_model(model, device, path="vit_cifar10.pth"):
    state_dict = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model.to(device)
    return model



device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device {device}")

model = ViT(patch_size, num_t_blocks, embed_dim, num_heads, num_classes).to(device)
train(model, train_loader)
evaluate(model, test_loader)
save_model(model)




        
        
        
