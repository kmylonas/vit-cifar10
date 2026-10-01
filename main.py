from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision import datasets, transforms
from model import ViT
from pathlib import Path
from dataset import ciphar_dataset
import argparse




# num_classes=10
# patch_size=4
# num_t_blocks=3
# embed_dim=192
# num_heads=3
# num_epochs=4
# batch_size=128
# resume = False



def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train a Vision Transformer on CIFAR-10"
    )

    # Model hyperparameters
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Paths and checkpointing
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/vit_cifar10_latest.pth",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint-path",
    )

    # Hardware
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="mps",
    )

    return parser



def train(model, train_loader, optim, start_epoch, num_epochs, device, checkpoint_path):
    
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(start_epoch, num_epochs):
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

        average_loss = epoch_loss / len(train_loader)
        print(f'epoch: {epoch}\t loss: {average_loss}')
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": average_loss,
            },
            checkpoint_path,
            )


def evaluate(model, test_loader, device):
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



def select_device(device):
    avail_device = "cpu"
    if device == "mps":
        if torch.backends.mps.is_available():
            avail_device = "mps"
    elif device == "cuda":
        if torch.cuda.is_available():
            avail_device = "cuda"
            
    
    return avail_device
        



def main():

    parser = create_argument_parser()
    args = parser.parse_args()

    device = select_device(args.device)
    torch.device(device)
    print(f"Using device {device}")

    train_loader, test_loader = ciphar_dataset(args.batch_size)

    model = ViT(args.patch_size, args.num_blocks, args.embed_dim, args.num_heads, args.num_classes).to(device)
    start_epoch = 0
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:

        checkpoint = torch.load(
            args.checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1


    train(model, train_loader, optim, start_epoch, args.epochs, device, args.checkpoint_path)
    evaluate(model, test_loader, device)



if __name__ == "__main__":
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    main()



        
        
        
