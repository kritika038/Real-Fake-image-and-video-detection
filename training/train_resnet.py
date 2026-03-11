import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.resnet_model import ResNetBinary


def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder("data/train", transform=transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = ResNetBinary().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 3
    best_val_acc = 0

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        model.train()

        correct = 0
        total = 0

        loop = tqdm(train_loader, desc="Training")

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        print("Train Accuracy:", train_acc)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            loop = tqdm(val_loader, desc="Validation")

            for images, labels in loop:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print("Validation Accuracy:", val_acc)

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(model.state_dict(), "models/resnet_model.pth")

            print("ResNet model saved.")

    print("\nBest Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    main()