import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.baseline_model import BaselineModel
from sklearn.metrics import classification_report, confusion_matrix

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Test dataset
test_data = datasets.ImageFolder("data/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)

# Load model
model = BaselineModel().to(device)
model.load_state_dict(torch.load("models/efficientnet_model.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["REAL","FAKE"]))
