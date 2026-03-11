import torch
from PIL import Image
from torchvision import transforms

from models.baseline_model import EfficientNetBinary

device = torch.device("cpu")

model = EfficientNetBinary()
model.load_state_dict(torch.load("models/efficientnet_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(img)
    prob = torch.softmax(out, dim=1)

print("Fake probability:", prob[0][0].item())
print("Real probability:", prob[0][1].item())
