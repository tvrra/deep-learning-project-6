import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms, models

# preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# load CIFAR10
dataset = CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform
)

# load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# remove classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

# test on one image
image, label = dataset[0]

with torch.no_grad():
    feature = model(image.unsqueeze(0))

print("Feature shape:", feature.shape)
print("Label:", label)