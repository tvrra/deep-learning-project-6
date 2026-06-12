import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

from torchvision.datasets import CIFAR10
from torchvision import transforms

transform = transforms.ToTensor()

train_dataset = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Classes:", train_dataset.classes)