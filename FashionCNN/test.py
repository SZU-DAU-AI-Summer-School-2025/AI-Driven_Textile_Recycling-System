import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import FashionCNN

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset and loader
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=100)

# Label output map
def output_label(label):
    mapping = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    return mapping[label]

# Load model
model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashion_cnn.pth"))
model.eval()

# Evaluate accuracy
class_correct = [0. for _ in range(10)]
total_correct = [0. for _ in range(10)]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs, 1)[1]
        c = (predicted == labels).squeeze()
        for i in range(100):
            label = labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1

for i in range(10):
    print(f"Accuracy of {output_label(i)}: {class_correct[i] * 100 / total_correct[i]:.2f}%")
