import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import FashionCNN

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Label names
classes = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", 
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Load test set
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.ToTensor())
image, label = test_set[4]

# Load model
model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashion_cnn.pth"))
model.eval()

# Predict
def predict(image):
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

predicted_label = predict(image)
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Predicted: {classes[predicted_label]}, Actual: {classes[label]}")
plt.axis('off')
plt.show()