import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import FashionCNN
import matplotlib.pyplot as plt

# 저장용 리스트
tshirt_misclassified_as_shirt = []
shirt_misclassified_as_tshirt = []

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashion_cnn.pth"))
model.eval()

# Dataset and loader
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=100)

# Label output map
def output_label(label):
    mapping = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    return mapping[label]

# 다시 모델 평가 (이번에는 오답 추출 중심)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs, 1)[1]

        for img, true_label, pred_label in zip(images, labels, predicted):
            if true_label == 0 and pred_label == 6:
                tshirt_misclassified_as_shirt.append((img.cpu(), true_label.item(), pred_label.item()))
            elif true_label == 6 and pred_label == 0:
                shirt_misclassified_as_tshirt.append((img.cpu(), true_label.item(), pred_label.item()))

# 시각화 함수
def show_misclassified(images_list, title, n=5):
    plt.figure(figsize=(10, 2))
    for i, (img, true_label, pred_label) in enumerate(images_list[:n]):
        plt.subplot(1, n, i+1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"True: {output_label(true_label)}\nPred: {output_label(pred_label)}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# 시각화 실행
show_misclassified(tshirt_misclassified_as_shirt, "T-shirt/Top → misclassified as Shirt")
show_misclassified(shirt_misclassified_as_tshirt, "Shirt → misclassified as T-shirt/Top")
