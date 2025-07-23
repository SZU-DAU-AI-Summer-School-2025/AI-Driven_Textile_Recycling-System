import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model import FashionCNN
from tqdm import trange

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset and Dataloader
transform = transforms.ToTensor()
full_train_set = datasets.FashionMNIST("./data", download=True, train=True, transform=transform)
train_size = int(0.8 * len(full_train_set))
val_size = len(full_train_set) - train_size
train_set, val_set = random_split(full_train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=100)
val_loader = DataLoader(val_set, batch_size=100)
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=100)

# Model setup
model = FashionCNN().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
count = 0
loss_list, iteration_list, accuracy_list = [], [], []
val_accuracy_list = []

for epoch in trange(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = Variable(images.view(100, 1, 28, 28))

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        count += 1
        if count % 50 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_images = val_images.view(val_images.size(0), 1, 28, 28)
                    val_outputs = model(val_images)
                    val_preds = torch.max(val_outputs, 1)[1]
                    correct += (val_preds == val_labels).sum().item()
                    total += val_labels.size(0)
            val_accuracy = correct * 100 / total
            val_accuracy_list.append(val_accuracy)
            loss_list.append(loss.item())
            iteration_list.append(count)
            print(f"Iteration: {count}, Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "fashion_cnn.pth")