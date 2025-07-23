import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np
from model import FashionCNN
from matplotlib import animation

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load test set
transform = transforms.ToTensor()
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False)

# Class labels
classes = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", 
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Extract features
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = torch.nn.Sequential(
            model.layer1,
            model.layer2
        )
        self.fc1 = model.fc1

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashion_cnn.pth"))
model.eval()
extractor = FeatureExtractor(model).to(device)

# Feature extraction
all_features, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        features = extractor(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

features_tensor = torch.cat(all_features, dim=0)
labels_tensor = torch.cat(all_labels, dim=0)

# t-SNE
tsne = TSNE(n_components=3, random_state=42, perplexity=30)
embeddings_3d = tsne.fit_transform(features_tensor.numpy())

# 3D scatter setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.get_cmap("tab10", 10)

scatters = []
for i in range(10):
    idx = labels_tensor == i
    scat = ax.scatter(embeddings_3d[idx, 0],
                      embeddings_3d[idx, 1],
                      embeddings_3d[idx, 2],
                      label=classes[i],
                      alpha=0.6, s=15, color=colors(i))
    scatters.append(scat)

ax.set_title("3D t-SNE of FashionMNIST Features")
ax.legend()

# Animation function
def rotate(angle):
    ax.view_init(elev=30, azim=angle)

# Create animation
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)

# Save as GIF
rot_animation.save("fashion_3d.gif", dpi=100, writer='pillow')
