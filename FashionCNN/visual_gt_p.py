import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.manifold import TSNE
import numpy as np
from model import FashionCNN

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.ToTensor()
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False)

# Labels
classes = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", 
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
colors = plt.cm.get_cmap("tab10", 10)

# Load trained model
model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashion_cnn.pth"))
model.eval()

# Intermediate feature extractor
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

extractor = FeatureExtractor(model).to(device)

# Extract features
features_list, gt_labels_list, pred_labels_list = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        features = extractor(images)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        features_list.append(features.cpu())
        gt_labels_list.append(labels)
        pred_labels_list.append(preds.cpu())

features = torch.cat(features_list).numpy()
gt_labels = torch.cat(gt_labels_list).numpy()
pred_labels = torch.cat(pred_labels_list).numpy()

# 3D t-SNE
tsne = TSNE(n_components=3, random_state=42, perplexity=30)
embeddings = tsne.fit_transform(features)

# --- Set up figure for animation ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Pre-plot scatter points
for i in range(10):
    idx = (gt_labels == i)
    matched = pred_labels[idx] == gt_labels[idx]
    mismatched = ~matched

    # Correct
    ax.scatter(embeddings[idx][matched, 0], embeddings[idx][matched, 1], embeddings[idx][matched, 2],
               color=colors(i), marker='o', alpha=0.6, label=classes[i] + " (✓)")

    # Incorrect
    ax.scatter(embeddings[idx][mismatched, 0], embeddings[idx][mismatched, 1], embeddings[idx][mismatched, 2],
               color=colors(i), marker='x', alpha=0.4)

ax.set_title("3D t-SNE: GT Color + ✓/✗ Marker")
ax.legend(loc="upper right")

# Animation function
def rotate(angle):
    ax.view_init(elev=30, azim=angle)

# Animate rotation
rot = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)

# Save as GIF
rot.save("fashion_tsne_gt_vs_pred.gif", dpi=100, writer='pillow')

# --- 클래스별 match / mismatch 비율 출력 ---
print("\n[📊 Class-wise Prediction Accuracy]")
for i in range(10):
    idx = (gt_labels == i)
    total = np.sum(idx)
    correct = np.sum(pred_labels[idx] == gt_labels[idx])
    incorrect = total - correct
    match_ratio = correct / total * 100
    mismatch_ratio = incorrect / total * 100
    print(f"{classes[i]:<12} ➤ Match: {match_ratio:5.2f}%, Mismatch: {mismatch_ratio:5.2f}%  (Total: {total})")


from collections import Counter

print("\n[❌ Mismatch Distribution per GT Class]")
for i in range(10):
    idx = (gt_labels == i)
    mismatches = pred_labels[idx] != gt_labels[idx]
    mismatch_preds = pred_labels[idx][mismatches]

    if len(mismatch_preds) == 0:
        print(f"{classes[i]:<12} ➤ Perfectly classified ✅")
        continue

    counter = Counter(mismatch_preds)
    total_mismatches = len(mismatch_preds)

    print(f"{classes[i]:<12} ➤ Total mismatches: {total_mismatches}")
    for wrong_label, count in counter.most_common():
        ratio = count / total_mismatches * 100
        print(f"    → Predicted as {classes[wrong_label]:<12} : {count:3d} times ({ratio:5.2f}%)")
