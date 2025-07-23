import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.manifold import TSNE
import numpy as np
from model import FashionCNN

# Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# FashionMNIST test 데이터셋 로딩
transform = transforms.ToTensor()
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False)

# 클래스 라벨 정의
classes = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", 
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
colors = plt.cm.get_cmap("tab10", 10)

# 모델 정의 및 로딩
model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashion_cnn.pth"))
model.eval()

# 중간 feature 추출기 정의
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

# 특징 벡터와 GT 라벨 수집
features_list, gt_labels_list = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        features = extractor(images)
        features_list.append(features.cpu())
        gt_labels_list.append(labels)

features = torch.cat(features_list).numpy()
gt_labels = torch.cat(gt_labels_list).numpy()

# t-SNE로 3차원 축소
tsne = TSNE(n_components=3, random_state=42, perplexity=30)
embeddings = tsne.fit_transform(features)

# --- 시각화 준비 ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 클래스별 3D 산점도
for i in range(10):
    idx = gt_labels == i
    ax.scatter(embeddings[idx, 0], embeddings[idx, 1], embeddings[idx, 2],
               color=colors(i), label=classes[i], alpha=0.6, s=15)

ax.set_title("3D t-SNE of FashionMNIST (GT only)")
ax.legend()

# --- 회전 애니메이션 함수 ---
def rotate(angle):
    ax.view_init(elev=30, azim=angle)

# 애니메이션 객체 생성
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)

# GIF로 저장
rot_animation.save("fashion_tsne_gt_only.gif", dpi=100, writer='pillow')
