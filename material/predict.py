import os
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from model import build_model
from sklearn.preprocessing import LabelEncoder
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_mat_and_preprocess(mat_path, crop_size=50, rgb_bands=(29,19,9)):
    mat = scipy.io.loadmat(mat_path)
    if 'A' not in mat:
        raise ValueError(f"'A' 키가 {mat_path}에 없습니다.")
    data = mat['A']  # (H, W, Bands)
    H, W, B = data.shape
    
    # 중앙 크롭 좌표
    y_start = H//2 - crop_size//2
    y_end = y_start + crop_size
    x_start = W//2 - crop_size//2
    x_end = x_start + crop_size
    
    cropped = data[y_start:y_end, x_start:x_end, :]  # (crop_size, crop_size, B)
    mean_spectrum = np.mean(cropped, axis=(0,1))  # 1D 스펙트럼 벡터
    
    rgb = cropped[:, :, rgb_bands]
    rgb_norm = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[:, :, i]
        band_min = band.min()
        band_max = band.max()
        if band_max - band_min > 0:
            rgb_norm[:, :, i] = (band - band_min) / (band_max - band_min)
    rgb_img = (rgb_norm * 255).astype(np.uint8)
    
    return mean_spectrum, rgb_img, data  # data 전체도 같이 반환

model_path = './processed/mlp_model.pth'
label_encoder_path = './processed/label_encoder.pkl'
mat_file_path = '/Users/jeong-yeonghun/Desktop/2025_SZU_PROJECT/material/mat_data/99#.mat'

mean_spectrum, rgb_img, full_data = load_mat_and_preprocess(mat_file_path)

label_encoder = joblib.load(label_encoder_path)

input_dim = mean_spectrum.shape[0]
num_classes = len(label_encoder.classes_)

model = build_model(input_dim, num_classes, device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 모델 입력
X_tensor = torch.tensor(mean_spectrum, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(X_tensor)
    _, pred = torch.max(output, 1)

pred_label = label_encoder.inverse_transform(pred.cpu().numpy())[0]

# 전체 데이터에서 스펙트럼 평균 영상
mean_img = np.mean(full_data, axis=2)

# 크롭 위치 표시
crop_size = 50
H, W, _ = full_data.shape
y_start = H//2 - crop_size//2
x_start = W//2 - crop_size//2

plt.figure(figsize=(6,6))
plt.imshow(mean_img, cmap='gray')
plt.gca().add_patch(plt.Rectangle((x_start, y_start), crop_size, crop_size,
                                  linewidth=2, edgecolor='red', facecolor='none'))
plt.axis('off')
plt.title(f"Predicted class: {pred_label}", fontsize=16)
plt.show()
