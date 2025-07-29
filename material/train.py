# # first try

# # # import os
# # # import glob
# # # import scipy.io
# # # import numpy as np
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.neural_network import MLPClassifier
# # # from sklearn.metrics import classification_report, accuracy_score
# # # import matplotlib.pyplot as plt

# # # # ==== 1. 데이터 로드 및 전처리 ====

# # # # .mat 파일이 모여있는 폴더 경로
# # # mat_folder = '/Users/jeong-yeonghun/Desktop/2025_SZU_PROJECT/mat_data/'  
# # # mat_files = sorted(glob.glob(os.path.join(mat_folder, '*.mat')))

# # # print(f"Found {len(mat_files)} mat files.")

# # # # 사용자: 각 mat 파일별 레이블 (예: 파일명에 포함된 소재명 혹은 별도 리스트)
# # # # 여기선 임시 예시 (실제 데이터에 맞게 수정 필요)
# # # labels_dict = {
# # #     '5#.mat': 'polyester',
# # #     '8#.mat': 'polyester',
# # #     '11#.mat': 'polyester',
# # #     '12#.mat': 'polyester',
# # #     '14#.mat': 'polyester',
# # #     '18#.mat': 'polyester',
# # #     '19#.mat': 'polyester',
# # #     '35#.mat': 'wool',
# # #     '52#.mat': 'wool',
# # #     '54#.mat': 'wool',
# # #     '55#.mat': 'wool',
# # #     '58#.mat': 'wool',
# # #     '60#.mat': 'silk',
# # #     '62#.mat': 'silk',
# # #     '63#.mat': 'silk',
# # #     '64#.mat': 'silk',
# # #     '65#.mat': 'silk',
# # #     '66#.mat': 'silk',
# # #     '69#.mat': 'linen',
# # #     '71#.mat': 'linen',
# # #     '72#.mat': 'linen',
# # #     '73#.mat': 'linen',
# # #     '74#.mat': 'linen',
# # #     '75#.mat': 'cotton',
# # #     '76#.mat': 'cotton',
# # #     '77#.mat': 'cotton',
# # #     '79#.mat': 'cotton',
# # #     '80#.mat': 'cotton',
# # #     '81#.mat': 'cotton',
# # #     '83#.mat': 'cotton',
# # #     '99#.mat': 'recycled',
# # #     '100#.mat': 'recycled'
# # # }

# # # X = []
# # # y = []

# # # # 크롭 영역 설정 (예: 이미지 중앙 50x50 크롭) - 필요 시 조정 가능
# # # crop_size = 50

# # # for filepath in mat_files:
# # #     filename = os.path.basename(filepath)
    
# # #     # 라벨 매칭
# # #     if filename not in labels_dict:
# # #         print(f"Warning: No label for {filename}, skipping.")
# # #         continue
# # #     label = labels_dict[filename]
    
# # #     # mat 파일 로드
# # #     mat = scipy.io.loadmat(filepath)
# # #     if 'A' not in mat:
# # #         print(f"Warning: 'A' key not found in {filename}, skipping.")
# # #         continue
# # #     data = mat['A']  # (H, W, Bands)
    
# # #     H, W, B = data.shape
# # #     # 중앙 크롭 영역 좌표 계산
# # #     y_start = H//2 - crop_size//2
# # #     y_end = y_start + crop_size
# # #     x_start = W//2 - crop_size//2
# # #     x_end = x_start + crop_size
    
# # #     cropped = data[y_start:y_end, x_start:x_end, :]
    
# # #     # 평균 스펙트럼 계산
# # #     mean_spectrum = np.mean(cropped, axis=(0,1))  # shape (Bands,)
    
# # #     X.append(mean_spectrum)
# # #     y.append(label)
# # #     print(f"Processed {filename} label={label}")

# # # X = np.array(X)
# # # y = np.array(y)

# # # print(f"Dataset size: {X.shape[0]} samples, each with {X.shape[1]} bands")

# # # # ==== 2. 학습 / 평가 데이터 분리 ====

# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# # # # ==== 3. MLP 모델 학습 ====

# # # mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
# # # mlp.fit(X_train, y_train)

# # # # ==== 4. 평가 ====

# # # y_pred = mlp.predict(X_test)
# # # acc = accuracy_score(y_test, y_pred)
# # # print(f"Test accuracy: {acc:.4f}")
# # # print("Classification report:")
# # # print(classification_report(y_test, y_pred))

# # # # ==== 5. 중요 밴드 시각화 (feature importance 대신 weight의 절대값 사용) ====

# # # weights = np.abs(mlp.coefs_[0]).mean(axis=1)  # 입력층→첫 은닉층 가중치 절대값 평균

# # # plt.figure(figsize=(10,4))
# # # plt.bar(range(len(weights)), weights)
# # # plt.title("Average absolute weight per band (input layer)")
# # # plt.xlabel("Band Index")
# # # plt.ylabel("Weight magnitude")
# # # plt.show()

# # import os
# # import glob
# # import scipy.io
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder
# # import joblib
# # from model import build_model

# # # ==== 1. 데이터 로드 및 전처리 ====

# # mat_folder = '/Users/jeong-yeonghun/Desktop/2025_SZU_PROJECT/material/mat_data'
# # mat_files = sorted(glob.glob(os.path.join(mat_folder, '*.mat')))

# # labels_dict = {
# #     '5#.mat': 'polyester', '8#.mat': 'polyester', '11#.mat': 'polyester',
# #     '12#.mat': 'polyester', '14#.mat': 'polyester', '18#.mat': 'polyester',
# #     '19#.mat': 'polyester', '35#.mat': 'wool', '52#.mat': 'wool',
# #     '54#.mat': 'wool', '55#.mat': 'wool', '58#.mat': 'wool',
# #     '60#.mat': 'silk', '62#.mat': 'silk', '63#.mat': 'silk',
# #     '64#.mat': 'silk', '65#.mat': 'silk', '66#.mat': 'silk',
# #     '69#.mat': 'linen', '71#.mat': 'linen', '72#.mat': 'linen',
# #     '73#.mat': 'linen', '74#.mat': 'linen', '75#.mat': 'cotton',
# #     '76#.mat': 'cotton', '77#.mat': 'cotton', '79#.mat': 'cotton',
# #     '80#.mat': 'cotton', '81#.mat': 'cotton', '83#.mat': 'cotton',
# #     '99#.mat': 'recycled', '100#.mat': 'recycled'
# # }

# # X, y = [], []
# # crop_size = 50

# # print(f"Found {len(mat_files)} mat files.")

# # for filepath in mat_files:
# #     filename = os.path.basename(filepath)
# #     if filename not in labels_dict:
# #         print(f"Warning: No label for {filename}, skipping.")
# #         continue
# #     label = labels_dict[filename]
# #     mat = scipy.io.loadmat(filepath)
# #     if 'A' not in mat:
# #         print(f"Warning: 'A' key not found in {filename}, skipping.")
# #         continue
# #     data = mat['A']
# #     data = np.transpose(data, (1, 0, 2))
# #     H, W, B = data.shape
# #     y_start = H // 2 - crop_size // 2
# #     y_end = y_start + crop_size
# #     x_start = W // 2 - crop_size // 2
# #     x_end = x_start + crop_size
# #     cropped = data[y_start:y_end, x_start:x_end, :]
# #     mean_spectrum = np.mean(cropped, axis=(0, 1))
# #     X.append(mean_spectrum)
# #     y.append(label)
# #     print(f"Processed {filename} label={label}")

# # X = np.array(X)
# # y = np.array(y)
# # print(f"Total samples: {len(X)}, feature dim: {X.shape[1]}")

# # # 레이블 숫자 인코딩
# # le = LabelEncoder()
# # y_encoded = le.fit_transform(y)

# # # 저장
# # np.save('X_train.npy', X)
# # np.save('y_train.npy', y_encoded)
# # joblib.dump(le, 'label_encoder.pkl')

# # # ==== 2. 데이터 분할 및 학습 ====
# # X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42)

# # model = build_model()
# # model.fit(X_train, y_train)

# # joblib.dump(model, 'mlp_model.pkl')
# # print("✅ Model trained and saved to mlp_model.pkl")

# # second try

# # import os
# # import glob
# # import scipy.io
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.neural_network import MLPClassifier
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.metrics import classification_report, accuracy_score
# # import matplotlib.pyplot as plt

# # # ==== 1. 데이터 로드 및 전처리 ====
# # mat_folder = '/Users/jeong-yeonghun/Desktop/2025_SZU_PROJECT/material/mat_data'
# # mat_files = sorted(glob.glob(os.path.join(mat_folder, '*.mat')))

# # labels_dict = {
# #     '5#.mat': 'polyester', '8#.mat': 'polyester', '11#.mat': 'polyester',
# #     '12#.mat': 'polyester', '14#.mat': 'polyester', '18#.mat': 'polyester',
# #     '19#.mat': 'polyester', '35#.mat': 'wool', '52#.mat': 'wool',
# #     '54#.mat': 'wool', '55#.mat': 'wool', '58#.mat': 'wool',
# #     '60#.mat': 'silk', '62#.mat': 'silk', '63#.mat': 'silk',
# #     '64#.mat': 'silk', '65#.mat': 'silk', '66#.mat': 'silk',
# #     '69#.mat': 'linen', '71#.mat': 'linen', '72#.mat': 'linen',
# #     '73#.mat': 'linen', '74#.mat': 'linen', '75#.mat': 'cotton',
# #     '76#.mat': 'cotton', '77#.mat': 'cotton', '79#.mat': 'cotton',
# #     '80#.mat': 'cotton', '81#.mat': 'cotton', '83#.mat': 'cotton',
# #     '99#.mat': 'recycled', '100#.mat': 'recycled'
# # }

# # X, y = [], []
# # crop_size = 50

# # for filepath in mat_files:
# #     filename = os.path.basename(filepath)
# #     if filename not in labels_dict:
# #         continue
# #     label = labels_dict[filename]
# #     mat = scipy.io.loadmat(filepath)
# #     if 'A' not in mat:
# #         continue
# #     data = mat['A']
# #     data = np.transpose(data, (1, 0, 2))  # (W, H, B) → (H, W, B)
# #     H, W, B = data.shape
# #     y_start = H // 2 - crop_size // 2
# #     x_start = W // 2 - crop_size // 2
# #     cropped = data[y_start:y_start+crop_size, x_start:x_start+crop_size, :]
# #     if cropped.shape[:2] != (crop_size, crop_size):
# #         continue
# #     mean_spectrum = np.mean(cropped, axis=(0, 1))  # shape: (Bands,)
# #     if mean_spectrum.ndim != 1:
# #         continue
# #     X.append(mean_spectrum)
# #     y.append(label)

# # X = np.array(X)
# # le = LabelEncoder()
# # y_encoded = le.fit_transform(y)

# # print(f"Total samples: {X.shape[0]}, Bands: {X.shape[1]}")

# # # ==== 2. 전체 밴드로 초기 MLP 학습 ====
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42
# # )

# # model_full = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
# # model_full.fit(X_train, y_train)

# # # ==== 3. 입력층 weight 기반 밴드 중요도 계산 ====
# # weights = np.abs(model_full.coefs_[0])  # shape: (Bands, hidden)
# # importance = weights.mean(axis=1)       # shape: (Bands,)

# # top_k = 20
# # top_k_indices = np.argsort(importance)[-top_k:]

# # # 시각화 (선택)
# # plt.figure(figsize=(10, 4))
# # plt.bar(range(len(importance)), importance)
# # plt.title("Band Importance from Full MLP")
# # plt.xlabel("Band Index")
# # plt.ylabel("Importance (Avg. |Weight|)")
# # plt.axvline(top_k_indices.min(), color='r', linestyle='--', label='Top-K Range')
# # plt.axvline(top_k_indices.max(), color='r', linestyle='--')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# # # ==== 4. 상위 20개 밴드만 선택하여 재학습 ====
# # X_train_top = X_train[:, top_k_indices]
# # X_test_top = X_test[:, top_k_indices]

# # model_top = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
# # model_top.fit(X_train_top, y_train)
# # y_pred_top = model_top.predict(X_test_top)

# # acc = accuracy_score(y_test, y_pred_top)
# # print(f"\n✅ Test Accuracy (Top-{top_k} bands): {acc:.4f}")
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred_top, target_names=le.classes_))


# # train.py

# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from model import build_model

# # Load preprocessed data
# X = np.load('./processed/X.npy')
# y = np.load('./processed/y.npy')

# # Split data (70% train / 30% test)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, stratify=y, random_state=42
# )

# # Build and train model
# model = build_model()
# model.fit(X_train, y_train)

# # Save model and test data
# joblib.dump(model, './processed/mlp_model.pkl')
# np.save('./processed/X_test.npy', X_test)
# np.save('./processed/y_test.npy', y_test)

# print(f"\n✅ Model trained and saved to './processed/mlp_model.pkl'")
# print(f"📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from model import build_model
from load_data import load_data

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 5000
learning_rate = 0.00001

# 데이터 로드
X, y, label_encoder = load_data()

# train/test 분리 (70%/30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

input_dim = X.shape[1]
num_classes = len(label_encoder.classes_)

# 텐서 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델, 손실함수, 옵티마이저 생성
model = build_model(input_dim, num_classes, device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), './processed/mlp_model.pth')
joblib.dump(label_encoder, './processed/label_encoder.pkl')

print("✅ Training complete and model saved.")
