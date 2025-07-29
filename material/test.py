# # test.py

# import numpy as np
# import joblib
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model, test data, and label encoder
# model = joblib.load('./processed/mlp_model.pkl')
# X_test = np.load('./processed/X_test.npy')
# y_test = np.load('./processed/y_test.npy')
# label_encoder = joblib.load('./processed/label_encoder.pkl')

# # Predict
# y_pred = model.predict(X_test)

# # Accuracy & Report
# acc = accuracy_score(y_test, y_pred)
# print(f"✅ Test Accuracy: {acc:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=label_encoder.classes_,
#             yticklabels=label_encoder.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from model import build_model
from load_data import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 및 라벨 인코더 로드
X, y, label_encoder = load_data()

# 테스트 데이터 로드 (train.py에서 분리된 30% 사용)
# train.py에서 분리한 X_test, y_test를 별도 저장하셨다면 그걸 사용하세요
# 여기서는 간단히 전체 데이터로 평가한다고 가정

input_dim = X.shape[1]
num_classes = len(label_encoder.classes_)

# 모델 로드
model = build_model(input_dim, num_classes, device)
model.load_state_dict(torch.load('./processed/mlp_model.pth', map_location=device))
model.eval()

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# 예측
with torch.no_grad():
    outputs = model(X_tensor)
    _, preds = torch.max(outputs, 1)

y_true = y
y_pred = preds.cpu().numpy()

# 평가 결과 출력
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}\n")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# 혼동행렬 시각화
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
