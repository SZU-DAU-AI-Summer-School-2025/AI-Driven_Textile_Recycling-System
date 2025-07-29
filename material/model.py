# from sklearn.neural_network import MLPClassifier

# def build_model():
#     return MLPClassifier(
#         hidden_layer_sizes=(256, 128, 64, 32),  # 깊은 네트워크
#         activation='relu',                     # 비선형 활성화
#         solver='adam',                         # 안정적 최적화
#         alpha=1e-8,                            # L2 정규화
#         batch_size='auto',
#         learning_rate='adaptive',             # 성능 좋음
#         learning_rate_init=0.001,
#         max_iter=10000,
#         early_stopping=False,                  # 과적합 방지
#         random_state=42,
#         verbose=True                        # 학습 로그 출력
#     )

import torch
import torch.nn as nn

class CustomMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def build_model(input_dim, num_classes, device='cpu'):
    model = CustomMLP(input_dim, num_classes).to(device)
    return model
