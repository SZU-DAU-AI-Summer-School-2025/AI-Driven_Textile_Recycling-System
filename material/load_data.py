import numpy as np
import joblib

def load_data():
    X = np.load('./processed/X.npy')
    y = np.load('./processed/y.npy')
    label_encoder = joblib.load('./processed/label_encoder.pkl')
    return X, y, label_encoder
