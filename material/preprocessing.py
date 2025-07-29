# preprocessing.py

import os
import glob
import numpy as np
import scipy.io
import joblib
from sklearn.preprocessing import LabelEncoder

def extract_random_crops(data, num_crops=20, crop_size=50):
    H, W, B = data.shape
    spectra = []
    for _ in range(num_crops):
        if H < crop_size or W < crop_size:
            continue
        y = np.random.randint(0, H - crop_size)
        x = np.random.randint(0, W - crop_size)
        patch = data[y:y+crop_size, x:x+crop_size, :]
        mean_spectrum = np.mean(patch, axis=(0, 1))  # (Bands,)
        spectra.append(mean_spectrum)
    return spectra

def load_and_preprocess(mat_folder, num_crops=5, crop_size=50, save_dir='./processed'):
    os.makedirs(save_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(mat_folder, '*.mat')))

    labels_dict = {
        '5#.mat': 'polyester', '8#.mat': 'polyester', '11#.mat': 'polyester',
        '12#.mat': 'polyester', '14#.mat': 'polyester', '18#.mat': 'polyester',
        '19#.mat': 'polyester', '35#.mat': 'wool', '52#.mat': 'wool',
        '54#.mat': 'wool', '55#.mat': 'wool', '58#.mat': 'wool',
        '60#.mat': 'silk', '62#.mat': 'silk', '63#.mat': 'silk',
        '64#.mat': 'silk', '65#.mat': 'silk', '66#.mat': 'silk',
        '69#.mat': 'linen', '71#.mat': 'linen', '72#.mat': 'linen',
        '73#.mat': 'linen', '74#.mat': 'linen', '75#.mat': 'cotton',
        '76#.mat': 'cotton', '77#.mat': 'cotton', '79#.mat': 'cotton',
        '80#.mat': 'cotton', '81#.mat': 'cotton', '83#.mat': 'cotton',
        '99#.mat': 'recycled', '100#.mat': 'recycled'
    }

    X, y = [], []

    print(f"📂 Found {len(mat_files)} mat files.")

    for filepath in mat_files:
        filename = os.path.basename(filepath)
        if filename not in labels_dict:
            print(f"⚠ Skipping: no label for {filename}")
            continue

        label = labels_dict[filename]
        mat = scipy.io.loadmat(filepath)
        if 'A' not in mat:
            print(f"⚠ Skipping: 'A' key not found in {filename}")
            continue

        data = mat['A']
        if data.shape[0] < 10 or data.shape[1] < 10:
            print(f"⚠ Skipping: invalid shape for {filename}: {data.shape}")
            continue

        # Convert (W, H, B) to (H, W, B)
        if data.shape[2] < 5:
            print(f"⚠ Skipping: Not enough bands in {filename}")
            continue
        data = np.transpose(data, (1, 0, 2))

        spectra = extract_random_crops(data, num_crops=num_crops, crop_size=crop_size)
        for spec in spectra:
            X.append(spec)
            y.append(label)

        print(f"✅ Processed {filename}: {len(spectra)} crops")

    X = np.array(X)
    y = np.array(y)
    print(f"\n📊 Final dataset: {X.shape[0]} samples, {X.shape[1]} bands")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save outputs
    np.save(os.path.join(save_dir, 'X.npy'), X)
    np.save(os.path.join(save_dir, 'y.npy'), y_encoded)
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))

    print(f"\n💾 Saved X.npy, y.npy, and label_encoder.pkl to {save_dir}")


load_and_preprocess(
    mat_folder='/Users/jeong-yeonghun/Desktop/2025_SZU_PROJECT/material/mat_data',
    num_crops=20,
    crop_size=50,
    save_dir='./processed'
)