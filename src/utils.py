import os
import numpy as np
import joblib

def save_features_npy(features, save_path, filename):
    """
    Save features to .npy file

    Args:
        features: list hoặc numpy array
        save_path: thư mục lưu
        filename: tên file (không cần .npy)
    """
    os.makedirs(save_path, exist_ok=True)

    # convert sang numpy array nếu cần
    features = np.array(features)

    full_path = os.path.join(save_path, f"{filename}.npy")
    np.save(full_path, features)

    return full_path


def load_features_npy(file_path):
    """
    Load features từ file .npy
    """
    return np.load(file_path)


def save_model(model, scaler, save_dir="results/models", name="svm_traditional"):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, f"{name}.joblib"))
    joblib.dump(scaler, os.path.join(save_dir, f"{name}_scaler.joblib"))


def load_model(save_dir="results/models", name="svm_traditional"):
    model = joblib.load(os.path.join(save_dir, f"{name}.joblib"))
    scaler = joblib.load(os.path.join(save_dir, f"{name}_scaler.joblib"))

    return model, scaler