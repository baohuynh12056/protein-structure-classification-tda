import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "protein_dataset_clean.npy")

TDA_FEATURES_DIR = os.path.join(PROJECT_ROOT, "features", "tda_features")
TDA_FEATURES_FILE = os.path.join(TDA_FEATURES_DIR, "h1_intervals_all.npy")

GEO_FEATURES_DIR = os.path.join(PROJECT_ROOT, "features", "geometric_features")
GEO_FEATURES_FILE = os.path.join(GEO_FEATURES_DIR, "geometric_features_all.npy")

PIMAGE_FEATURES_DIR = os.path.join(PROJECT_ROOT, "features", "pimage_features")
PIMAGE_FEATURES_FILE = os.path.join(PIMAGE_FEATURES_DIR, "pimage_features_all.npy")

BASELINE_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "baseline_models")
TDA_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "tda_models")

RCSB_BASE_URL = "https://files.rcsb.org/download"

PROTEIN_GROUPS = {
    "alpha_helix": ['1lt1', '3mgn', '3he4', '3miw', '3he5', '2yf2', '1ovv', '1jm0', '3lt6', '3h7x', '3lt7', '2xdj', '2zfc', '1ovr', '1fe6', '3if4', '1ec5', '2guv', '2q1k', '2wz7'],
    "beta_sheet":  ['7p93', '3lkf', '6u2s', '6u33', '7p8s', '6u3f', '1pvl', '4q7g', '4i0n', '1qwy', '5azo', '2gu1', '2b13', '2b44', '2b0p', '3msw', '4dix', '1h09', '7p8x', '1oba'],
    "mixed":       ['6e2a', '7l6p', '5gz2', '4xap', '5c7h', '5gvh', '5gyz', '2gjn', '2gjl', '6omz', '6cia', '6bka', '4pmj', '5v1t', '9ej8', '3bw2', '5syd', '4puw', '3bw3', '3hzs'],
}


SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "test_size": 0.3, 
    "random_state": 42
}

PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "random_state": 42
}

PI_PARAMS = {
    "bandwidth": 0.2,
    "weight": lambda x: x[1], 
    "resolution": [20, 20]    
}