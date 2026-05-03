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
    "alpha_helix": ["1ake", "2rhe", "1gcn"],
    "beta_sheet":  ["1cd8", "2pka", "1qqt"],
    "mixed":       ["1tim", "3tnd", "4hhb"],
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