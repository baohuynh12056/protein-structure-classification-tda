import os
import numpy as np
import time
# Import scikit-learn để đánh giá lại model khi load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder 
# Import cấu hình
from src import config

# Import các pipeline đã viết
from src import config
from src.preprocessing import download_pdb_data, preprocess_protein_data
from src.tda import process_tda
from src.features import compute_geometric_features, compute_pimage_features
from src.models import SVMClassifier
from src.utils import (
    save_model, load_model, 
    plot_confusion_matrix, plot_model_comparison,
    plot_feature_space, plot_pca_decision_boundary, log_training_results 
)

def load_dataset_for_model(filepath):
    """
    Hàm helper để load dữ liệu từ file .npy và tách X (features), y (labels)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
        
    dataset = np.load(filepath, allow_pickle=True)
    X = np.array([item["features"] for item in dataset])
    y = np.array([item["label"] for item in dataset])
    return X, y

def evaluate_loaded_model(svm_instance, X, y):
    """
    Hàm helper để lấy các chỉ số (metrics) từ model đã load mà không cần train lại.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=svm_instance.test_size,
        stratify=y,
        random_state=svm_instance.random_state
    )
    
    # Do hàm predict của SVMClassifier đã tích hợp sẵn self.scaler.transform
    y_pred = svm_instance.predict(X_test) 
    
    acc = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm
    }

def main():
    print(f"{'='*60}\n DỰ ÁN TDA: PHÂN LOẠI CẤU TRÚC PROTEIN\n{'='*60}")

    # ==========================================
    # PHẦN 1: TẢI & TIỀN XỬ LÝ
    # ==========================================
    print("\n[STEP 1] KIỂM TRA & TIỀN XỬ LÝ DỮ LIỆU...")
    if os.path.exists(config.PROCESSED_DATA_FILE):
        print("  -> [OK] Dữ liệu tiền xử lý đã tồn tại. Bỏ qua bước tải và chuẩn hóa.")
    else:
        fetched_groups = download_pdb_data(limit_per_class=20, raw_dir=config.RAW_DATA_DIR)
        preprocess_protein_data(fetched_groups, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR)
    
    # ==========================================
    # PHẦN 2: PHÂN TÍCH TDA
    # ==========================================
    print("\n[STEP 2] KIỂM TRA & PHÂN TÍCH TDA...")
    if os.path.exists(config.TDA_FEATURES_FILE):
        print("  -> [OK] Đặc trưng TDA (H1 Intervals) đã tồn tại. Bỏ qua bước tính toán Rips Filtration.")
    else:
        process_tda() 
    
    # ==========================================
    # PHẦN 3: TRÍCH XUẤT ĐẶC TRƯNG
    # ==========================================
    print("\n[STEP 3] KIỂM TRA & TRÍCH XUẤT ĐẶC TRƯNG...")
    if os.path.exists(config.GEO_FEATURES_FILE):
        print("  -> [OK] Đặc trưng Hình học đã tồn tại.")
    else:
        compute_geometric_features(config.PROCESSED_DATA_FILE, config.GEO_FEATURES_DIR)
        
    if os.path.exists(config.PIMAGE_FEATURES_FILE):
        print("  -> [OK] Đặc trưng Persistence Image đã tồn tại.")
    else:
        compute_pimage_features(config.PROCESSED_DATA_FILE, config.TDA_FEATURES_FILE, config.PIMAGE_FEATURES_DIR)

# ==========================================
    # PHẦN 4: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
    # ==========================================
    print(f"\n{'='*60}\n BẮT ĐẦU HUẤN LUYỆN VÀ SO SÁNH MÔ HÌNH\n{'='*60}")
    
    # Định nghĩa thư mục lưu log và ảnh
    EVAL_DIR = os.path.join(config.PROJECT_ROOT, "results", "evaluations")
    LOG_DIR = os.path.join(config.PROJECT_ROOT, "results")
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # Bộ chuyển đổi Nhãn Chữ -> Nhãn Số (Ví dụ: alpha_helix -> 0)
    le = LabelEncoder()
    class_names = ["alpha_helix", "beta_sheet", "mixed"]
    le.fit(class_names)

    # ---------------------------------------------------------
    # MÔ HÌNH 1: BASELINE (Hình học)
    # ---------------------------------------------------------
    print("\n>>> 1. ĐÀO TẠO BASELINE MODEL (Hình học) <<<")
    X_geo, y_geo_str = load_dataset_for_model(config.GEO_FEATURES_FILE)
    y_geo = le.transform(y_geo_str) # Chuyển thành 0, 1, 2
    
    baseline_svm = SVMClassifier(**config.SVM_PARAMS, verbose=False)
    
    start_time = time.time()
    # Force train để luôn đo được thời gian thực và lấy model mới (hoặc có thể dùng cơ chế caching cũ)
    metrics_geo = baseline_svm.fit(X_geo, y_geo) 
    exec_time_geo = time.time() - start_time
    
    b_model, b_scaler = baseline_svm.get_model()
    save_model(b_model, b_scaler, save_dir=config.BASELINE_MODELS_DIR, name="svm_geometric")
    
    # Ghi Log
    log_training_results(LOG_DIR, "Geometric Features", baseline_svm, metrics_geo, len(X_geo), y_geo, exec_time_geo)
    
    # Vẽ Biểu đồ Không gian 2D và Ranh giới
    plot_feature_space(X_geo, y_geo, class_names, title="PCA - Geometric Features", 
                       save_path=os.path.join(EVAL_DIR, "pca_geometric.png"))
    plot_pca_decision_boundary(X_geo, y_geo, config.PARAMS, class_names, 
                               title="Decision Boundary (PCA) - Geometric", 
                               save_path=os.path.join(EVAL_DIR, "decision_boundary_geometric.png"))

    # ---------------------------------------------------------
    # MÔ HÌNH 2: TDA MODEL (Persistence Image)
    # ---------------------------------------------------------
    print("\n>>> 2. ĐÀO TẠO TDA MODEL (Persistence Image) <<<")
    X_tda, y_tda_str = load_dataset_for_model(config.PIMAGE_FEATURES_FILE)
    y_tda = le.transform(y_tda_str)
    
    tda_svm = SVMClassifier(**config.SVM_PARAMS, verbose=False)
    
    start_time = time.time()
    metrics_tda = tda_svm.fit(X_tda, y_tda)
    exec_time_tda = time.time() - start_time
    
    t_model, t_scaler = tda_svm.get_model()
    save_model(t_model, t_scaler, save_dir=config.TDA_MODELS_DIR, name="svm_pimage")
    
    # Ghi Log
    log_training_results(LOG_DIR, "TDA (Persistence Image)", tda_svm, metrics_tda, len(X_tda), y_tda, exec_time_tda)
    
    # Vẽ Biểu đồ Không gian 2D và Ranh giới
    plot_feature_space(X_tda, y_tda, class_names, title="PCA - TDA Features", 
                       save_path=os.path.join(EVAL_DIR, "pca_tda.png"))
    plot_pca_decision_boundary(X_tda, y_tda, config.PARAMS, class_names, 
                               title="Decision Boundary (PCA) - TDA", 
                               save_path=os.path.join(EVAL_DIR, "decision_boundary_tda.png"))

    # ==========================================
    # SO SÁNH TỔNG KẾT VÀ VẼ BIỂU ĐỒ BAR/MATRIX
    # ==========================================
    print(f"\n{'='*60}\n ĐANG VẼ BIỂU ĐỒ ĐÁNH GIÁ CHUNG\n{'='*60}")
    
    plot_confusion_matrix(metrics_geo['confusion_matrix'], class_names, 'Confusion Matrix - Hình học', os.path.join(EVAL_DIR, 'cm_geometric.png'))
    plot_confusion_matrix(metrics_tda['confusion_matrix'], class_names, 'Confusion Matrix - TDA', os.path.join(EVAL_DIR, 'cm_tda.png'))
    plot_model_comparison(metrics_geo, metrics_tda, "Geometric", "TDA", os.path.join(EVAL_DIR, 'model_comparison_barchart.png'))

    print(f"\n[*] Toàn bộ Logs đã được lưu tại     : {os.path.join(LOG_DIR, 'training_logs.txt')}")
    print(f"[*] Toàn bộ Hình ảnh đã được xuất tại: {EVAL_DIR}")

if __name__ == "__main__":
    main()