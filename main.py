import os
import numpy as np

# Import cấu hình
from src import config

# Import các pipeline đã viết
from src.preprocessing import download_pdb_data, preprocess_protein_data
from src.tda import process_tda
from src.features import compute_geometric_features, compute_pimage_features
from src.models import SVMClassifier
from src.utils import save_model, plot_confusion_matrix, plot_model_comparison

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

def main():
    print(f"{'='*60}\n DỰ ÁN TDA: PHÂN LOẠI CẤU TRÚC PROTEIN\n{'='*60}")

    print("\n[STEP 1] KHỞI CHẠY PIPELINE XỬ LÝ DỮ LIỆU...")
    
    # 1. Tải và Tiền xử lý (Nếu chưa có)
    fetched_groups = download_pdb_data(limit_per_class=20, raw_dir=config.RAW_DATA_DIR)
    preprocess_protein_data(fetched_groups, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR)
    
    # 2. Phân tích TDA (H1)
    process_tda() 
    
    # 3. Trích xuất Đặc trưng (Hình học & Vector hóa TDA)
    compute_geometric_features(config.PROCESSED_DATA_FILE, config.GEO_FEATURES_DIR)
    compute_pimage_features(config.PROCESSED_DATA_FILE, config.TDA_FEATURES_FILE, config.PIMAGE_FEATURES_DIR)

    # ==========================================
    # PHẦN 4: HUẤN LUYỆN VÀ SO SÁNH MÔ HÌNH
    # ==========================================
    print(f"\n{'='*60}\n BẮT ĐẦU HUẤN LUYỆN VÀ SO SÁNH MÔ HÌNH\n{'='*60}")

    # ---------------------------------------------------------
    # MÔ HÌNH 1: BASELINE (Đặc trưng hình học truyền thống)
    # ---------------------------------------------------------
    print("\n>>> 1. ĐÀO TẠO BASELINE MODEL (Hình học) <<<")
    X_geo, y_geo = load_dataset_for_model(config.GEO_FEATURES_FILE)
    
    baseline_svm = SVMClassifier(**config.SVM_PARAMS, verbose=False)
    metrics_geo = baseline_svm.fit(X_geo, y_geo)
    
    # Lấy model và scaler để lưu
    b_model, b_scaler = baseline_svm.get_model()
    save_model(b_model, b_scaler, save_dir=config.BASELINE_MODELS_DIR, name="svm_geometric")
    print(f"[*] Accuracy: {metrics_geo['accuracy']*100:.2f}%")
    print(f"[*] Đã lưu mô hình tại: {config.BASELINE_MODELS_DIR}")

    # ---------------------------------------------------------
    # MÔ HÌNH 2: TDA MODEL (Đặc trưng Persistence Image)
    # ---------------------------------------------------------
    print("\n>>> 2. ĐÀO TẠO TDA MODEL (Persistence Image) <<<")
    X_tda, y_tda = load_dataset_for_model(config.PIMAGE_FEATURES_FILE)
    
    tda_svm = SVMClassifier(**config.SVM_PARAMS, verbose=False)
    metrics_tda = tda_svm.fit(X_tda, y_tda)
    
    # Lấy model và scaler để lưu
    t_model, t_scaler = tda_svm.get_model()
    save_model(t_model, t_scaler, save_dir=config.TDA_MODELS_DIR, name="svm_pimage")
    print(f"[*] Accuracy: {metrics_tda['accuracy']*100:.2f}%")
    print(f"[*] Đã lưu mô hình tại: {config.TDA_MODELS_DIR}")

    # ==========================================
    # SO SÁNH TỔNG KẾT
    # ==========================================
    print(f"\n{'='*60}\n KẾT LUẬN SO SÁNH\n{'='*60}")
    print(f"Độ chính xác Phương pháp Truyền thống (Geometric)  : {metrics_geo['accuracy']*100:.2f}%")
    print(f"Độ chính xác Phương pháp TDA (Persistence Image)   : {metrics_tda['accuracy']*100:.2f}%")
    
    if metrics_tda['accuracy'] > metrics_geo['accuracy']:
        print("-> Nhận xét: Đặc trưng TDA (cấu trúc liên kết lỗ/vòng) mang lại khả năng phân loại protein tốt hơn hẳn so với các số đo hình học cơ bản.")
    else:
        print("-> Nhận xét: Đặc trưng hình học đã hoạt động rất tốt, nhưng TDA cung cấp một góc nhìn topo bổ sung rất giá trị cho các cấu trúc phức tạp.")

    print(f"\n{'='*60}\n VẼ BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH\n{'='*60}")
    
    # Định nghĩa thư mục lưu kết quả đánh giá (lưu chung với models)
    EVAL_DIR = os.path.join(config.PROJECT_ROOT, "results", "evaluations")
    
    # Lấy danh sách tên các class (label) để vẽ Confusion Matrix
    class_names = ["alpha_helix", "beta_sheet", "mixed"]

    # 1. Vẽ Confusion Matrix cho Model Hình học
    plot_confusion_matrix(
        cm=metrics_geo['confusion_matrix'], 
        classes=class_names, 
        title='Confusion Matrix - Hình học truyền thống', 
        save_path=os.path.join(EVAL_DIR, 'cm_geometric.png')
    )

    # 2. Vẽ Confusion Matrix cho Model TDA
    plot_confusion_matrix(
        cm=metrics_tda['confusion_matrix'], 
        classes=class_names, 
        title='Confusion Matrix - TDA (Persistence Image)', 
        save_path=os.path.join(EVAL_DIR, 'cm_tda.png')
    )

    # 3. Vẽ biểu đồ Bar Chart so sánh tổng quan 4 chỉ số
    plot_model_comparison(
        metrics_1=metrics_geo, 
        metrics_2=metrics_tda, 
        name_1="Geometric Features", 
        name_2="TDA Features", 
        save_path=os.path.join(EVAL_DIR, 'model_comparison_barchart.png')
    )

    print(f"\nKẾT LUẬN SO SÁNH:")
    print(f"- Accuracy Hình học: {metrics_geo['accuracy']*100:.2f}%")
    print(f"- Accuracy TDA     : {metrics_tda['accuracy']*100:.2f}%")
    print(f"\n[*] Toàn bộ biểu đồ đánh giá đã được xuất tại thư mục:\n -> {EVAL_DIR}")

if __name__ == "__main__":
    main()