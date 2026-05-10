import os
import numpy as np
import gudhi as gd

from src.utils import save_features_npy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "protein_dataset_clean.npy")
TDA_FEATURES_DIR = os.path.join(PROJECT_ROOT, "features", "tda_features")



def process_tda():
    """
    Hàm thực thi chính: Xây dựng Rips, tính H1 và lưu đặc trưng.
    """
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"[!] Không tìm thấy dữ liệu: {PROCESSED_DATA_FILE}. Hãy chạy preprocessing.py trước!")
        return

    print(f"{'='*60}\n BẮT ĐẦU PHÂN TÍCH TDA (RIPS FILTRATION & H1)\n{'='*60}")
    dataset = np.load(PROCESSED_DATA_FILE, allow_pickle=True)
    
    tda_features_dict = {}

    for record in dataset:
        pid = record["pdb_id"]
        points = record["point_cloud"]
        
        print(f"[*] Đang xử lý tính toán TDA cho {pid.upper()}...")

        # 1. Xây dựng Rips Complex & Simplex Tree
        rips = gd.RipsComplex(points=points, max_edge_length=2.0)
        st = rips.create_simplex_tree(max_dimension=2)
        
        # 2. Tính Persistence Diagram (Chỉ tính 1 lần duy nhất)
        persistence = st.persistence()
        
        # Lưu toàn bộ chu kỳ H1 (Birth, Death) thô
        h1_intervals = np.array([interval[1] for interval in persistence if interval[0] == 1])
        tda_features_dict[pid] = h1_intervals

    # 4. Lưu trữ TDA Features (H1 Intervals) thô ra file .npy
    save_features_npy(tda_features_dict, TDA_FEATURES_DIR, "h1_intervals_all")
    print(f"\n[OK] Đã lưu dữ liệu H1 Intervals tại: {TDA_FEATURES_DIR}")


if __name__ == "__main__":
    process_tda()
