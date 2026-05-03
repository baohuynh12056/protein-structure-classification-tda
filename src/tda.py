import os
import numpy as np
import pandas as pd
import gudhi as gd

from src.utils import save_features_npy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "protein_dataset_clean.npy")
TDA_FEATURES_DIR = os.path.join(PROJECT_ROOT, "features", "tda_features")

def get_h1_persistence_stats(persistence, pid, threshold=0.4):
    """
    Trích xuất các thông số liên quan đến cấu trúc vòng (H1) từ persistence đã tính.
    """
    # Lọc lấy các intervals của chiều 1 (H1)
    h1 = np.array([interval[1] for interval in persistence if interval[0] == 1])

    if len(h1) == 0:
        return {"PDB_ID": pid.upper(), "So_vong_H1_Stable": 0, "Max_Death": 0, "Max_Persistence": 0}

    # Tính độ bền (Death - Birth)
    persistence_values = h1[:, 1] - h1[:, 0]
    stable_h1 = persistence_values[persistence_values > threshold]

    return {
        "PDB_ID": pid.upper(),
        "So_vong_H1_Stable": len(stable_h1),
        "Max_Death": round(float(np.max(h1[:, 1])), 2),
        "Max_Persistence": round(float(np.max(persistence_values)), 2)
    }

def process_tda():
    """
    Hàm thực thi chính: Xây dựng Rips, tính H1 và lưu đặc trưng.
    """
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"[!] Không tìm thấy dữ liệu: {PROCESSED_DATA_FILE}. Hãy chạy preprocessing.py trước!")
        return

    print(f"{'='*60}\n BẮT ĐẦU PHÂN TÍCH TDA (RIPS FILTRATION & H1)\n{'='*60}")
    dataset = np.load(PROCESSED_DATA_FILE, allow_pickle=True)
    
    all_h1_stats = []
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
        
        # 3. Trích xuất đặc trưng H1 và lưu vào danh sách thống kê
        stats = get_h1_persistence_stats(persistence, pid)
        all_h1_stats.append(stats)
        
        # Lưu toàn bộ chu kỳ H1 (Birth, Death) thô
        h1_intervals = np.array([interval[1] for interval in persistence if interval[0] == 1])
        tda_features_dict[pid] = h1_intervals

    # 4. Lưu trữ TDA Features (H1 Intervals) thô ra file .npy
    save_features_npy(tda_features_dict, TDA_FEATURES_DIR, "h1_intervals_all")
    print(f"\n[OK] Đã lưu dữ liệu H1 Intervals tại: {TDA_FEATURES_DIR}")

    # 5. In và lưu bảng tổng hợp
    df_h1_summary = pd.DataFrame(all_h1_stats).sort_values("PDB_ID")
    print("\nBẢNG THỐNG KÊ CẤU TRÚC VÒNG (H1) - Ngưỡng bền vững > 0.4:")
    print(df_h1_summary.to_string(index=False))
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    df_h1_summary.to_csv(os.path.join(PROJECT_ROOT, "results", "h1_summary.csv"), index=False)

if __name__ == "__main__":
    process_tda()