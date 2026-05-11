import os
import numpy as np
import gudhi as gd

from src.utils import save_features_npy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "protein_dataset_clean.npy")
TDA_FEATURES_DIR = os.path.join(PROJECT_ROOT, "features", "tda_features")

def get_h1_persistence_stats(persistence, pid, label, threshold=0):
    """
    Trích xuất các thông số vòng (H1), bao gồm cả Thời gian tồn tại (Độ bền).
    """
    h1 = np.array([interval[1] for interval in persistence if interval[0] == 1])

    if len(h1) == 0:
        return {
            "PDB_ID": pid.upper(), "Label": label, 
            "So_Vong_H1": 0, "Max_Death": 0, 
            "Max_Persistence": 0, "Mean_Persistence": 0
        }

    # Thời gian tồn tại (Độ bền) = Death - Birth
    persistence_values = h1[:, 1] - h1[:, 0]
    
    # Chỉ tính các vòng ổn định (lọc bỏ nhiễu nhỏ hơn threshold)
    stable_h1 = persistence_values[persistence_values > threshold]

    if len(stable_h1) == 0:
        return {
            "PDB_ID": pid.upper(), "Label": label, 
            "So_Vong_H1": 0, 
            "Max_Death": round(float(np.max(h1[:, 1])), 3),
            "Max_Persistence": round(float(np.max(persistence_values)), 3),
            "Mean_Persistence": round(float(np.mean(persistence_values)), 3)
        }

    return {
        "PDB_ID": pid.upper(),
        "Label": label,
        "So_Vong_H1": len(stable_h1),
        "Max_Death": round(float(np.max(h1[:, 1])), 3),
        "Max_Persistence": round(float(np.max(stable_h1)), 3),
        "Mean_Persistence": round(float(np.mean(stable_h1)), 3) # Thời gian tồn tại trung bình
    }

def process_tda():
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"[!] Không tìm thấy dữ liệu: {PROCESSED_DATA_FILE}. Hãy chạy preprocessing.py trước!")
        return

    print(f"{'='*60}\n BẮT ĐẦU PHÂN TÍCH TDA (RIPS FILTRATION & H1)\n{'='*60}")
    dataset = np.load(PROCESSED_DATA_FILE, allow_pickle=True)
    
    tda_features_dict = {}

    for record in dataset:
        pid = record["pdb_id"]
        label = record["label"]
        points = record["point_cloud"]
        
        # 1. Xây dựng Rips Complex
        rips = gd.RipsComplex(points=points, max_edge_length=2.5)
        st = rips.create_simplex_tree(max_dimension=2)
        
        # 2. Tính Persistence
        persistence = st.persistence()
        
        # 3. Thống kê H1
        stats = get_h1_persistence_stats(persistence, pid, label)
        all_h1_stats.append(stats)
        
        # Lưu TDA Features thô
        h1_intervals = np.array([interval[1] for interval in persistence if interval[0] == 1])
        tda_features_dict[pid] = h1_intervals

    save_features_npy(tda_features_dict, TDA_FEATURES_DIR, "h1_intervals_all")

    # ========================================================
    # 4. XỬ LÝ VÀ IN THỐNG KÊ CHI TIẾT THEO NHÓM
    # ========================================================
    df_h1_summary = pd.DataFrame(all_h1_stats)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    df_h1_summary.to_csv(os.path.join(PROJECT_ROOT, "results", "h1_summary_per_protein.csv"), index=False)
    
    print(f"\n{'='*70}\n BẢNG PHÂN TÍCH THỜI GIAN TỒN TẠI VÒNG H1 THEO TỪNG NHÓM\n{'='*70}")
    
    # Nhóm dữ liệu để phân tích (Tính Tổng số vòng, Trung bình vòng, và Độ bền trung bình)
    group_stats = df_h1_summary.groupby("Label").agg(
        Tong_So_Vong_H1=('So_Vong_H1', 'sum'),
        TB_So_Vong_Moi_Protein=('So_Vong_H1', 'mean'),
        Do_Ben_H1_Trung_Binh=('Mean_Persistence', 'mean'), # Đây chính là Lifetime trung bình
        Do_Ben_H1_Lon_Nhat=('Max_Persistence', 'max')
    ).reset_index()
    
    # Làm tròn số cho đẹp
    group_stats = group_stats.round(3)
    print(group_stats.to_string(index=False))
    group_stats.to_csv(os.path.join(PROJECT_ROOT, "results", "h1_summary_by_group.csv"), index=False)

if __name__ == "__main__":
    process_tda()
