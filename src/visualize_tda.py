import os
import numpy as np
import gudhi as gd

# Import các hàm vẽ từ utils.py
from src.utils import plot_h1_persistence_diagram, plot_multidim_filtration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "protein_dataset_clean.npy")
DIAGRAMS_DIR = os.path.join(PROJECT_ROOT, "tda", "diagrams")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "tda", "images")

REPRESENTATIVE_PROTEINS = ['1ake', '1cd8', '1tim']
EPSILON_LIST = [0.5, 0.9, 1.2, 1.5]

def generate_tda_visualizations():
    """
    Tùy chọn vẽ minh họa: Tính toán lại nhanh gọn cho các protein đại diện và xuất ảnh.
    """
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("[!] Cần chạy preprocessing.py trước để có dữ liệu vẽ.")
        return

    print(f"{'='*60}\n TẠO ẢNH MINH HỌA TDA CHO PROTEIN ĐẠI DIỆN\n{'='*60}")
    dataset = np.load(PROCESSED_DATA_FILE, allow_pickle=True)

    for record in dataset:
        pid = record["pdb_id"]
        
        # Chỉ vẽ cho các protein đại diện để tiết kiệm thời gian
        if pid in REPRESENTATIVE_PROTEINS:
            points = record["point_cloud"]
            print(f"[*] Đang vẽ biểu đồ cho {pid.upper()}...")

            # Khởi tạo lại Rips để vẽ (do không lưu object Rips từ file tda.py)
            rips = gd.RipsComplex(points=points, max_edge_length=2.0)
            st = rips.create_simplex_tree(max_dimension=2)
            persistence = st.persistence()

            # Vẽ và lưu ảnh
            plot_h1_persistence_diagram(persistence, pid, save_dir=DIAGRAMS_DIR)
            plot_multidim_filtration(points, EPSILON_LIST, pid, save_dir=IMAGES_DIR)

    print("\n[OK] Quá trình xuất ảnh hoàn tất!")

if __name__ == "__main__":
    generate_tda_visualizations()