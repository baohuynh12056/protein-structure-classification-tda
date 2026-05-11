import os
import numpy as np
import gudhi as gd
from gudhi.representations import PersistenceImage

# Import cấu hình và các hàm vẽ từ utils.py
from src import config
from src.utils import (
    plot_h1_persistence_diagram, 
    plot_multidim_filtration, 
    plot_persistence_image
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "protein_dataset_clean.npy")
DIAGRAMS_DIR = os.path.join(PROJECT_ROOT, "tda", "diagrams")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "tda", "images")
PI_IMAGES_DIR = os.path.join(PROJECT_ROOT, "tda", "persistence_images") # Thư mục lưu PI

REPRESENTATIVE_PROTEINS = ['1lt1', '7p93', '6e2a']
EPSILON_LIST = [0.5, 0.9, 1.2, 1.5]

def generate_tda_visualizations():
    """
    Tùy chọn vẽ minh họa: Tính toán lại nhanh gọn cho các protein đại diện và xuất ảnh.
    Bao gồm: Rips Filtration, Persistence Diagram và Persistence Image.
    """
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("[!] Cần chạy preprocessing.py trước để có dữ liệu vẽ.")
        return

    print(f"{'='*60}\n TẠO ẢNH MINH HỌA TDA CHO PROTEIN ĐẠI DIỆN\n{'='*60}")
    dataset = np.load(PROCESSED_DATA_FILE, allow_pickle=True)

    # Khởi tạo bộ chuyển đổi PI từ tham số trong config
    pi_transformer = PersistenceImage(
        bandwidth=config.PI_PARAMS["bandwidth"],
        weight=config.PI_PARAMS["weight"],
        resolution=config.PI_PARAMS["resolution"]
    )

    for record in dataset:
        # Đưa về chữ thường để so sánh an toàn
        pid = record["pdb_id"].lower() 
        
        if pid in REPRESENTATIVE_PROTEINS:
            points = record["point_cloud"]
            print(f"\n[*] Đang vẽ biểu đồ cho {pid.upper()}...")

            rips = gd.RipsComplex(points=points, max_edge_length=2.0)
            st = rips.create_simplex_tree(max_dimension=2)
            persistence = st.persistence()

            # 1. Vẽ và lưu Persistence Diagram
            plot_h1_persistence_diagram(persistence, pid, save_dir=DIAGRAMS_DIR)
            
            # 2. Vẽ và lưu Multidimensional Filtration
            plot_multidim_filtration(points, EPSILON_LIST, pid, save_dir=IMAGES_DIR)

            # 3. Tính toán và vẽ Persistence Image
            # Lấy chu kỳ H1
            h1_intervals = np.array([interval[1] for interval in persistence if interval[0] == 1])
            
            # Tránh lỗi khi protein không có vòng H1 nào
            if len(h1_intervals) == 0:
                h1_intervals = np.array([[0.0, 0.0]])
                
            # Transform thành vector (Flatten)
            pi_vector = pi_transformer.fit_transform([h1_intervals])[0]
            
            # Gọi hàm vẽ và truyền vào vector (hàm utils sẽ tự reshape nó theo config)
            plot_persistence_image(pi_vector, pid, save_dir=PI_IMAGES_DIR)

    print("\n[OK] Quá trình xuất ảnh hoàn tất!")

if __name__ == "__main__":
    generate_tda_visualizations()