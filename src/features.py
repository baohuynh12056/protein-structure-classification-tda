import os
import numpy as np
import pandas as pd
import warnings
from scipy.spatial import cKDTree
from gudhi.representations import PersistenceImage

# Import hàm lưu dữ liệu từ utils.py
from src.utils import save_features_npy

def extract_single_geometric(points):
    """
    Tính toán 4 đặc trưng hình học truyền thống cho một point cloud duy nhất.
    """
    num_points = points.shape[0]
    
    # Bán kính bao
    centroid = np.mean(points, axis=0)
    bounding_radius = np.max(np.linalg.norm(points - centroid, axis=1))

    # Khoảng cách lân cận trung bình
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    mean_neighbor_distance = np.mean(distances[:, 1])

    # Độ cong trung bình (PCA cục bộ)
    k_neighbors = min(10, num_points)
    _, indices = tree.query(points, k=k_neighbors)
    
    curvatures = []
    for neighbors_idx in indices:
        local_points = points[neighbors_idx]
        local_centroid = np.mean(local_points, axis=0)
        cov_matrix = np.cov((local_points - local_centroid).T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        sum_eigen = np.sum(eigenvalues)
        curvature = eigenvalues[0] / sum_eigen if sum_eigen > 0 else 0
        curvatures.append(curvature)
        
    mean_curvature = np.mean(curvatures)

    return [num_points, bounding_radius, mean_curvature, mean_neighbor_distance]


def compute_geometric_features(processed_data_path, output_dir):
    """
    Load điểm 3D, trích xuất đặc trưng hình học và lưu trữ.
    """
    print("\n[*] Đang tính toán Đặc trưng hình học (Geometric Features)...")
    
    if not os.path.exists(processed_data_path):
        print(f"  [!] Lỗi: Không tìm thấy file {processed_data_path}")
        return

    dataset = np.load(processed_data_path, allow_pickle=True)
    
    geo_features_list = []
    geo_stats = []

    for record in dataset:
        pid = record["pdb_id"]
        label = record["label"]
        points = record["point_cloud"]
        
        # Gọi hàm lõi
        f_geom = extract_single_geometric(points)
        
        # Đóng gói kết quả
        geo_features_list.append({
            "pdb_id": pid,
            "label": label,
            "features": np.array(f_geom)
        })
        geo_stats.append([pid.upper(), label] + f_geom)

    # Lưu dữ liệu
    save_features_npy(geo_features_list, output_dir, "geometric_features_all")
    print(f"  -> [OK] Đã lưu thành công tại: {output_dir}")

    # In bảng thống kê
    df_geo = pd.DataFrame(geo_stats, columns=["PDB_ID", "Label", "Num Points", "Bounding Radius", "Mean Curvature", "Mean Neighbor Dist"])
    print(df_geo.round(4).to_string(index=False))


def compute_pimage_features(processed_data_path, tda_features_path, output_dir):
    """
    Load H1 Intervals, vector hóa thành Persistence Image và lưu trữ.
    """
    print("\n[*] Đang Vector hóa Persistence Diagram thành Persistence Image...")
    
    if not os.path.exists(processed_data_path) or not os.path.exists(tda_features_path):
        print("  [!] Lỗi: Thiếu dữ liệu đầu vào. Hãy chắc chắn đã chạy tda.py!")
        return

    # Load dữ liệu gốc (để lấy Label) và dữ liệu TDA
    dataset = np.load(processed_data_path, allow_pickle=True)
    h1_intervals_dict = np.load(tda_features_path, allow_pickle=True).item()
    
    diags = []
    for record in dataset:
        pid = record["pdb_id"]
        diag = h1_intervals_dict.get(pid, [])
        # Nếu protein không có vòng H1, tạo mảng rỗng hợp lệ để tránh lỗi
        if len(diag) == 0:
            diag = np.array([[0.0, 0.0]])
        diags.append(diag)

    # Cấu hình transformer Persistence Image
    pi = PersistenceImage(
        bandwidth=0.2, 
        weight=lambda x: x[1], 
        resolution=[20, 20] # Vector 400 chiều
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_images_matrix = pi.fit_transform(diags)

    pimage_features_list = []
    for i, record in enumerate(dataset):
        pimage_features_list.append({
            "pdb_id": record["pdb_id"],
            "label": record["label"],
            "features": p_images_matrix[i]
        })

    # Lưu dữ liệu
    save_features_npy(pimage_features_list, output_dir, "pimage_features_all")
    print(f"  -> [OK] Đã lưu Vector P-Image tại: {output_dir}")
    print(f"  -> Kích thước mỗi Vector: {p_images_matrix.shape[1]} chiều.")



def main():
    print(f"{'='*60}\n BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG (FEATURES EXTRACTION)\n{'='*60}")
    
    # Thiết lập đường dẫn động dựa trên vị trí file hiện tại
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Định nghĩa các đường dẫn Input
    PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "protein_dataset_clean.npy")
    TDA_FEATURES = os.path.join(PROJECT_ROOT, "features", "tda_features", "h1_intervals_all.npy")
    
    # Định nghĩa các đường dẫn Output
    GEO_OUT_DIR = os.path.join(PROJECT_ROOT, "features", "geometric_features")
    PIMAGE_OUT_DIR = os.path.join(PROJECT_ROOT, "features", "pimage_features")

    # Truyền đường dẫn vào các hàm xử lý
    compute_geometric_features(
        processed_data_path=PROCESSED_DATA, 
        output_dir=GEO_OUT_DIR
    )
    
    compute_pimage_features(
        processed_data_path=PROCESSED_DATA,
        tda_features_path=TDA_FEATURES,
        output_dir=PIMAGE_OUT_DIR
    )
    
    print(f"\n{'='*60}\n [HOÀN TẤT] Mọi dữ liệu đã sẵn sàng cho mô hình Machine Learning!\n{'='*60}")

if __name__ == "__main__":
    main()