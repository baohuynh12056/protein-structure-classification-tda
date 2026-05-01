import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gudhi as gd
import seaborn as sns
def save_features_npy(features, save_path, filename):
    """
    Save features to .npy file

    Args:
        features: list hoặc numpy array
        save_path: thư mục lưu
        filename: tên file (không cần .npy)
    """
    os.makedirs(save_path, exist_ok=True)

    # convert sang numpy array nếu cần
    features = np.array(features)

    full_path = os.path.join(save_path, f"{filename}.npy")
    np.save(full_path, features)

    return full_path


def load_features_npy(file_path):
    """
    Load features từ file .npy
    """
    return np.load(file_path, allow_pickle=True)


def save_model(model, scaler, save_dir="results/models", name="svm_traditional"):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, f"{name}.joblib"))
    joblib.dump(scaler, os.path.join(save_dir, f"{name}_scaler.joblib"))


def load_model(save_dir="results/models", name="svm_traditional"):
    model = joblib.load(os.path.join(save_dir, f"{name}.joblib"))
    scaler = joblib.load(os.path.join(save_dir, f"{name}_scaler.joblib"))

    return model, scaler

def plot_h1_persistence_diagram(persistence, pdb_id, save_dir=None):
    """
    Vẽ và lưu biểu đồ Persistence Diagram.
    """
    plt.figure(figsize=(7, 7))
    gd.plot_persistence_diagram(persistence, legend=True)
    plt.title(f"Persistence Diagram - Protein {pdb_id.upper()}")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Nếu có đường dẫn, lưu ảnh thay vì chỉ hiển thị
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{pdb_id}_diagram.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [+] Đã lưu biểu đồ Diagram: {save_path}")
    
    plt.close()


def plot_multidim_filtration(points, epsilon_list, pdb_id, save_dir=None):
    """
    Vẽ sự tiến hóa của Phức hợp Rips qua các ngưỡng epsilon khác nhau.
    """
    n_steps = len(epsilon_list)
    fig = plt.figure(figsize=(6 * n_steps, 7))

    print(f"\n--- Đang vẽ Filtration đa chiều cho Protein {pdb_id.upper()} ---")

    for i, eps in enumerate(epsilon_list):
        ax = fig.add_subplot(1, n_steps, i + 1, projection='3d')

        # Xây dựng Simplex Tree cho ngưỡng eps hiện tại để lấy simplices
        rips = gd.RipsComplex(points=points, max_edge_length=eps)
        st = rips.create_simplex_tree(max_dimension=4)
        simplices = list(st.get_skeleton(3))

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=5, alpha=0.3)

        edges, faces = [], []
        for simplex, _ in simplices:
            if len(simplex) == 2: # Cạnh (H1)
                edges.append(points[list(simplex)])
            elif len(simplex) == 3: # Mặt (H2)
                faces.append(points[list(simplex)])

        for edge in edges:
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], c='red', alpha=0.2, linewidth=0.5)

        if faces:
            poly3d = Poly3DCollection(faces, facecolors='cyan', linewidths=0.1, edgecolors='blue', alpha=0.1)
            ax.add_collection3d(poly3d)

        ax.set_title(f"Epsilon = {eps}\nEdges: {len(edges)} | Faces: {len(faces)}")
        ax.set_axis_off()
        ax.set_xlim([-2, 2]); ax.set_ylim([-2, 2]); ax.set_zlim([-2, 2])
        print(f"  [+] Epsilon {eps:0.2f}: {len(edges)} cạnh, {len(faces)} mặt tam giác.")

    plt.suptitle(f"Sự tiến hóa Phức hợp Vietoris-Rips (H1 - Đỏ, H2 - Xanh)\nProtein: {pdb_id.upper()}", fontsize=15)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{pdb_id}_filtration.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [+] Đã lưu biểu đồ Filtration: {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None):
    """
    Vẽ và lưu biểu đồ Confusion Matrix (Ma trận nhầm lẫn).
    """
    plt.figure(figsize=(6, 5))
    # Dùng seaborn heatmap để vẽ ma trận cho đẹp
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Nhãn thực tế (True Label)', fontsize=12)
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [+] Đã lưu biểu đồ: {save_path}")
    
    plt.close()

def plot_model_comparison(metrics_1, metrics_2, name_1="Geometric", name_2="TDA", save_path=None):
    """
    Vẽ biểu đồ cột (Bar chart) so sánh 4 chỉ số: Accuracy, Precision, Recall, F1.
    """
    # Lấy các chỉ số trung bình (Macro) để đánh giá tổng quan
    labels = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    
    vals_1 = [metrics_1['accuracy'], metrics_1['precision_macro'], 
              metrics_1['recall_macro'], metrics_1['f1_macro']]
              
    vals_2 = [metrics_2['accuracy'], metrics_2['precision_macro'], 
              metrics_2['recall_macro'], metrics_2['f1_macro']]

    x = np.arange(len(labels))  # Vị trí các nhóm cột
    width = 0.35  # Độ rộng của cột

    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Vẽ cột cho 2 model
    rects1 = ax.bar(x - width/2, vals_1, width, label=name_1, color='#4C72B0') # Xanh dương
    rects2 = ax.bar(x + width/2, vals_2, width, label=name_2, color='#DD8452') # Cam

    # Trang trí biểu đồ
    ax.set_ylabel('Điểm số (0.0 - 1.0)', fontsize=12)
    ax.set_title('SO SÁNH HIỆU SUẤT MÔ HÌNH: HÌNH HỌC vs TDA', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15) # Dư ra một chút phía trên để hiển thị số
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Thêm giá trị text trên đầu mỗi cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Cách đỉnh cột 3 points
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [+] Đã lưu biểu đồ so sánh: {save_path}")
        
    plt.close()