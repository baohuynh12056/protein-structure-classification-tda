import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gudhi as gd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from datetime import datetime
from src import config

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

    plt.suptitle(f"Sự tiến hóa Phức hợp Vietoris-Rips \nProtein: {pdb_id.upper()}", fontsize=15)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{pdb_id}_filtration.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [+] Đã lưu biểu đồ Filtration: {save_path}")
    
        plt.close()

def plot_persistence_image(pi_matrix, pdb_id, save_dir=None):
    """
    Vẽ và lưu biểu đồ Persistence Image (Vector hóa từ Diagram thành Heatmap).
    """
    plt.figure(figsize=(7, 6))
    

    resolution = config.PI_PARAMS["resolution"][0] 
    pi_image = pi_matrix.reshape(resolution, resolution)

    sns.heatmap(pi_image, cmap='magma', annot=False, cbar=True)
    
    # Trang trí biểu đồ
    plt.title(f"Persistence Image (PI) - Protein {pdb_id.upper()}\nVector {resolution*resolution} chiều (Reshape {resolution}x{resolution})", 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Birth Pixel", fontsize=11)
    plt.ylabel("Persistence Pixel", fontsize=11)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{pdb_id}_persistence_image.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [+] Đã lưu biểu đồ Persistence Image: {save_path}")
        
        plt.close()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None):
    """
    Vẽ và lưu biểu đồ Confusion Matrix
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

    ax.set_ylabel('Điểm số (0.0 - 1.0)', fontsize=12)
    ax.set_title('SO SÁNH HIỆU SUẤT MÔ HÌNH: HÌNH HỌC vs TDA', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15) 
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

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

def plot_feature_space(X, y, class_names, title="PCA Feature Space", save_path=None, method='pca'):
    """
    Giảm chiều dữ liệu xuống 2D bằng PCA hoặc t-SNE để trực quan hóa sự phân tách các cụm.
    """
    plt.figure(figsize=(8, 6))
    
    # Giảm chiều
    if method.lower() == 'tsne':
        # Perplexity nên nhỏ hơn số lượng sample. Với 60 sample, perplexity = 15 là hợp lý.
        reducer = TSNE(n_components=2, perplexity=15, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
        
    X_2d = reducer.fit_transform(X)
    
    # Vẽ Scatter plot
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1], 
        hue=[class_names[label] for label in y], # Chuyển số thành tên nhãn
        palette=['#4C72B0', '#DD8452', '#55A868'], # Xanh, Cam, Xanh lá
        s=100, alpha=0.8, edgecolor='k'
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.legend(title="Protein Class")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [+] Đã lưu biểu đồ {method.upper()}: {save_path}")
        plt.close()

def plot_pca_decision_boundary(X, y, svm_params, class_names, title="Decision Boundary", save_path=None):
    """
    Ép X xuống 2D qua PCA, sau đó train 1 mô hình SVM (cùng params) để vẽ ranh giới 2D.
    """
    # 1. Ép dữ liệu xuống 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # 2. Train mô hình SVM xấp xỉ trên không gian 2D
    svm_2d = SVC(**svm_params)
    svm_2d.fit(X_pca, y)
    
    # 3. Tạo lưới điểm (Grid) để vẽ contour
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 4. Vẽ bản đồ ranh giới
    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#A0CBE8', '#FFBE7D', '#8CD17D'])
    cmap_bold = ListedColormap(['#4C72B0', '#DD8452', '#55A868'])
    
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
    
    # Vẽ các điểm dữ liệu thực tế lên trên
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap_bold, 
                          edgecolor='k', s=60, alpha=0.9)
    
    # Chú thích
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_bold(i), markersize=10, markeredgecolor='k') for i in range(len(class_names))]
    plt.legend(handles, class_names, title="Classes", loc='best')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [+] Đã lưu ranh giới quyết định: {save_path}")
        plt.close()

def log_training_results(log_dir, model_name, svm_instance, metrics, X_len, y, execution_time):
    """
    Ghi nhật ký chi tiết quá trình huấn luyện vào file txt.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_logs.txt")
    
    # Trích xuất thông tin
    model = svm_instance.model
    test_size_ratio = svm_instance.test_size
    n_test = int(X_len * test_size_ratio)
    n_train = X_len - n_test
    
    # Số lượng support vectors cho từng class
    n_supports = model.n_support_
    total_support = np.sum(n_supports)
    
    # Format Confusion Matrix cho đẹp
    cm_str = np.array2string(metrics['confusion_matrix'], separator=', ')
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_content = f"""
=========================================================
[{current_time}] MODEL: {model_name.upper()}
=========================================================
1. Hyperparameters:
   - Kernel      : {model.kernel}
   - C (Penalty) : {model.C}
   - Gamma       : {model.gamma}

2. Data Split:
   - Total samples : {X_len}
   - Train / Test  : {n_train} / {n_test} (Test size: {test_size_ratio*100}%)

3. Performance Metrics:
   - Accuracy    : {metrics['accuracy']*100:.2f}%
   - F1 (Macro)  : {metrics['f1_macro']:.4f}
   - Precision   : {metrics['precision_macro']:.4f}
   - Recall      : {metrics['recall_macro']:.4f}

4. Model Complexity:
   - Total Support Vectors : {total_support}
   - SVs per class         : {n_supports}

5. Execution:
   - Training + Eval Time  : {execution_time:.4f} seconds

6. Confusion Matrix:
{cm_str}
=========================================================
"""
    # Ghi nối (append) vào file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_content)
    print(f"  [+] Đã ghi log thành công vào: {log_file}")