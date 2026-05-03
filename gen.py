import json

notebook_data = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧬 BÁO CÁO TỔNG HỢP: PHÂN LOẠI CẤU TRÚC PROTEIN ỨNG DỤNG TDA\n",
    "\n",
    "Notebook này trình bày toàn bộ luồng thực thi (pipeline) của dự án. Bao gồm các bước: Tiền xử lý dữ liệu, Minh họa trực quan các khái niệm TDA (Vietoris-Rips, Persistence Diagram, Persistence Image) cho 3 cấu trúc đại diện, và cuối cùng là huấn luyện mô hình học máy (SVM) để so sánh hiệu suất."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Thư viện và Cấu hình"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import gudhi as gd\n",
    "from gudhi.representations import PersistenceImage\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Import modules từ project\n",
    "from src import config\n",
    "from src.preprocessing import download_pdb_data, preprocess_protein_data\n",
    "from src.tda import process_tda\n",
    "from src.features import compute_geometric_features, compute_pimage_features\n",
    "from src.models import SVMClassifier\n",
    "\n",
    "# IMPORT ĐÚNG CÁC HÀM CỦA BẠN\n",
    "from src.utils import (\n",
    "    plot_multidim_filtration, \n",
    "    plot_h1_persistence_diagram, \n",
    "    plot_persistence_image,\n",
    "    plot_confusion_matrix, \n",
    "    plot_model_comparison, \n",
    "    plot_feature_space, \n",
    "    plot_pca_decision_boundary\n",
    ")\n",
    "\n",
    "print(\"✅ Môi trường đã sẵn sàng!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Tải và Tiền xử lý Dữ liệu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"[*] Đang thực hiện Tải và Tiền xử lý dữ liệu...\")\n",
    "fetched_groups = download_pdb_data(limit_per_class=20, raw_dir=config.RAW_DATA_DIR)\n",
    "dataset = preprocess_protein_data(fetched_groups, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR)\n",
    "print(f\"\\n✅ Hoàn tất xử lý {len(dataset)} proteins.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Trực quan hóa TDA cho các Cấu trúc Đại diện\n",
    "Lặp qua 3 protein: `1ake`, `1cd8`, `1tim`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "representative_pids = ['1ake', '1cd8', '1tim']\n",
    "eps_list = [0.8, 1.2, 1.5, 2.0] \n",
    "\n",
    "pi_transformer = PersistenceImage(\n",
    "    bandwidth=config.PI_PARAMS[\"bandwidth\"],\n",
    "    weight=config.PI_PARAMS[\"weight\"],\n",
    "    resolution=config.PI_PARAMS[\"resolution\"]\n",
    ")\n",
    "\n",
    "for pid in representative_pids:\n",
    "    print(f\"\\n{'='*70}\\n 🌟 PHÂN TÍCH ĐẠI DIỆN: {pid.upper()}\\n{'='*70}\")\n",
    "    \n",
    "    record = next(item for item in dataset if item[\"pdb_id\"].lower() == pid)\n",
    "    points = record[\"point_cloud\"]\n",
    "    \n",
    "    # 3.1 VẼ RIPS FILTRATION 3D (Dùng hàm của bạn)\n",
    "    plot_multidim_filtration(points, eps_list, pid)\n",
    "    plt.show()\n",
    "    \n",
    "    rips_complex = gd.RipsComplex(points=points, max_edge_length=2.5)\n",
    "    st = rips_complex.create_simplex_tree(max_dimension=2)\n",
    "    persistence = st.persistence()\n",
    "    \n",
    "    # 3.2 VẼ PERSISTENCE DIAGRAM (Dùng hàm của bạn)\n",
    "    plot_h1_persistence_diagram(persistence, pid)\n",
    "    plt.show()\n",
    "    \n",
    "    h1_intervals = np.array([interval[1] for interval in persistence if interval[0] == 1])\n",
    "    if len(h1_intervals) == 0:\n",
    "        h1_intervals = np.array([[0.0, 0.0]])\n",
    "        \n",
    "    pi_vector = pi_transformer.fit_transform([h1_intervals])[0]\n",
    "    \n",
    "    # 3.3 VẼ PERSISTENCE IMAGE\n",
    "    plot_persistence_image(pi_vector, pid)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Xử lý Đặc trưng hàng loạt (Feature Extraction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"[*] Bước 1: Tính toán TDA...\")\n",
    "process_tda()\n",
    "print(\"\\n[*] Bước 2: Hình học Truyền thống...\")\n",
    "compute_geometric_features(config.PROCESSED_DATA_FILE, config.GEO_FEATURES_DIR)\n",
    "print(\"\\n[*] Bước 3: Vector hóa PD thành Persistence Image...\")\n",
    "compute_pimage_features(config.PROCESSED_DATA_FILE, config.TDA_FEATURES_FILE, config.PIMAGE_FEATURES_DIR)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Huấn luyện SVM và Đánh giá"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_dataset_for_model(filepath):\n",
    "    data = np.load(filepath, allow_pickle=True)\n",
    "    X = np.array([item[\"features\"] for item in data])\n",
    "    y = np.array([item[\"label\"] for item in data])\n",
    "    return X, y\n",
    "\n",
    "class_names = [\"alpha_helix\", \"beta_sheet\", \"mixed\"]\n",
    "le = LabelEncoder()\n",
    "le.fit(class_names)\n",
    "\n",
    "X_geo, y_geo_str = load_dataset_for_model(config.GEO_FEATURES_FILE)\n",
    "y_geo = le.transform(y_geo_str)\n",
    "svm_geo = SVMClassifier(**config.SVM_PARAMS, verbose=False)\n",
    "metrics_geo = svm_geo.fit(X_geo, y_geo)\n",
    "\n",
    "X_tda, y_tda_str = load_dataset_for_model(config.PIMAGE_FEATURES_FILE)\n",
    "y_tda = le.transform(y_tda_str)\n",
    "svm_tda = SVMClassifier(**config.SVM_PARAMS, verbose=False)\n",
    "metrics_tda = svm_tda.fit(X_tda, y_tda)\n",
    "\n",
    "plot_confusion_matrix(metrics_geo['confusion_matrix'], class_names, title='Confusion Matrix - Hình học')\n",
    "plt.show()\n",
    "plot_confusion_matrix(metrics_tda['confusion_matrix'], class_names, title='Confusion Matrix - TDA (PI)')\n",
    "plt.show()\n",
    "\n",
    "plot_feature_space(X_tda, y_tda, class_names, title=\"Không gian TDA (PCA 2D)\")\n",
    "plt.show()\n",
    "plot_pca_decision_boundary(X_tda, y_tda, config.SVM_PARAMS, class_names, title=\"Ranh giới phân loại TDA\")\n",
    "plt.show()\n",
    "\n",
    "plot_model_comparison(metrics_geo, metrics_tda, \"Geometric\", \"TDA\")\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, ensure_ascii=False, indent=2)

print("✅ Đã tạo thành công file TDA_Final_Report.ipynb đồng bộ với code của bạn!")