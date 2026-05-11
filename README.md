# 🧬 Protein Structure Classification using TDA

## 📌 Giới thiệu
Dự án này áp dụng **Topological Data Analysis (TDA)** để phân loại cấu trúc protein dựa trên đặc trưng hình học và topo.

Pipeline chính bao gồm:
- Tiền xử lý dữ liệu protein
- Trích xuất đặc trưng hình học
- Tính toán đặc trưng topo (Persistent Homology)
- Chuyển đổi thành Persistence Image
- Huấn luyện và so sánh mô hình


---

## 📂 Cấu trúc thư mục
```
protein-structure-classification-tda/
│
├── data/ # Dữ liệu thô và đã xử lý
├── results/
│ ├── evaluations/ # PCA, biểu đồ
│ ├── training_logs.txt
│
├── src/
│ ├── preprocessing.py
│ ├── tda.py
│ ├── features.py
│ ├── models.py
│ └── utils.py
│
├── config.py
├── main.py
└── notebook.ipynb
```
## 🚀 Cách chạy
### 1. Tạo môi trường ảo
```bash
python -m venv env
source env/bin/activate
```
### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
### 3. Chạy toàn bộ pipeline
```bash
python main.py
```
