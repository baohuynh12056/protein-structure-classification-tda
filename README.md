# Protein Structure Classification using TDA

## Giới thiệu
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
PROTEIN_TDA_PROJECT/
│
├── data/
│   ├── processed/
│   └── raw/
│
├── features/
│   ├── geometric_features/
│   ├── pimage_features/
│   └── tda_features/
│
├── models/
│   ├── baseline_models/
│   └── tda_models/
│
├── results/
│   ├── evaluations/
│   └── training_logs.txt
│
├── src/
│   ├── config.py
│   ├── features.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── tda.py
│   ├── utils.py
│   └── visualize_tda.py
│
├── tda/
│   ├── diagrams/
│   ├── images/
│   └── persistence_images/
│
├── main.py
├── notebook.ipynb
├── README.md
└── requirements.txt
```
## Cách chạy
### 1. Tạo môi trường ảo
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Windows
```bash
python -m venv venv
source venv/bin/activate
```
Sau khi kích hoạt thành công, tiền tố (venv) sẽ xuất hiện trên terminal.
### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
### 3. Chạy toàn bộ pipeline
```bash
python main.py
```
Pipeline sẽ tự động thực hiện:
1. Tiền xử lý dữ liệu protein
2.Xây dựng Vietoris–Rips Filtration
3.Tính Persistent Homology
4.Sinh Persistence Diagram
5.Vector hóa bằng Persistence Image
6.Huấn luyện mô hình SVM
7.Đánh giá hiệu năng mô hình
8.Lưu kết quả và trực quan hóa
##  Cách chạy
Khởi động Jupyter Notebook:
```bash
jupyter notebook
```
Sau đó mở file:
```bash
notebook.ipynb
```
Có thể:
- Chọn Run All để chạy toàn bộ notebook
- Hoặc dùng:
```bash
Shift + Enter
```
để thực thi từng ô mã riêng lẻ.
# Mã nguồn
GitHub Repository: https://github.com/baohuynh12056/protein-structure-classification-tda