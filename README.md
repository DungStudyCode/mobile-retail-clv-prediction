mobile-clv-prediction/
  ├── data/
  │   ├── raw/                # Dữ liệu gốc (không bao giờ sửa trực tiếp)
  │   └── processed/          # Dữ liệu sau khi làm sạch và tính toán RFM
  ├── notebooks/
  │   ├── 01_eda_analysis.ipynb       # Phân tích dữ liệu và trực quan hóa (Thành viên B)
  │   ├── 02_feature_engineering.ipynb # Trích xuất đặc trưng hành vi (Thành viên A, B)
  │   └── 03_model_training.ipynb     # Huấn luyện và so sánh mô hình (Thành viên C, D)
  ├── src/
  │   ├── __init__.py
  │   ├── data_preprocessing.py # Hàm xử lý dữ liệu dùng chung
  │   └── models.py             # Định nghĩa cấu trúc các mô hình
  ├── models/                   # Lưu trữ các file model đã train (.pkl, .bst)
  ├── reports/                  # Chứa file báo cáo (PDF) và hình ảnh kết quả
  │   └── figures/              # Các biểu đồ quan trọng dùng cho slide
  ├── app/                      # Code cho giao diện Demo (Streamlit)
  │   └── main.py
  ├── requirements.txt          # Danh sách thư viện (pandas, sklearn, xgboost...)
  └── README.md                 # Hướng dẫn chạy code và mô tả dự án
