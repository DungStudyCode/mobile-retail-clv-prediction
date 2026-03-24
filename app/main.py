# app\main.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Cấu hình trang Web
st.set_page_config(
    page_title="Dự báo Khách hàng VIP - Nhóm 10",
    page_icon="📱",
    layout="wide"
)

# 2. Tiêu đề và giới thiệu
st.title("📱 Hệ Thống Phân Lớp Khách Hàng Tiềm Năng (VIP)")
st.markdown("""
**Đồ án môn học: Học máy (Machine Learning) - Nhóm 10**  
Hệ thống sử dụng mô hình AI để dự báo xác suất một khách hàng trở thành khách hàng VIP (chi tiêu cao, mua sắm thường xuyên) dựa trên lịch sử giao dịch.
""")
st.divider()

# 3. Tạo Sidebar để nhập thông tin khách hàng
st.sidebar.header("🔍 Nhập thông tin Khách hàng")

def user_input_features():
    recency = st.sidebar.number_input("1. Recency (Số ngày từ lần cuối mua)", min_value=0, max_value=1000, value=30)
    frequency = st.sidebar.number_input("2. Frequency (Tổng số đơn hàng)", min_value=1, max_value=100, value=5)
    monetary = st.sidebar.number_input("3. Monetary (Tổng tiền đã chi - VNĐ)", min_value=0, value=15000000, step=1000000)
    
    st.sidebar.markdown("---")
    installment_rate = st.sidebar.slider("4. Tỷ lệ mua trả góp", 0.0, 1.0, 0.3)
    flagship_ratio = st.sidebar.slider("5. Tỷ lệ mua dòng Flagship", 0.0, 1.0, 0.5)
    accessories_ratio = st.sidebar.slider("6. Tỷ lệ mua Phụ kiện", 0.0, 1.0, 0.2)
    
    st.sidebar.markdown("---")
    favorite_brand = st.sidebar.selectbox("7. Thương hiệu yêu thích", ("Apple", "Samsung", "Xiaomi", "Oppo"))
    
    # Tạo DataFrame từ input
    data = {
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'Installment_Rate': installment_rate,
        'Flagship_Ratio': flagship_ratio,
        'Accessories_Ratio': accessories_ratio,
        'Favorite_Brand': favorite_brand
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Hiển thị thông tin vừa nhập
st.subheader("Bảng thông tin đầu vào:")
st.dataframe(input_df, hide_index=True)

# 5. Xử lý dữ liệu và Dự báo
# Xác định thư mục chứa model
current_dir = os.getcwd()
models_dir = os.path.join(current_dir, 'models')

model_path = os.path.join(models_dir, 'best_xgb_model.pkl')
scaler_path = os.path.join(models_dir, 'scaler.pkl')

if st.button("🚀 Dự Báo Khách Hàng Này", type="primary"):
    try:
        # Load mô hình và scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Tiền xử lý input cho khớp với mô hình đã train (One-Hot Encoding)
        # Giả định các cột lúc train là: Recency, Frequency, Monetary, Installment_Rate, Flagship_Ratio, Accessories_Ratio, Favorite_Brand_Oppo, Favorite_Brand_Samsung, Favorite_Brand_Xiaomi
        
        # Khởi tạo các cột Brand bằng 0
        input_df['Favorite_Brand_Oppo'] = 0
        input_df['Favorite_Brand_Samsung'] = 0
        input_df['Favorite_Brand_Xiaomi'] = 0
        
        # Lấy giá trị thương hiệu mà người dùng đã chọn từ input_df
        selected_brand = input_df['Favorite_Brand'][0]
        
        # Cập nhật giá trị nếu không phải là Apple
        if selected_brand == 'Oppo': input_df['Favorite_Brand_Oppo'] = 1
        elif selected_brand == 'Samsung': input_df['Favorite_Brand_Samsung'] = 1
        elif selected_brand == 'Xiaomi': input_df['Favorite_Brand_Xiaomi'] = 1
            
        # Bỏ cột Favorite_Brand dạng chữ đi
        features = input_df.drop('Favorite_Brand', axis=1)
        
        # Scaler dữ liệu
        features_scaled = scaler.transform(features)
        
        # Dự đoán
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        st.divider()
        st.subheader("🎯 Kết Quả Dự Báo")
        
        col1, col2 = st.columns(2)
        
        if prediction[0] == 1:
            col1.success("🌟 ĐÂY LÀ KHÁCH HÀNG VIP")
            col2.metric(label="Xác suất trở thành VIP", value=f"{prediction_proba[0][1] * 100:.2f} %")
            st.info("💡 **Hành động đề xuất:** Khách hàng này có giá trị cao. Hãy gửi tặng mã giảm giá 1.000.000đ cho lần lên đời điện thoại tiếp theo để giữ chân họ!")
        else:
            col1.warning("👤 Khách Hàng Tiêu Chuẩn (Thường)")
            col2.metric(label="Xác suất trở thành VIP", value=f"{prediction_proba[0][1] * 100:.2f} %")
            st.info("💡 **Hành động đề xuất:** Khách hàng này chưa có thói quen chi tiêu lớn. Hãy gửi các gói combo khuyến mãi phụ kiện hoặc dịch vụ vệ sinh máy miễn phí.")
            
    except FileNotFoundError:
        st.error("🚨 LỖI: Không tìm thấy file mô hình (`best_xgb_model.pkl` hoặc `scaler.pkl`) trong thư mục `models/`.")
        st.warning("👉 Hãy chắc chắn rằng bạn đã chạy thành công file `02_model_training.ipynb` để hệ thống lưu file mô hình trước khi mở giao diện này.")
        
st.markdown("---")
st.caption("Giao diện được xây dựng bằng Streamlit bởi Nhóm 10.")