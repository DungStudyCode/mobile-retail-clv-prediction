# app/main.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Hệ Thống CRM - Phân Loại VIP",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. HEADER & GIỚI THIỆU
# ==========================================
st.title("📱 Hệ Thống Phân Lớp Khách Hàng Tiềm Năng (VIP)")
st.markdown("""
**Đồ án môn học: Học máy (Machine Learning) - Nhóm 10** Hệ thống AI phân tích Đa chiều (Demographics, RFM, Hành vi) để dự báo xác suất một khách hàng đạt chuẩn VIP.
""")
st.divider()

# ==========================================
# 3. SIDEBAR: NHẬP LIỆU KHÁCH HÀNG
# ==========================================
st.sidebar.header("🔍 Cấu hình Khách hàng")

def user_input_features():
    st.sidebar.markdown("**👤 1. Nhân khẩu học**")
    age = st.sidebar.slider("Tuổi", 16, 60, 25)
    location = st.sidebar.radio("Khu vực sống", ["Tier 1", "Tier 2"])
    membership_months = st.sidebar.number_input("Số tháng làm thành viên", 1, 60, 12)
    
    st.sidebar.markdown("**🛒 2. Lịch sử Giao dịch (RFM)**")
    recency = st.sidebar.number_input("Recency (Số ngày từ lần cuối mua)", 0, 365, 30)
    frequency = st.sidebar.number_input("Frequency (Tổng số đơn hàng)", 1, 30, 5)
    monetary = st.sidebar.number_input("Monetary (Tổng tiền - VNĐ)", 0, 200000000, 15000000, step=1000000, format="%d")
    
    st.sidebar.markdown("**⚙️ 3. Hành vi & Tương tác**")
    favorite_brand = st.sidebar.selectbox("Thương hiệu yêu thích", ("Apple", "Samsung", "Xiaomi", "Oppo"))
    
    # Sử dụng thang 0-100 và format % cho thanh trượt
    flagship_ratio_pct = st.sidebar.slider("Tỷ lệ mua dòng Flagship", 0, 100, 50, format="%d%%")
    accessories_ratio_pct = st.sidebar.slider("Tỷ lệ mua Phụ kiện", 0, 100, 20, format="%d%%")
    installment_rate_pct = st.sidebar.slider("Tỷ lệ mua trả góp", 0, 100, 30, format="%d%%")
    credit_card_usage_pct = st.sidebar.slider("Tỷ lệ dùng Thẻ tín dụng", 0, 100, 50, format="%d%%")
    
    st.sidebar.markdown("**🎧 4. Dịch vụ Hậu mãi**")
    warranty_claims = st.sidebar.slider("Số lần đi bảo hành", 0, 5, 0)
    app_logins = st.sidebar.number_input("Số lần mở App (30 ngày qua)", 0, 100, 10)

    # Đóng gói thành Dictionary (Chia 100 cho các tỷ lệ để đưa về khoảng 0-1 cho Model)
    data = {
        'Customer_Age': age,
        'Location_Type': location,
        'Membership_Months': membership_months,
        'Frequency': frequency,
        'Recency': recency,
        'Favorite_Brand': favorite_brand,
        'Monetary': monetary,
        'Installment_Rate': installment_rate_pct / 100.0,
        'Flagship_Ratio': flagship_ratio_pct / 100.0,
        'Accessories_Ratio': accessories_ratio_pct / 100.0,
        'Credit_Card_Usage': credit_card_usage_pct / 100.0,
        'Warranty_Claims': warranty_claims,
        'App_Logins_L30D': app_logins
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ==========================================
# 4. HIỂN THỊ DATA ĐẦU VÀO
# ==========================================
st.subheader("Bảng thông tin khách hàng hiện tại:")

# Format tiền tệ có dấu phẩy và tỷ lệ hiển thị dạng %
styled_df = input_df.style.format({
    'Monetary': '{:,.0f}',
    'Installment_Rate': '{:.0%}',
    'Flagship_Ratio': '{:.0%}',
    'Accessories_Ratio': '{:.0%}',
    'Credit_Card_Usage': '{:.0%}'
})

st.dataframe(styled_df, hide_index=True)

# ==========================================
# 5. XỬ LÝ LÕI AI & DỰ BÁO
# ==========================================
current_dir = os.getcwd()
models_dir = os.path.join(current_dir, 'models')

model_path = os.path.join(models_dir, 'best_xgb_model.pkl')
scaler_path = os.path.join(models_dir, 'scaler.pkl')

if st.button("🚀 KÍCH HOẠT AI DỰ BÁO", type="primary"):
    try:
        # Load file mô hình và scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # 1. Định nghĩa chuẩn xác các cột lúc huấn luyện (Phải khớp 100% với file notebook)
        expected_columns = [
            'Customer_Age', 'Membership_Months', 'Frequency', 'Recency', 'Monetary',
            'Installment_Rate', 'Flagship_Ratio', 'Accessories_Ratio', 
            'Credit_Card_Usage', 'Warranty_Claims', 'App_Logins_L30D',
            'Favorite_Brand_Oppo', 'Favorite_Brand_Samsung', 'Favorite_Brand_Xiaomi',
            'Location_Type_Tier 2'
        ]
        
        # 2. One-Hot Encoding dữ liệu đầu vào từ người dùng
        features = pd.get_dummies(input_df, columns=['Favorite_Brand', 'Location_Type'])
        
        # 3. Đồng bộ hóa Schema dữ liệu (Ngăn chặn triệt để lỗi NameError thiếu/thừa cột)
        features = features.reindex(columns=expected_columns, fill_value=0)
        
        # 4. Standard Scaling
        features_scaled = scaler.transform(features)
        
        # 5. Thực hiện dự báo
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        vip_probability = prediction_proba[0][1] * 100
        
        # ==========================================
        # 6. TRỰC QUAN HÓA KẾT QUẢ (DASHBOARD)
        # ==========================================
        st.divider()
        st.subheader("🎯 Báo Cáo Phân Tích Chuyên Sâu")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # BIỂU ĐỒ 1: GAUGE CHART (Đồng hồ đo xác suất)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = vip_probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Tỉ lệ đạt chuẩn VIP", 'font': {'size': 20}},
                number = {'suffix': "%", 'font': {'size': 36, 'color': "white"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "white"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "#FF4B4B"},   # Đỏ
                        {'range': [30, 70], 'color': "#FFA500"},  # Cam
                        {'range': [70, 100], 'color': "#00CC96"}  # Xanh
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': vip_probability
                    }
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_chart2:
            # BIỂU ĐỒ 2: RADAR CHART (Mạng nhện so sánh đặc trưng)
            categories = ['Tần suất (F)', 'Sức mua (M)', 'Tương tác App', 'Mua Flagship', 'Điểm tín dụng']
            
            # Chuẩn hóa input của khách về thang 10 để vẽ
            customer_scores = [
                min(10, input_df['Frequency'][0] * 1.5),
                min(10, input_df['Monetary'][0] / 5000000),
                min(10, input_df['App_Logins_L30D'][0] / 3),
                input_df['Flagship_Ratio'][0] * 10,
                input_df['Credit_Card_Usage'][0] * 10
            ]
            
            # Profile giả lập của 1 VIP chuẩn
            vip_benchmark = [8.5, 9, 8, 8, 7.5]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                  r=customer_scores,
                  theta=categories,
                  fill='toself',
                  name='Khách hàng hiện tại',
                  line_color='#00CC96' if prediction[0] == 1 else '#FFA500'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                  r=vip_benchmark,
                  theta=categories,
                  fill='toself',
                  name='Mức chuẩn VIP',
                  line_color='#FF4B4B'
            ))
            fig_radar.update_layout(
              polar=dict(
                  radialaxis=dict(visible=True, range=[0, 10], gridcolor="gray"),
                  bgcolor="rgba(0,0,0,0)"
              ),
              showlegend=True,
              height=350,
              margin=dict(l=40, r=40, t=50, b=20),
              paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(0,0,0,0)",
              font=dict(color="white")
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ==========================================
        # 7. KẾT LUẬN & HÀNH ĐỘNG ĐỀ XUẤT
        # ==========================================
        st.markdown("### 💡 Hành Động Khuyến Nghị Dành Cho CRM")
        if prediction[0] == 1:
            st.success("🌟 **PHÂN LOẠI: KHÁCH HÀNG VIP**")
            st.info("""
            **Chiến lược giữ chân (Retention):** - Gán thẻ Khách hàng Ưu tiên trên hệ thống.
            - Gửi tặng voucher giảm 2,000,000đ khi nâng cấp các dòng máy Flagship đời mới.
            - Ưu tiên xử lý nhanh gọn các yêu cầu bảo hành (nếu có) để duy trì trải nghiệm hoàn hảo.
            """)
        else:
            st.warning("👤 **PHÂN LOẠI: KHÁCH HÀNG TIÊU CHUẨN**")
            st.info("""
            **Chiến lược bán chéo (Cross-sell/Up-sell):**
            - Khách hàng này có tiềm năng nhưng chưa đủ tương tác. 
            - Gửi Push Notification qua App các chương trình khuyến mãi Phụ kiện (Ốp lưng, Tai nghe).
            - Khuyến khích tham gia mua trả góp qua thẻ tín dụng để tăng kích cỡ giỏ hàng.
            """)
            
    except FileNotFoundError:
        st.error("🚨 LỖI: Không tìm thấy file mô hình (`best_xgb_model.pkl` hoặc `scaler.pkl`) trong thư mục `models/`.")
        st.warning("👉 Hãy chắc chắn rằng bạn đã chạy file `02_model_training.ipynb` với dữ liệu mới để hệ thống lưu model.")
    except Exception as e:
         st.error(f"🚨 LỖI HỆ THỐNG: {e}")
         st.warning("👉 Kiểm tra lại mảng `expected_columns` ở dòng 94 xem đã khớp 100% với các cột lúc train model chưa.")

st.markdown("---")
st.caption("Giao diện & Mô hình AI được thiết kế bởi Nhóm 10 - Môn Học Máy.")