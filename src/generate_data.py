import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

def get_project_root():
    """Tự động tìm đường dẫn gốc của dự án để tránh lỗi FileNotFoundError."""
    # Lấy thư mục chứa file hiện tại (thư mục src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Lùi lại 1 cấp để ra thư mục gốc (mobile-clv-prediction)
    root_dir = os.path.dirname(current_dir)
    return root_dir

def create_mobile_retail_data(num_customers=500, num_transactions=2500):
    """
    Sinh dữ liệu giả lập cho ngành bán lẻ thiết bị di động.
    Tạo ra số lượng giao dịch lớn hơn số khách hàng để đảm bảo có sự lặp lại (dùng cho tính CLV).
    """
    print("⏳ Đang khởi tạo dữ liệu giả lập...")
    
    # 1. Cố định seed để đảm bảo tính tái lập (Yêu cầu 2.7 của đồ án)
    np.random.seed(42)
    random.seed(42)

    # 2. Định nghĩa danh mục sản phẩm phức tạp hơn
    brands = ['Apple', 'Samsung', 'Xiaomi', 'Oppo']
    categories = ['Flagship', 'Mid-range', 'Entry-level', 'Accessories']
    
    # Cấu trúc: {Brand: {Category: [(Tên sản phẩm, Giá tiền)]}}
    catalog = {
        'Apple': {
            'Flagship': [('iPhone 15 Pro Max', 34000000), ('iPhone 14 Pro', 25000000)],
            'Mid-range': [('iPhone 13', 15000000), ('iPhone 11', 10000000)],
            'Accessories': [('AirPods Pro 2', 6000000), ('Ốp lưng MagSafe', 1200000), ('Sạc 20W', 500000)]
        },
        'Samsung': {
            'Flagship': [('Galaxy S24 Ultra', 31000000), ('Galaxy Z Fold 5', 40000000)],
            'Mid-range': [('Galaxy A54', 8500000), ('Galaxy M34', 7000000)],
            'Entry-level': [('Galaxy A05', 3000000)],
            'Accessories': [('Galaxy Buds 2 Pro', 4000000), ('Galaxy Watch 6', 6500000)]
        },
        'Xiaomi': {
            'Flagship': [('Xiaomi 14', 22000000)],
            'Mid-range': [('Redmi Note 13 Pro', 7500000)],
            'Entry-level': [('Redmi 12', 3500000)],
            'Accessories': [('Xiaomi Smart Band 8', 990000), ('Sạc dự phòng 10000mAh', 400000)]
        },
        'Oppo': {
            'Flagship': [('Find N3 Flip', 25000000)],
            'Mid-range': [('Reno 11', 10000000)],
            'Entry-level': [('A18', 3900000)],
            'Accessories': [('Tai nghe Enco Air', 1500000)]
        }
    }

    # 3. Khởi tạo danh sách khách hàng
    customers = [f'CUS{str(i).zfill(4)}' for i in range(1, num_customers + 1)]
    
    # Tạo phân phối mua hàng lệch (Pareto): Một số ít khách hàng mua rất nhiều lần
    customer_probs = np.random.dirichlet(np.ones(num_customers), size=1)[0]

    # 4. Sinh dữ liệu giao dịch
    data = []
    start_date = datetime(2022, 1, 1) # Lấy mốc từ 2 năm trước

    for i in range(1, num_transactions + 1):
        order_id = f'ORD{str(i).zfill(5)}'
        customer_id = np.random.choice(customers, p=customer_probs)
        
        # Chọn brand và category ngẫu nhiên nhưng có trọng số (phụ kiện bán nhiều hơn)
        brand = np.random.choice(brands, p=[0.4, 0.3, 0.15, 0.15]) # Apple và Samsung chiếm đa số
        available_cats = list(catalog[brand].keys())
        category = np.random.choice(available_cats)
        
        # Chọn sản phẩm cụ thể
        product_info = random.choice(catalog[brand][category])
        product_name = product_info[0]
        price = product_info[1]
        
        # Ngày mua hàng ngẫu nhiên trong khoảng 2 năm (730 ngày)
        order_date = start_date + timedelta(days=np.random.randint(0, 730))
        
        # Logic trả góp: Flagship tỷ lệ trả góp cao hơn (60%), phụ kiện không trả góp
        if category == 'Flagship':
            is_installment = np.random.choice([1, 0], p=[0.6, 0.4])
        elif category == 'Accessories':
            is_installment = 0
        else:
            is_installment = np.random.choice([1, 0], p=[0.3, 0.7])

        # Số lượng món hàng trong 1 đơn (Thường là 1, đôi khi là 2)
        quantity = np.random.choice([1, 2], p=[0.9, 0.1])
        total_price = price * quantity

        data.append([
            order_id, customer_id, order_date.strftime('%Y-%m-%d'), 
            brand, category, product_name, quantity, price, total_price, is_installment
        ])

    # 5. Chuyển thành DataFrame và sắp xếp theo thời gian
    columns = ['order_id', 'customer_id', 'order_date', 'brand', 'category', 'product_name', 'quantity', 'unit_price', 'total_price', 'is_installment']
    df = pd.DataFrame(data, columns=columns)
    df = df.sort_values('order_date').reset_index(drop=True)

    # 6. Lưu file an toàn
    root_dir = get_project_root()
    output_dir = os.path.join(root_dir, 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True) # Đảm bảo thư mục tồn tại
    
    output_path = os.path.join(output_dir, 'mobile_sales_raw.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✅ Đã tạo thành công {num_transactions} giao dịch cho {num_customers} khách hàng.")
    print(f"📁 Dữ liệu được lưu tại: {output_path}")
    print("\nXem thử 5 dòng đầu tiên:")
    print(df.head())

if __name__ == "__main__":
    create_mobile_retail_data()