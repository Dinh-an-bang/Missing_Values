import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'D:/CIC_IDS_Data' 

# Lấy danh sách tất cả các file CSV 
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

df_list = []
print("BẮT ĐẦU KẾT HỢP DỮ LIỆU:")
print(f"Tìm thấy {len(all_files)} tệp CSV.")

for filename in all_files:
    try:
        df = pd.read_csv(filename, low_memory=False) 
        df_list.append(df)
        print(f"Đã đọc thành công: {os.path.basename(filename)}")
    except Exception as e:
        print(f"Lỗi khi đọc {os.path.basename(filename)}. Vui lòng kiểm tra file: {e}")

# Kết hợp tất cả các DataFrame thành một DataFrame lớn duy nhất
df_combined = pd.concat(df_list, axis=0, ignore_index=True)

print(f"\n--- BÁO CÁO TÓM TẮT DỮ LIỆU ---")
print(f"Tổng số bản ghi (Flows) sau khi kết hợp: {len(df_combined)}")

# 1. Chuẩn hóa tên cột và xử lý giá trị không hợp lệ
df_combined.columns = df_combined.columns.str.strip() 
df_combined.columns = df_combined.columns.str.replace(' ', '_') 

# Thay thế giá trị vô cùng (Inf) và giá trị chuỗi không phải số bằng NaN
df_combined.replace([np.inf, -np.inf, 'NaN', 'Infinity'], np.nan, inplace=True) 

#KIỂM TRA MISSING VALUES (DỮ LIỆU THIẾU)
print("\nKIỂM TRA DỮ LIỆU THIẾU (MISSING VALUES):")

missing_percentage = (df_combined.isnull().sum() / len(df_combined)) * 100
missing_cols = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

if not missing_cols.empty:
    print("Các cột có Missing Values/Inf cần xử lý (tỷ lệ %):")
    print(missing_cols.head(5)) 
    print(f"\nTổng số cột có giá trị thiếu: {len(missing_cols)}")
else:
    print("Tuyệt vời! Không có Missing Values hoặc giá trị Inf nào được tìm thấy.")

#KIỂM TRA MẤT CÂN BẰNG LỚP (CLASS IMBALANCE)
print("\nKIỂM TRA MẤT CÂN BẰNG LỚP (IMBALANCE): ")

df_combined['Label'] = df_combined['Label'].astype(str).str.strip() 

label_counts = df_combined['Label'].value_counts()
label_percentage = df_combined['Label'].value_counts(normalize=True) * 100

imbalance_report = pd.DataFrame({
    'Count': label_counts,
    'Percentage': label_percentage.round(4)
})
print(imbalance_report)

rare_classes = imbalance_report[imbalance_report['Percentage'] < 0.5]
if not rare_classes.empty:
    print("\nCẢNH BÁO MẤT CÂN BẰNG NGHIÊM TRỌNG (Các lớp dưới 0.5%):")
    print(rare_classes)


#PHẦN 2: KIỂM TRA OUTLIER

# 1. Lấy danh sách các đặc trưng liên tục để kiểm tra Outlier
flow_features_to_check = ['Flow_Duration', 'Total_Length_of_Fwd_Packets', 'Flow_Bytes/s']

print("\nKIỂM TRA OUTLIER (GIÁ TRỊ NGOẠI LAI):")
print("Box Plots sẽ được hiển thị trong cửa sổ riêng. (Chỉ hiển thị dữ liệu đến ngưỡng 99% để biểu đồ rõ ràng hơn)")

# Thiết lập kích thước figure tổng thể
plt.figure(figsize=(15, 5)) 

for i, feature in enumerate(flow_features_to_check):
    # Tạo Box Plot cho từng đặc trưng
    plt.subplot(1, 3, i + 1) # Tạo 1 hàng, 3 cột biểu đồ
    
    # Tính ngưỡng 99% để loại bỏ 1% giá trị cực lớn (Outlier mạnh)
    upper_bound = df_combined[feature].quantile(0.99)
    
    # Lọc dữ liệu: chỉ lấy các giá trị nhỏ hơn hoặc bằng ngưỡng 99%
    filtered_data = df_combined[df_combined[feature] <= upper_bound][feature]
    
    # Vẽ Box Plot
    sns.boxplot(y=filtered_data, color='skyblue')
    plt.title(f'Box Plot: {feature} (Filtered @ 99%)', fontsize=10)
    plt.ylabel(feature)

plt.tight_layout() 
plt.show() # Hiển thị các biểu đồ đã vẽ

print("Đã tạo Box Plot cho các đặc trưng quan trọng.")
