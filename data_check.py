import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. THIẾT LẬP ĐƯỜNG DẪN
DATA_PATH = 'D:/CIC_IDS_Data' 
SAVE_PATH = 'D:/CIC_IDS_Data/cleaned_data.parquet' # File sẽ lưu

#PHẦN 1: ĐỌC VÀ KẾT HỢP DỮ LIỆU
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
df_list = []

print("BẮT ĐẦU ĐỌC DỮ LIỆU: ")
for filename in all_files:
    try:
        df = pd.read_csv(filename, low_memory=False) 
        df_list.append(df)
        print(f"Đã đọc: {os.path.basename(filename)}")
    except Exception as e:
        print(f"Lỗi khi đọc {os.path.basename(filename)}: {e}")

# Kết hợp tất cả thành một DataFrame duy nhất
df_combined = pd.concat(df_list, axis=0, ignore_index=True)

# Chuẩn hóa tên cột (Xóa khoảng trắng, thay bằng dấu gạch dưới)
df_combined.columns = df_combined.columns.str.strip().str.replace(' ', '_')

print(f"\nTổng số dòng dữ liệu: {len(df_combined)}")

#PHẦN 2 và 3: KIỂM TRA (MISSING VALUES & OUTLIERS)

# Kiểm tra Missing Values
missing_percentage = (df_combined.isnull().sum() / len(df_combined)) * 100
print("\nKIỂM TRA DỮ LIỆU THIẾU: ")
print(missing_percentage[missing_percentage > 0])

# Kiểm tra Outlier bằng Box Plot
print("\nKIỂM TRA OUTLIER (GIÁ TRỊ NGOẠI LAI):")
flow_features_to_check = ['Flow_Duration', 'Total_Length_of_Fwd_Packets', 'Flow_Bytes/s']
plt.figure(figsize=(15, 5)) 
for i, feature in enumerate(flow_features_to_check):
    plt.subplot(1, 3, i + 1)
    upper_bound = df_combined[feature].quantile(0.99)
    filtered_data = df_combined[df_combined[feature] <= upper_bound][feature]
    sns.boxplot(y=filtered_data, color='skyblue')
    plt.title(f'Box Plot: {feature}')
plt.tight_layout()
plt.show()
# sau khi chạy xong hãy đóng cửa sổ của cái box này mới có thể chạy tiếp

#PHẦN 4: LÀM SẠCH DỮ LIỆU (FEATURE ENGINEERING)
print("\nBẮT ĐẦU LÀM SẠCH DỮ LIỆU (STEP 4): ")

# 4.1 Xử lý NaN và Inf
df_combined.replace([np.inf, -np.inf, 'NaN', 'Infinity'], np.nan, inplace=True)
df_combined.fillna(0, inplace=True)
print("Đã xử lý NaN và Inf.")

# 4.2 Loại bỏ các cột Metadata không cần thiết
cols_to_drop = ['Flow_ID', 'Source_IP', 'Source_Port', 'Destination_IP', 'Destination_Port', 'Protocol', 'Timestamp']
existing_cols_to_drop = [c for c in cols_to_drop if c in df_combined.columns]
df_combined.drop(columns=existing_cols_to_drop, inplace=True)
print(f"Đã loại bỏ các cột định danh: {existing_cols_to_drop}")

# 4.3 Gộp nhãn để xử lý mất cân bằng
def consolidate_label(label):
    label = str(label).strip().upper()
    if label == 'BENIGN': return 'Benign'
    if 'DOS' in label or 'HEARTBLEED' in label: return 'DoS'
    if 'DDOS' in label: return 'DDoS'
    if 'WEB ATTACK' in label: return 'Web_Attack'
    if 'PATATOR' in label: return 'Brute_Force'
    if 'INFILTRATION' in label: return 'Infiltration'
    if 'BOT' in label: return 'Bot'
    return 'Other'

df_combined['Label_Category'] = df_combined['Label'].apply(consolidate_label)

print("\n--- THỐNG KÊ NHÃN SAU KHI GỘP ---")
print(df_combined['Label_Category'].value_counts())

#LƯU DỮ LIỆU RA FILE PARQUET
print("\n ĐANG LƯU DỮ LIỆU SẠCH (Vui lòng đợi một chút)...")
df_combined.to_parquet(SAVE_PATH, index=False)

print(f"File dữ liệu sạch đã được làm lại và lưu tại bằng thư viện parquet: {SAVE_PATH}")
