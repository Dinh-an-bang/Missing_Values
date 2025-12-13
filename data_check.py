import pandas as pd
import numpy as np
import glob
import os

# thêm đường dẫn đến link
DATA_PATH = 'D:/CIC_IDS_Data' 

# Lấy danh sách tất cả các file CSV 
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

df_list = []
print("--- BẮT ĐẦU KẾT HỢP DỮ LIỆU ---")
print(f"Tìm thấy {len(all_files)} tệp CSV.")

for filename in all_files:
    try:
        # low_memory=False để tránh cảnh báo khi xử lý dữ liệu lớn
        # Header=0 để đảm bảo dòng đầu tiên là tên cột
        df = pd.read_csv(filename, low_memory=False) 
        df_list.append(df)
        print(f"Đã đọc thành công: {os.path.basename(filename)}")
    except Exception as e:
        print(f"Lỗi khi đọc {os.path.basename(filename)}. Vui lòng kiểm tra file: {e}")

# Kết hợp tất cả các DataFrame thành một DataFrame lớn duy nhất
df_combined = pd.concat(df_list, axis=0, ignore_index=True)

print(f"\nBÁO CÁO TÓM TẮT DỮ LIỆU")
print(f"Tổng số bản ghi (Flows) sau khi kết hợp: {len(df_combined)}")

# 1. Chuẩn hóa tên cột và xử lý giá trị không hợp lệ
df_combined.columns = df_combined.columns.str.strip() 
df_combined.columns = df_combined.columns.str.replace(' ', '_') # Thay thế khoảng trắng bằng gạch dưới

# Thay thế giá trị vô cùng (Inf) và giá trị chuỗi không phải số bằng NaN
df_combined.replace([np.inf, -np.inf, 'NaN', 'Infinity'], np.nan, inplace=True) 

#KIỂM TRA MISSING VALUES (DỮ LIỆU THIẾU) 
print("\nKIỂM TRA DỮ LIỆU THIẾU (MISSING VALUES):")

# Tính tỷ lệ phần trăm dữ liệu thiếu của mỗi cột
missing_percentage = (df_combined.isnull().sum() / len(df_combined)) * 100
missing_cols = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

if not missing_cols.empty:
    print("Các cột có Missing Values --- Inf cần xử lý (tỷ lệ %):")
    print(missing_cols.head(5)) # Chỉ hiển thị 5 cột bị thiếu nhiều nhất
    print(f"\nTổng số cột có giá trị thiếu: {len(missing_cols)}")
else:
    print("Tuyệt vời! Không có Missing Values hoặc giá trị Inf nào được tìm thấy.")

#KIỂM TRA MẤT CÂN BẰNG LỚP (CLASS IMBALANCE)
print("\nKIỂM TRA MẤT CÂN BẰNG LỚP (IMBALANCE):")

# Đảm bảo cột nhãn có tên là 'Label' và loại bỏ khoảng trắng thừa
df_combined['Label'] = df_combined['Label'].astype(str).str.strip() 

label_counts = df_combined['Label'].value_counts()
label_percentage = df_combined['Label'].value_counts(normalize=True) * 100

imbalance_report = pd.DataFrame({
    'Count': label_counts,
    'Percentage': label_percentage.round(4)
})
print(imbalance_report)

# Cảnh báo về các lớp hiếm
rare_classes = imbalance_report[imbalance_report['Percentage'] < 0.5]
if not rare_classes.empty:
    print("\nCẢNH BÁO MẤT CÂN BẰNG NGHIÊM TRỌNG (Các lớp dưới 0.5%):")
    print(rare_classes)
