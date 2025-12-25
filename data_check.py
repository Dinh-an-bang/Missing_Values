import pandas as pd
import numpy as np
import glob
import os

# 1. THIẾT LẬP ĐƯỜNG DẪN
DATA_PATH = 'D:/CIC_IDS_Data' 
SAVE_PATH = 'D:/CIC_IDS_Data/step4_cleaned_balanced.parquet'

# --- PHẦN 1: ĐỌC VÀ KẾT HỢP DỮ LIỆU ---
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
df_list = []

print("BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ TỔNG HỢP...")
for filename in all_files:
    try:
        df = pd.read_csv(filename, low_memory=False) 
        df_list.append(df)
        print(f"Đã đọc: {os.path.basename(filename)}")
    except Exception as e:
        print(f"Lỗi khi đọc {os.path.basename(filename)}: {e}")

df_combined = pd.concat(df_list, axis=0, ignore_index=True)
# Chuẩn hóa tên cột để dễ làm việc
df_combined.columns = df_combined.columns.str.strip().str.replace(' ', '_')
print(f"Tổng số dòng gốc: {len(df_combined)}")

# --- PHẦN 2: DỌN DẸP LỖI KỸ THUẬT (NaN & Inf) ---
# Thay thế tất cả các dạng lỗi bằng NaN rồi đưa về 0
df_combined.replace([np.inf, -np.inf, 'NaN', 'Infinity', 'infinity'], np.nan, inplace=True)
df_combined.fillna(0, inplace=True)
print("Đã xử lý triệt để các lỗi NaN và Infinity.")

# --- PHẦN 3: LOẠI BỎ METADATA (Tránh học vẹt) ---
# Xóa các cột định danh để mô hình tập trung vào hành vi flow
cols_to_drop = ['Flow_ID', 'Source_IP', 'Source_Port', 'Destination_IP', 'Destination_Port', 'Protocol', 'Timestamp']
existing_cols = [c for c in cols_to_drop if c in df_combined.columns]
df_combined.drop(columns=existing_cols, inplace=True)
print(f"Đã loại bỏ các cột định danh: {existing_cols}")

# --- PHẦN 4: GỘP NHÃN (Label Consolidation) ---
def consolidate_label(label):
    label = str(label).strip().upper()
    if label == 'BENIGN': return 'Benign'
    if 'DOS' in label or 'HEARTBLEED' in label: return 'DoS'
    if 'DDOS' in label: return 'DDoS'
    if 'INFILTRATION' in label: return 'Infiltration'
    if 'PATATOR' in label: return 'Brute_Force'
    if 'BOT' in label or 'WEB ATTACK' in label: return 'Other_Attack'
    return 'Other'

df_combined['Label_Category'] = df_combined['Label'].apply(consolidate_label)

# --- PHẦN 5: CÂN BẰNG DỮ LIỆU THÔNG MINH (Đúng ý thầy) ---
print("Đang thực hiện cân bằng dữ liệu chú ý tới thời gian...")

# Tách nhóm
df_benign = df_combined[df_combined['Label_Category'] == 'Benign']
df_attack = df_combined[df_combined['Label_Category'] != 'Benign']

# 1. Systematic Sampling cho Benign (Giảm xuống còn 500,000 dòng để cân bằng)
# Việc dùng iloc[::step] giúp giữ lại các mẫu trải dài theo thời gian
step_size = len(df_benign) // 500000
df_benign_balanced = df_benign.iloc[::step_size, :].copy()

# 2. Oversampling cho Infiltration (Lớp quá ít mẫu - 36 dòng)
df_infiltration = df_attack[df_attack['Label_Category'] == 'Infiltration']
df_inf_boosted = pd.concat([df_infiltration] * 100, ignore_index=True)

# 3. Giữ nguyên các loại tấn công khác
df_attack_others = df_attack[df_attack['Label_Category'] != 'Infiltration']

# --- PHẦN 6: HỢP NHẤT VÀ BẢO TOÀN THỨ TỰ ---
# Sử dụng sort_index() để đảm bảo các flow mạng quay về đúng trình tự xảy ra
df_final = pd.concat([df_benign_balanced, df_attack_others, df_inf_boosted]).sort_index()

print("\n--- THỐNG KÊ CUỐI CÙNG SAU KHI LÀM SẠCH VÀ CÂN BẰNG ---")
print(df_final['Label_Category'].value_counts())

# --- PHẦN 7: LƯU TRỮ HIỆU NĂNG CAO ---
print("\n Đang lưu dữ liệu vào định dạng Parquet...")
df_final.to_parquet(SAVE_PATH, index=False)
print(f" HOÀN THÀNH! Dữ liệu sạch đã sẵn sàng tại: {SAVE_PATH}")
