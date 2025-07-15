# check_channels.py
import mne
import os

# Đường dẫn đến thư mục chứa file EDF
DATASET_DIR = "datasets"

# Lấy file EDF đầu tiên trong thư mục để kiểm tra
# (Giả định rằng tất cả các file trong thư mục đều có cùng cấu trúc kênh)
try:
    first_file = [f for f in os.listdir(DATASET_DIR) if f.endswith('.edf')][0]
    filepath = os.path.join(DATASET_DIR, first_file)

    print(f"Đang kiểm tra các kênh trong file: {filepath}\n")

    # Đọc file EDF mà không cần tải dữ liệu vào bộ nhớ
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose='ERROR')

    # In ra danh sách các tên kênh
    print("Các tên kênh có sẵn trong file là:")
    print(raw.ch_names)

except IndexError:
    print(f"Lỗi: Không tìm thấy file .edf nào trong thư mục '{DATASET_DIR}'.")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")