# ./app.py

import os
import json
from collections import OrderedDict
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
import mne
import io  # Đặt ở đầu file nếu chưa có
import tempfile


# Import class model đã được cập nhật cho 3 kênh
from model import MultiChannelSleepTransformer, MultiChannelDeepSleepNet

# --- KHỞI TẠO ỨNG DỤNG FLASK ---
app = Flask(__name__)

# --- CẤU HÌNH ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "datasets" 
ALLOWED_EXTENSIONS = {'.edf'}

def allowed_file(filename):
    """Kiểm tra xem file có phần mở rộng hợp lệ không (.edf)"""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# Cập nhật CHANNEL_MAPPING để chỉ chứa 3 kênh, khớp với lúc training
CHANNEL_MAPPING = {
    'eeg': 'EEG Fpz-Cz',
    'eog': 'EOG horizontal',
    'emg': 'EMG submental'
}

print(f"Sử dụng thiết bị: {device}")

# Các tham số của Transformer cần khớp với lúc train
TRANSFORMER_PARAMS = {"input_size": 3000, "patch_size": 30, "d_model": 128, "num_heads": 8, "num_layers": 6, "num_classes": 5}
# Đảm bảo tên file trọng số khớp với file bạn đã tải về từ Kaggle
TRANSFORMER_PATH = "model_weights/MultiChannelSleepTransformer_best.pt" 
model_transformer = MultiChannelSleepTransformer(**TRANSFORMER_PARAMS)

# Logic tải model trực tiếp vì model đã được lưu đúng cấu trúc
try:
    print(f"Đang tải trọng số từ {TRANSFORMER_PATH}...")
    model_transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device))
    print("Đã tải thành công trọng số cho Model Transformer.")
except Exception as e:
    print(f"[LỖI] Không thể tải model Transformer. Lỗi: {e}. Model sẽ sử dụng trọng số ngẫu nhiên.")
model_transformer.to(device).eval()

# Đảm bảo tên file trọng số khớp với file bạn đã tải về
DEEPSLEEPNET_PATH = "model_weights/MultiChannelDeepSleepNet_best.pt" 
deepsleepnet_available = False
model_deepsleepnet = None
try:
    print(f"Đang tải trọng số từ {DEEPSLEEPNET_PATH}...")
    model_deepsleepnet = MultiChannelDeepSleepNet(num_classes=5)
    model_deepsleepnet.load_state_dict(torch.load(DEEPSLEEPNET_PATH, map_location=device))
    model_deepsleepnet.to(device).eval()
    print("Đã tải thành công trọng số cho Model DeepSleepNet.")
    deepsleepnet_available = True
except Exception as e:
    print(f"[CẢNH BÁO] Lỗi khi tải DeepSleepNet: {e}. Chức năng này sẽ bị vô hiệu hóa.")

# Các dictionary để hiển thị
MODEL_DISPLAY_NAMES = {"transformer": "Sleep Transformer", "deepsleepnet": "DeepSleepNet"}
label_keys = ["W", "N1", "N2", "N3", "REM"]
full_labels = {"W": "Wake (Thức)", "N1": "N1 (Ngủ nông)", "N2": "N2 (Ngủ nông)", "N3": "N3 (Ngủ sâu)", "REM": "REM (Ngủ mơ)"}


def find_dataset_files():
    """Quét thư mục DATASET_DIR và trả về danh sách các file .edf đã được sắp xếp."""
    if not os.path.exists(DATASET_DIR):
        return []
    return sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.edf')])

# ==============================================================================
# ROUTE CHÍNH: HIỂN THỊ GIAO DIỆN
# ==============================================================================
@app.route("/", methods=['GET'])
def index():
    """Hiển thị trang chính với danh sách các file epoch có sẵn."""
    available_files = find_dataset_files()
    return render_template("index.html", available_files=available_files)

# ==============================================================================
# API ROUTE: LẤY DỮ LIỆU TỪ MỘT FILE EPOCH
# ==============================================================================
@app.route("/get-epoch-data", methods=["POST"])
def get_epoch_data():
    """Trích xuất dữ liệu tín hiệu từ một file .edf (vốn là một epoch)."""
    try:
        data = request.get_json()
        filename = data.get("filename")

        if not filename:
            return jsonify({"error": "Tên file không được cung cấp."}), 400

        filepath = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": f"Không tìm thấy file: {filename}"}), 404

        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        for ch_name in CHANNEL_MAPPING.values():
            if ch_name not in raw.ch_names:
                return jsonify({"error": f"Không tìm thấy kênh '{ch_name}' trong file này. Vui lòng kiểm tra file EDF và biến CHANNEL_MAPPING."}), 400
        
        signals = {}
        for signal_type, channel_name in CHANNEL_MAPPING.items():
            signal_data = raw.get_data(picks=[channel_name])[0][:3000].tolist()
            signals[signal_type] = signal_data
        
        return jsonify({
            "success": True, 
            "signals": signals
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================================================================
# API ROUTE: THỰC HIỆN DỰ ĐOÁN VÀ SO SÁNH VỚI LABEL GỐC
# ==============================================================================
@app.route("/predict", methods=["POST"])
def predict():
    """Nhận dữ liệu, dự đoán, và trả về kết quả cùng với label gốc được trích xuất từ tên file."""
    try:
        data = request.get_json()
        filename = data['filename']
        eeg = data['eeg']
        eog = data['eog']
        emg = data['emg']

        if any(len(sig) != 3000 for sig in [eeg, eog, emg]):
            return jsonify({"error": "Dữ liệu đầu vào không hợp lệ. Mỗi kênh phải có 3000 điểm dữ liệu."}), 400

        # Chuẩn bị tensor đầu vào, chuẩn hóa Z-score giống hệt lúc training
        x = np.stack([eeg, eog, emg])
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)
        
        prediction_results = OrderedDict()
        models_to_run = {"transformer": model_transformer}
        if deepsleepnet_available:
            models_to_run["deepsleepnet"] = model_deepsleepnet

        for name, model in models_to_run.items():
            with torch.no_grad():
                output = model(x_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                pred_index = probabilities.argmax().item()
                pred_key = label_keys[pred_index]
                probs_percent = (probabilities * 100).tolist()
                
                prediction_results[name] = {
                    "name": MODEL_DISPLAY_NAMES[name],
                    "prediction": full_labels[pred_key],
                    "prediction_key": pred_key, # Gửi key ('W', 'N1',..) để JS so sánh
                    "probabilities": {label_keys[i]: f"{probs_percent[i]:.2f}%" for i in range(len(label_keys))}
                }
        
        # Trích xuất label thật từ tên file
        ground_truth_key = "UNKNOWN"
        try:
            # Tách chuỗi dựa trên '_label_' và lấy phần sau nó, sau đó bỏ phần đuôi '.edf'
            # Ví dụ: '..._label_W.edf' -> 'W'
            ground_truth_key = filename.split('_label_')[-1].split('.')[0]
        except Exception:
            # Bỏ qua nếu tên file không có định dạng đúng
            pass
            
        ground_truth_full = full_labels.get(ground_truth_key, "Không rõ")
        
        return jsonify({
            "success": True,
            "prediction_results": prediction_results,
            "ground_truth_key": ground_truth_key,
            "ground_truth_full": ground_truth_full
        })
        
    except Exception as e:
        print(f"Lỗi trong hàm predict: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/upload-and-predict", methods=["POST"])
def upload_and_predict():
    """Nhận file EDF tải lên, trích xuất dữ liệu và trả về kết quả dự đoán."""
    if "file" not in request.files:
        return jsonify({"error": "Không có file EDF nào được gửi lên."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên file trống."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File không hợp lệ. Chỉ chấp nhận .edf"}), 400

    temp_file_path = ""
    try:
        # Sử dụng tempfile để tạo file tạm an toàn
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_f:
            file.save(temp_f.name)
            temp_file_path = temp_f.name

        raw = mne.io.read_raw_edf(temp_file_path, preload=True, verbose=False)

        for ch_name in CHANNEL_MAPPING.values():
            if ch_name not in raw.ch_names:
                return jsonify({"error": f"Thiếu kênh '{ch_name}' trong file EDF."}), 400

        signals = {}
        for signal_type, channel_name in CHANNEL_MAPPING.items():
            signal_data = raw.get_data(picks=[channel_name])[0][:3000].tolist()
            signals[signal_type] = signal_data

        x = np.stack([signals['eeg'], signals['eog'], signals['emg']])
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)

        prediction_results = OrderedDict()
        models_to_run = {"transformer": model_transformer}
        if deepsleepnet_available:
            models_to_run["deepsleepnet"] = model_deepsleepnet

        for name, model in models_to_run.items():
            with torch.no_grad():
                output = model(x_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                pred_index = probabilities.argmax().item()
                pred_key = label_keys[pred_index]
                probs_percent = (probabilities * 100).tolist()

                prediction_results[name] = {
                    "name": MODEL_DISPLAY_NAMES[name],
                    "prediction": full_labels[pred_key],
                    "prediction_key": pred_key,
                    "probabilities": {label_keys[i]: f"{probs_percent[i]:.2f}%" for i in range(len(label_keys))}
                }
        
        # ======================================================================
        # === BẮT ĐẦU THAY ĐỔI: TRÍCH XUẤT LABEL TỪ TÊN FILE TẢI LÊN ===
        # ======================================================================
        ground_truth_key = "UNKNOWN"
        ground_truth_full = "Không rõ (file tải lên)"
        try:
            # Lấy tên file gốc từ đối tượng 'file' mà người dùng tải lên
            original_filename = file.filename 
            # Tách chuỗi dựa trên '_label_' và lấy phần sau nó, sau đó bỏ phần đuôi '.edf'
            key_from_filename = original_filename.split('_label_')[-1].split('.')[0]
            
            # Chỉ cập nhật nếu key tìm thấy nằm trong danh sách label hợp lệ
            if key_from_filename in full_labels:
                ground_truth_key = key_from_filename
                ground_truth_full = full_labels[ground_truth_key]

        except Exception:
            # Nếu tên file không có định dạng đúng hoặc có lỗi, giữ nguyên giá trị mặc định
            pass
        # ======================================================================
        # === KẾT THÚC THAY ĐỔI ===
        # ======================================================================

        return jsonify({
            "success": True,
            "prediction_results": prediction_results,
            "signals": signals,
            "ground_truth_key": ground_truth_key,      # Sử dụng key đã trích xuất
            "ground_truth_full": ground_truth_full    # Sử dụng label đầy đủ đã tìm thấy
        })

    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý EDF: {str(e)}"}), 500
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
# --- KHỐI CHẠY CHÍNH ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)