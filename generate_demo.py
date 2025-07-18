import os
import numpy as np
import mne

# --- CÁC THAM SỐ CẤU HÌNH ---

OUTPUT_DIR = "generated_full_demo_dataset" # Đặt tên thư mục mới để tránh nhầm lẫn
CHANNEL_NAMES = ['EEG Fpz-Cz', 'EOG horizontal', 'EMG submental']
SFREQ = 100
DURATION = 30
N_POINTS = SFREQ * DURATION

STAGES_TO_GENERATE = ["W", "N1", "N2", "N3", "REM"]

def generate_staged_signals(stage, n_points, sfreq):
    time = np.linspace(0, DURATION, n_points, endpoint=False)
    eeg = 10 * np.random.randn(n_points)
    eog = 5 * np.random.randn(n_points)
    emg = 5 * np.random.randn(n_points)
    if stage == "W":
        eeg = (20 * np.sin(2 * np.pi * 12 * time) + 15 * np.sin(2 * np.pi * 20 * time)) + 5 * np.random.randn(n_points)
        for _ in range(5):
            idx = np.random.randint(0, n_points - 50)
            eog[idx:idx+50] += np.random.choice([-1, 1]) * 100
        emg = 25 * np.random.randn(n_points)
    elif stage == "N1":
        eeg = 30 * np.sin(2 * np.pi * 6 * time) + 5 * np.random.randn(n_points)
        eog = 20 * np.sin(2 * np.pi * 0.5 * time) + 3 * np.random.randn(n_points)
        emg = 15 * np.random.randn(n_points)
    elif stage == "N2":
        eeg = 20 * np.sin(2 * np.pi * 5 * time) + 5 * np.random.randn(n_points)
        k_idx = np.random.randint(1 * sfreq, n_points - 2 * sfreq)
        eeg[k_idx:k_idx+25] -= 120
        eeg[k_idx+25:k_idx+100] += 80
        for _ in range(2):
            s_idx = np.random.randint(0, n_points - int(1.5 * sfreq))
            spindle_time = np.linspace(0, 1.5, int(1.5 * sfreq), endpoint=False)
            spindle = 35 * np.sin(2 * np.pi * 13 * spindle_time)
            eeg[s_idx:s_idx+len(spindle)] += spindle
        eog = 3 * np.random.randn(n_points)
        emg = 10 * np.random.randn(n_points)
    elif stage == "N3":
        eeg = 150 * np.sin(2 * np.pi * 1.5 * time) + 10 * np.random.randn(n_points)
        eog = 2 * np.random.randn(n_points)
        emg = 5 * np.random.randn(n_points)
    elif stage == "REM":
        eeg = (25 * np.sin(2 * np.pi * 7 * time) + 10 * np.sin(2 * np.pi * 25 * time)) + 5 * np.random.randn(n_points)
        for _ in range(3):
            rem_idx = np.random.randint(0, n_points - 2 * sfreq)
            for i in range(4):
                eog[rem_idx+i*25 : rem_idx+i*25+20] += np.random.choice([-1, 1]) * 80
        emg = 2 * np.random.randn(n_points)
    return eeg, eog, emg

if __name__ == "__main__":
    print(f"Bắt đầu tạo bộ dataset demo tại thư mục: '{OUTPUT_DIR}'")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục: {OUTPUT_DIR}")

    file_counter = 1
    for stage in STAGES_TO_GENERATE:
        print(f"... Đang tạo file cho giai đoạn: {stage}")
        eeg_data, eog_data, emg_data = generate_staged_signals(stage, N_POINTS, SFREQ)
        all_data_uv = np.stack([eeg_data, eog_data, emg_data])
        all_data_volts = all_data_uv * 1e-6
        info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=SFREQ, ch_types='eeg')
        raw = mne.io.RawArray(all_data_volts, info, verbose=False)
        filename = f"demo_epoch_{file_counter:03d}_label_{stage}.edf"
        output_path = os.path.join(OUTPUT_DIR, filename)
        raw.export(output_path, fmt='edf', physical_range='auto', overwrite=True, verbose=False)
        
        file_counter += 1

    print("-" * 30)
    print(f"✅ ĐÃ HOÀN THÀNH!")
    print(f"Đã tạo {len(STAGES_TO_GENERATE)} file EDF với tên file chính xác trong thư mục '{OUTPUT_DIR}'.")