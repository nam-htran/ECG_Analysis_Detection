# ./model.py

import torch
import torch.nn as nn

#======================================================================
# Các class định nghĩa kiến trúc cho mô hình Transformer (3 kênh)
#======================================================================

class PositionalEncoding(nn.Module):
    """Thêm thông tin vị trí vào embedding đầu vào."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerBranch(nn.Module):
    """Một nhánh Transformer để xử lý một kênh tín hiệu."""
    def __init__(self, input_size=3000, patch_size=30, d_model=128, num_heads=8, num_layers=4):
        super(TransformerBranch, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        if input_size % patch_size != 0:
            raise ValueError(f"input_size ({input_size}) phải chia hết cho patch_size ({patch_size}).")
        self.n_patches = input_size // patch_size
        self.embedding = nn.Linear(patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=self.n_patches)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=256, dropout=0.1, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        x = x.squeeze(1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return x

class MultiChannelSleepTransformer(nn.Module):
    """Mô hình chính, kết hợp 3 nhánh Transformer."""
    def __init__(self, input_size=3000, patch_size=30, d_model=128, num_heads=8, num_layers=6, num_classes=5):
        super(MultiChannelSleepTransformer, self).__init__()
        self.eeg_branch = TransformerBranch(input_size, patch_size, d_model, num_heads, num_layers)
        self.eog_branch = TransformerBranch(input_size, patch_size, d_model, num_heads, num_layers)
        self.emg_branch = TransformerBranch(input_size, patch_size, d_model, num_heads, num_layers)
        self.fusion_dim = d_model * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        eeg_embedding = self.eeg_branch(x[:, 0:1, :])
        eog_embedding = self.eog_branch(x[:, 1:2, :])
        emg_embedding = self.emg_branch(x[:, 2:3, :])
        fused_embedding = torch.cat([eeg_embedding, eog_embedding, emg_embedding], dim=1)
        output = self.classifier(fused_embedding)
        return output

#======================================================================
# Các class định nghĩa kiến trúc cho mô hình DeepSleepNet (3 kênh)
#======================================================================
class DeepSleepNetBranch(nn.Module):
    def __init__(self):
        super(DeepSleepNetBranch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=22),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=200),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        min_len = min(x1.shape[2], x2.shape[2])
        x1 = x1[:, :, :min_len]
        x2 = x2[:, :, :min_len]
        x = x1 + x2
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        embedding = lstm_out[:, -1, :]
        return embedding

class MultiChannelDeepSleepNet(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiChannelDeepSleepNet, self).__init__()
        self.eeg_branch = DeepSleepNetBranch()
        self.eog_branch = DeepSleepNetBranch()
        self.emg_branch = DeepSleepNetBranch()
        self.fusion_dim = 1024 * 3
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        eeg_embedding = self.eeg_branch(x[:, 0:1, :])
        eog_embedding = self.eog_branch(x[:, 1:2, :])
        emg_embedding = self.emg_branch(x[:, 2:3, :])
        fused_embedding = torch.cat([eeg_embedding, eog_embedding, emg_embedding], dim=1)
        output = self.classifier(fused_embedding)
        return output