import os
import torch
import torchaudio
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from huggingface_hub import PyTorchModelHubMixin

# This is the only part of the script you need to modify.
# Set this to the path where your audio files are stored.
folder_paths = ["./database/dev/con_wav","./database/eval/con_wav","./database/train/con_wav"]
audio_formats = (".mp3", ".wav", ".flac", ".m4a")

# === Set device (use GPU if available) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Wrapper for the SSL model ===
class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Model config used to build SSL architecture
        cfg = Wav2Vec2Config(
            quantize_targets=True,
            extractor_mode="layer_norm",
            layer_norm_first=True,
            final_dim=768,
            latent_temp=(2.0, 0.1, 0.999995),
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            conv_bias=True,
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
            feature_grad_mult=1.0
        )
        # Initialize SSL model with random weights
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data):
        # If input has shape (B, T, 1), squeeze the last dim
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        # Extract features
        with torch.no_grad():
            features = self.model(input_data.to(device), mask=False, features_only=True)['x']
        return features

# === Function for reading and pre-processing waveforms ===
def load_wav_and_preprocess(wav_path, target_sr=16000):
    # Load audio file
    wav, sr = torchaudio.load(wav_path)
    # Convert to mono if stereo
    wav = wav.mean(dim=0)
    # Resample to target sampling rate
    wav = torchaudio.functional.resample(wav, sr, new_freq=target_sr)
    # Normalize waveform
    with torch.no_grad():
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
    # Add batch dimension and return
    return wav.unsqueeze(0).to(device)

# === The actual deepfake detection model using SSL frontend + FC backend ===
class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 1024
        self.num_classes = 2

        # Frontend: SSL model
        self.m_ssl = SSLModel()

        # Backend: Pooling + Classification
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(
            in_features=self.ssl_orig_output_dim,
            out_features=self.num_classes,
        )

    def forward(self, wav):
        emb = self.m_ssl.extract_feat(wav)  # [B, T, D]
        emb = emb.transpose(1, 2)           # [B, D, T]
        pooled_emb = self.adap_pool1d(emb)  # [B, D, 1]
        pooled_emb = pooled_emb.squeeze(-1) # [B, D]
        logits = self.proj_fc(pooled_emb)   # [B, 2]
        return logits

# === Load AntiDeepfake model from Hugging Face===
model = DeepfakeDetector.from_pretrained("nii-yamagishilab/wav2vec-large-anti-deepfake")
model.to(device)
model.eval()


import numpy as np
from sklearn.metrics import roc_curve, f1_score

# ================================
# Inference
# ================================
results = {
    "train": [],
    "valid": [],
    "test": []
}

for folder_path in folder_paths:
    cnt = 0
    if "train" in folder_path:
        split = "train"
    elif "dev" in folder_path:
        split = "valid"
    elif "eval" in folder_path:
        split = "test"
    else:
        continue

    print(f"Processing {split}: {folder_path}")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(audio_formats):
                input_path = os.path.join(root, file)

                with torch.no_grad():
                    wav = load_wav_and_preprocess(input_path)
                    logits = model(wav)
                    probs = torch.nn.functional.softmax(logits, dim=1)

                results[split].append((file, probs.cpu().numpy()[0]))

                # cnt += 1

                # if cnt > 30:
                    # break
        # break

# ================================
# Evaluation function
# ================================
def evaluate_split(name, data):
    y_true, y_pred, y_score = [], [], []
    correct_files = []

    for file_name, prob in data:
        fake_prob = prob[0]
        real_prob = prob[1]

        pred_label = 0 if fake_prob > real_prob else 1

        if "CON" in file_name:
            true_label = 0
        elif "LA" in file_name:
            true_label = 1
        else:
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_score.append(real_prob)

        if pred_label == true_label:
            correct_files.append(file_name)

    if len(y_true) == 0:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    print(f"\n{name} results")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"EER: {eer:.4f}")

    print(f"\n{name} 正确预测文件 ({len(correct_files)}):")
    for f in correct_files:
        print(f)



    return y_true, y_pred, y_score


# ================================
# Evaluate each split
# ================================
all_true, all_pred, all_score = [], [], []

for split in ["train", "valid", "test"]:
    out = evaluate_split(split, results[split])
    if out is not None:
        t, p, s = out
        all_true.extend(t)
        all_pred.extend(p)
        all_score.extend(s)

# ================================
# Overall metrics
# ================================
all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_score = np.array(all_score)

acc = np.mean(all_true == all_pred)
f1 = f1_score(all_true, all_pred)

fpr, tpr, _ = roc_curve(all_true, all_score)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

print("\n==============================")
print("OVERALL RESULTS")
print("==============================")
print(f"Total samples: {len(all_true)}")
print(f"Accuracy: {acc:.4f}")
print(f"F1: {f1:.4f}")
print(f"EER: {eer:.4f}")



exit()
# === Inference on a folder of audio files ===
results = []

# cnt = 0
for folder_path in folder_paths:
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(audio_formats):
                input_path = os.path.join(root, file)
                with torch.no_grad():
                    wav = load_wav_and_preprocess(input_path)
                    logits = model(wav)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    results.append((file, probs.cpu().numpy()[0]))
                # cnt += 1
                # if cnt > 20:
                    # break

        # break
    # break
# Sort results alphabetically by filename
results.sort(key=lambda x: x[0])

# Print formatted results
correct_files = []

for file_name, prob in results:
    fake_prob = prob[0]
    real_prob = prob[1]

    # 预测标签
    pred_label = "fake" if fake_prob > real_prob else "real"

    # 从文件名获取真实标签
    if "CON" in file_name:
        true_label = "fake"
    elif "LA" in file_name:
        true_label = "real"
    else:
        continue  # 跳过无法判断的文件名

    # print(f"{file_name}: real prob = {real_prob:.3f}, fake prob = {fake_prob:.3f}")

    # 判断是否预测正确
    if pred_label == true_label:
        correct_files.append(file_name)

print("\n预测正确的文件：")
for f in correct_files:
    print(f)

