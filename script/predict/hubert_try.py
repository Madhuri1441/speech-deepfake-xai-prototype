import os
import urllib.request
import torch
import torchaudio
import fairseq
import numpy as np
from sklearn.metrics import roc_curve, f1_score
from huggingface_hub import PyTorchModelHubMixin

# ================================
# DATA PATHS
# ================================
folder_paths = [
    "database/dev/con_wav",
]

audio_formats = (".mp3", ".wav", ".flac", ".m4a")

# ================================
# DEVICE
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================
# Download HuBERT checkpoint
# ================================
ssl_path = "hubert_xtralarge_ll60k.pt"
ssl_url = "https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt"

if not os.path.exists(ssl_path):
    print("Downloading HuBERT checkpoint...")
    urllib.request.urlretrieve(ssl_url, ssl_path)
    print("Download complete.")
else:
    print("HuBERT checkpoint exists.")

# ================================
# SSL MODEL (HuBERT)
# ================================
class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
        self.model = model[0].to(device)

    def extract_feat(self, input_data):
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        with torch.no_grad():
            features = self.model(
                input_data.to(device),
                mask=False,
                features_only=True
            )['x']
        return features

# ================================
# AUDIO LOADER
# ================================
def load_wav_and_preprocess(wav_path, target_sr=16000):
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)
    wav = torchaudio.functional.resample(wav, sr, new_freq=target_sr)

    with torch.no_grad():
        wav = torch.nn.functional.layer_norm(wav, wav.shape)

    return wav.unsqueeze(0).to(device)

# ================================
# Deepfake Detector
# ================================
class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 1280
        self.num_classes = 2

        self.m_ssl = SSLModel()
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(1)
        self.proj_fc = torch.nn.Linear(1280, 2)

    def forward(self, wav):
        emb = self.m_ssl.extract_feat(wav)
        emb = emb.transpose(1, 2)
        pooled_emb = self.adap_pool1d(emb).squeeze(-1)
        logits = self.proj_fc(pooled_emb)
        return logits

# ================================
# LOAD MODEL
# ================================
model = DeepfakeDetector.from_pretrained(
    "nii-yamagishilab/hubert-xlarge-anti-deepfake"
)
model.to(device)
model.eval()

# ================================
# INFERENCE
# ================================
results = {"train": [], "valid": [], "test": []}

for folder_path in folder_paths:
    if "train" in folder_path:
        split = "train"
    elif "dev" in folder_path:
        split = "valid"
    elif "eval" in folder_path:
        split = "test"
    else:
        continue

    print(f"Processing {split}")

    for root, _, files in os.walk(folder_path):
        for file in files:
            # 修改处：跳过以 . 开头的临时文件
            if file.startswith("."):
                continue

            if file.lower().endswith(audio_formats):
                input_path = os.path.join(root, file)

                with torch.no_grad():
                    wav = load_wav_and_preprocess(input_path)
                    logits = model(wav)
                    probs = torch.nn.functional.softmax(logits, dim=1)

                results[split].append((file, probs.cpu().numpy()[0]))

# ================================
# EVALUATION
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

    if name == "valid":
        with open("common_valid_correct_files.txt", "w") as fp:
            for fname in correct_files:
                fp.write(fname + "\n")
        print(f"Saved {len(correct_files)} filenames to common_valid_correct_files.txt")

    return y_true, y_pred, y_score


# ================================
# RUN EVAL
# ================================
all_true, all_pred, all_score = [], [], []

for split in ["train", "valid", "test"]:
    out = evaluate_split(split, results[split])
    if out is not None:
        t, p, s = out
        all_true.extend(t)
        all_pred.extend(p)
        all_score.extend(s)

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_score = np.array(all_score)

acc = np.mean(all_true == all_pred)
f1 = f1_score(all_true, all_pred)

fpr, tpr, _ = roc_curve(all_true, all_score)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

print("\nOVERALL RESULTS")
print(f"Accuracy: {acc:.4f}")
print(f"F1: {f1:.4f}")
print(f"EER: {eer:.4f}")
