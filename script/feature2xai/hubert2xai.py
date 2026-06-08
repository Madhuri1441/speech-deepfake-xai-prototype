import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import urllib.request
import fairseq
from huggingface_hub import PyTorchModelHubMixin
from captum.attr import Saliency, InputXGradient, IntegratedGradients
from sklearn.linear_model import LinearRegression
import warnings
import gc

# 忽略警告
warnings.filterwarnings("ignore")

# === 1. Configuration ===
# 你的音频根目录
database_root = "../database/dev"
# 你的文件名列表 txt
file_list_path = "./common_valid_correct_files.txt"

# 输出目录配置
output_root = "../XAI_Image/hubert/dev"
folders = {
    "IG": os.path.join(output_root, "IG"),
    "lime": os.path.join(output_root, "lime"),
    "saliency": os.path.join(output_root, "saliency")
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 确保输出目录存在
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)
print(f"Output directories created at: {output_root}")

# === Download HuBERT Checkpoint ===
ssl_path = "hubert_xtralarge_ll60k.pt"
ssl_url = "https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt"

if not os.path.exists(ssl_path):
    print("Downloading HuBERT checkpoint...")
    urllib.request.urlretrieve(ssl_url, ssl_path)
    print("Download complete.")
else:
    print("HuBERT checkpoint exists.")


# === 2. Model Definitions ===
class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
        self.model = model[0]

    def extract_feat(self, input_data):
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        features = self.model(
            input_data.to(device),
            mask=False,
            features_only=True
        )['x']
        return features


class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 1280
        self.num_classes = 2
        self.m_ssl = SSLModel()
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(
            in_features=self.ssl_orig_output_dim,
            out_features=self.num_classes,
        )

    def forward(self, wav):
        emb = self.m_ssl.extract_feat(wav)
        emb = emb.transpose(1, 2)
        pooled_emb = self.adap_pool1d(emb).squeeze(-1)
        logits = self.proj_fc(pooled_emb)
        return logits


# === 3. Helper Functions ===
def load_wav_and_preprocess(wav_path, target_sr=16000, requires_grad=False):
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, new_freq=target_sr)
    wav = wav.squeeze()
    with torch.no_grad():
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
    wav = wav.unsqueeze(0).to(device)

    if requires_grad:
        wav.requires_grad = True
    return wav


def compute_attribution_spectrogram(attr_waveform):
    attr_np = attr_waveform.squeeze().detach().cpu().numpy()
    S = librosa.stft(attr_np, n_fft=1024, hop_length=512)
    S_mag = np.abs(S)
    if S_mag.max() > 0:
        S_mag = S_mag / S_mag.max()
    return S_mag


def run_audio_lime(model, waveform, target_class, num_segments=50, num_samples=200):
    wav_tensor = waveform.clone().detach()
    T = wav_tensor.shape[1]
    if T < num_segments:
        num_segments = T // 2
    seg_len = T // num_segments
    masks = np.random.randint(0, 2, size=(num_samples, num_segments))

    batch_inputs = []
    for i in range(num_samples):
        temp_wav = wav_tensor.clone()
        for j in range(num_segments):
            if masks[i, j] == 0:
                start = j * seg_len
                end = min((j + 1) * seg_len, T)
                temp_wav[:, start:end] = 0
        batch_inputs.append(temp_wav)

    probs = []
    # 【关键修改】Batch Size 设为 1 以节省显存
    batch_size = 1
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_list = batch_inputs[i:i + batch_size]
            if not batch_list: break
            batch = torch.cat(batch_list, dim=0).to(device)
            logits = model(batch)
            p = torch.nn.functional.softmax(logits, dim=1)
            probs.extend(p[:, target_class].cpu().numpy())

    probs = np.array(probs)
    reg = LinearRegression()
    reg.fit(masks, probs)
    return reg.coef_


def plot_and_save_spectrogram(data, title, save_path, duration_sec, max_freq_hz):
    plt.figure(figsize=(10, 4))
    plt.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap="hot",
        extent=[0, duration_sec, 0, max_freq_hz]
    )
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format='%+2.0f')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_and_save_lime(weights, title, save_path, duration_sec):
    plt.figure(figsize=(10, 4))
    x_axis = np.linspace(0, duration_sec, len(weights))
    plt.bar(x_axis, weights, width=(x_axis[1] - x_axis[0]), align='edge', color='orange', alpha=0.9, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Importance")
    plt.xlim(0, duration_sec)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# === 4. Initialization ===

print("Loading HuBERT model...")
model = DeepfakeDetector.from_pretrained("nii-yamagishilab/hubert-xlarge-anti-deepfake")
model.to(device)
model.eval()

# === 文件映射 ===
print("Scanning database directory to map files...")
file_path_map = {}
for root, dirs, files in os.walk(database_root):
    for file in files:
        if file.startswith("._"): continue
        if file.lower().endswith((".wav", ".mp3", ".flac")):
            file_path_map[file] = os.path.join(root, file)

print(f"Mapped {len(file_path_map)} valid audio files.")

if not os.path.exists(file_list_path):
    print(f"Error: {file_list_path} not found.")
    exit()

with open(file_list_path, 'r') as f:
    target_files = [line.strip() for line in f if
                    line.strip() and not line.startswith("._") and not line.startswith(".")]

print(f"Found {len(target_files)} files to process.")

# === 5. Batch Processing Loop ===

for idx, filename in enumerate(target_files):
    # Locate File
    if filename not in file_path_map:
        print(f"[{idx + 1}/{len(target_files)}] ⚠️ Skip: {filename} not found.")
        continue

    full_path = file_path_map[filename]
    base_name = os.path.splitext(filename)[0]
    jpg_name = base_name + ".jpg"

    # 真·断点续传：如果在 load audio 之前发现图都齐了，直接跳过
    path_ig = os.path.join(folders["IG"], jpg_name)
    path_sal = os.path.join(folders["saliency"], jpg_name)
    path_lime = os.path.join(folders["lime"], jpg_name)

    if os.path.exists(path_ig) and os.path.exists(path_sal) and os.path.exists(path_lime):
        continue

    print(f"[{idx + 1}/{len(target_files)}] Processing: {filename}")

    try:
        # Class determination
        if "CON" in filename:
            target_class_idx = 0
            label_str = "Fake"
        elif "LA" in filename:
            target_class_idx = 1
            label_str = "Real"
        else:
            target_class_idx = 0
            label_str = "Unknown"

        # Load Audio (无截断)
        input_wav = load_wav_and_preprocess(full_path, requires_grad=True)
        duration_sec = input_wav.shape[-1] / 16000
        max_freq_hz = 8000

        # === A. Generate IG ===
        if not os.path.exists(path_ig):
            ig = IntegratedGradients(model)
            # 【关键修改】internal_batch_size=1：防止IG显存爆炸
            attr_ig = ig.attribute(input_wav, target=target_class_idx, n_steps=50, internal_batch_size=1)
            spec_ig = compute_attribution_spectrogram(attr_ig)
            plot_and_save_spectrogram(spec_ig, f"IG - {filename} ({label_str})", path_ig, duration_sec, max_freq_hz)

            # 手动清理
            del attr_ig, ig
            torch.cuda.empty_cache()
            gc.collect()

        # === B. Generate Saliency ===
        if not os.path.exists(path_sal):
            saliency = Saliency(model)
            attr_sal = saliency.attribute(input_wav, target=target_class_idx, abs=False)
            spec_sal = compute_attribution_spectrogram(attr_sal)
            plot_and_save_spectrogram(spec_sal, f"Saliency - {filename} ({label_str})", path_sal, duration_sec,
                                      max_freq_hz)

            del attr_sal, saliency
            torch.cuda.empty_cache()

        # === C. Generate LIME ===
        if not os.path.exists(path_lime):
            lime_weights = run_audio_lime(model, input_wav, target_class_idx, num_segments=40, num_samples=100)
            plot_and_save_lime(lime_weights, f"LIME - {filename} ({label_str})", path_lime, duration_sec)

            del lime_weights
            torch.cuda.empty_cache()

        # Final cleanup for loop
        del input_wav
        torch.cuda.empty_cache()
        gc.collect()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ❌ OOM Error (Skipped): {filename} is too heavy.")
            torch.cuda.empty_cache()
            gc.collect()
            if 'input_wav' in locals(): del input_wav
            continue
        else:
            print(f"  ❌ Runtime Error: {e}")
            continue
    except Exception as e:
        print(f"  ❌ Error processing {filename}: {e}")
        continue

print("\nBatch processing complete!")