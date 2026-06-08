import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from huggingface_hub import PyTorchModelHubMixin
from captum.attr import Saliency, InputXGradient, IntegratedGradients
from sklearn.linear_model import LinearRegression
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# === 1. Configuration ===
# 你的音频根目录
database_root = "./database/eval"
# 你的文件名列表 txt
file_list_path = "./Explanability-for-ALLM-for-deepfake/common_test_correct_files.txt"

# 输出目录配置
output_root = "XAI_Image/wav2vec/eval"
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


# === 2. Model Definitions ===
class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data):
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        features = self.model(input_data.to(device), mask=False, features_only=True)['x']
        return features


class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 1024
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
    batch_size = 16
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

print("Loading model...")
model = DeepfakeDetector.from_pretrained("nii-yamagishilab/wav2vec-large-anti-deepfake")
model.to(device)
model.eval()

# === 关键修改：文件映射构建部分 ===
print("Scanning database directory to map files...")
file_path_map = {}
for root, dirs, files in os.walk(database_root):
    for file in files:
        # 【修改】强制跳过以 ._ 开头的文件，以及非音频文件
        if file.startswith("._"):
            continue
        if file.lower().endswith((".wav", ".mp3", ".flac")):
            # 存储文件名到全路径的映射
            file_path_map[file] = os.path.join(root, file)

print(f"Mapped {len(file_path_map)} valid audio files (ignored '._' temp files).")

# 读取 txt 列表
if not os.path.exists(file_list_path):
    print(f"Error: {file_list_path} not found.")
    exit()

with open(file_list_path, 'r') as f:
    # 这里也加个过滤，防止 txt 里本身就写了临时文件名
    target_files = [line.strip() for line in f if
                    line.strip() and not line.startswith("._") and not line.startswith(".")]

print(f"Found {len(target_files)} files to process in {file_list_path}.")

# === 5. Batch Processing Loop ===

for idx, filename in enumerate(target_files):
    # 1. Locate File
    if filename not in file_path_map:
        # 如果 map 里没找到，说明它可能是一个临时文件或者名字不对，直接跳过
        print(f"[{idx + 1}/{len(target_files)}] ⚠️ Skip: {filename} not found in valid file map.")
        continue

    full_path = file_path_map[filename]

    # 2. Determine Output Filename
    base_name = os.path.splitext(filename)[0]
    jpg_name = base_name + ".jpg"

    print(f"[{idx + 1}/{len(target_files)}] Processing: {filename}")

    try:
        # 3. Determine Target Class
        if "CON" in filename:
            target_class_idx = 0
            label_str = "Fake"
        elif "LA" in filename:
            target_class_idx = 1
            label_str = "Real"
        else:
            print(f"  ⚠️ Warning: Could not determine class from filename (no CON/LA). Defaulting to Fake (0).")
            target_class_idx = 0
            label_str = "Unknown(0)"

        # 4. Load Audio (如果这里报错，说明是坏文件，try-except会捕获)
        input_wav = load_wav_and_preprocess(full_path, requires_grad=True)
        duration_sec = input_wav.shape[-1] / 16000
        max_freq_hz = 8000

        # === A. Generate IG ===
        save_path_ig = os.path.join(folders["IG"], jpg_name)
        if not os.path.exists(save_path_ig):  # 选做：如果存在就跳过
            ig = IntegratedGradients(model)
            attr_ig = ig.attribute(input_wav, target=target_class_idx, n_steps=50)
            spec_ig = compute_attribution_spectrogram(attr_ig)
            plot_and_save_spectrogram(spec_ig, f"IG - {filename} ({label_str})", save_path_ig, duration_sec,
                                      max_freq_hz)

        # === B. Generate Saliency ===
        save_path_sal = os.path.join(folders["saliency"], jpg_name)
        if not os.path.exists(save_path_sal):
            saliency = Saliency(model)
            attr_sal = saliency.attribute(input_wav, target=target_class_idx, abs=False)
            spec_sal = compute_attribution_spectrogram(attr_sal)
            plot_and_save_spectrogram(spec_sal, f"Saliency - {filename} ({label_str})", save_path_sal, duration_sec,
                                      max_freq_hz)

        # === C. Generate LIME ===
        save_path_lime = os.path.join(folders["lime"], jpg_name)
        if not os.path.exists(save_path_lime):
            lime_weights = run_audio_lime(model, input_wav, target_class_idx, num_segments=40, num_samples=100)
            plot_and_save_lime(lime_weights, f"LIME - {filename} ({label_str})", save_path_lime, duration_sec)

        # Clear GPU memory
        del input_wav
        if 'attr_ig' in locals(): del attr_ig
        if 'attr_sal' in locals(): del attr_sal
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ❌ Error processing {filename}: {e}")
        continue

print("\nBatch processing complete!")