import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from captum.attr import Saliency, InputXGradient, IntegratedGradients
from sklearn.linear_model import LinearRegression
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# === 1. Configuration ===
# Your audio root directory
database_root = "../database/dev"
# Your file list txt path
file_list_path = "./common_valid_correct_files.txt"

# Output directory configuration (Changed to wavlm)
output_root = "../XAI_Image/wavlm/dev"
folders = {
    "IG": os.path.join(output_root, "IG"),
    "lime": os.path.join(output_root, "lime"),
    "saliency": os.path.join(output_root, "saliency")
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure output directories exist
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)
print(f"Output directories created at: {output_root}")


# === 2. Model Definitions (WavLM) ===
# We wrap the HuggingFace model to ensure it works seamlessly with Captum
class WavLMWrapper(torch.nn.Module):
    def __init__(self, model_name="DavidCombei/wavLM-base-Deepfake_V2"):
        super().__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)

    def forward(self, input_values):
        # HuggingFace models often output a sequence classification object.
        # We need the raw logits for Captum.
        outputs = self.model(input_values)
        return outputs.logits


# === 3. Helper Functions ===
def load_wav_and_preprocess(wav_path, target_sr=16000, requires_grad=False):
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, new_freq=target_sr)

    # WavLM typically expects raw waveform input.
    # Normalization helps stability but isn't strictly enforced by the feature extractor
    # if we are bypassing it for gradient tracking.
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

print("Loading WavLM model...")
# Using the specific WavLM model for deepfake detection
model = WavLMWrapper(model_name="DavidCombei/wavLM-base-Deepfake_V2")
model.to(device)
model.eval()

# === Map Files ===
print("Scanning database directory to map files...")
file_path_map = {}
for root, dirs, files in os.walk(database_root):
    for file in files:
        # Skip temporary files starting with ._
        if file.startswith("._"):
            continue
        if file.lower().endswith((".wav", ".mp3", ".flac")):
            # Map filename to full path
            file_path_map[file] = os.path.join(root, file)

print(f"Mapped {len(file_path_map)} valid audio files (ignored '._' temp files).")

# Read txt list
if not os.path.exists(file_list_path):
    print(f"Error: {file_list_path} not found.")
    exit()

with open(file_list_path, 'r') as f:
    # Filter out empty lines and temporary files
    target_files = [line.strip() for line in f if
                    line.strip() and not line.startswith("._") and not line.startswith(".")]

print(f"Found {len(target_files)} files to process in {file_list_path}.")

# === 5. Batch Processing Loop ===

for idx, filename in enumerate(target_files):
    # 1. Locate File
    if filename not in file_path_map:
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

        # 4. Load Audio
        input_wav = load_wav_and_preprocess(full_path, requires_grad=True)
        duration_sec = input_wav.shape[-1] / 16000
        max_freq_hz = 8000

        # === A. Generate IG ===
        save_path_ig = os.path.join(folders["IG"], jpg_name)
        if not os.path.exists(save_path_ig):
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