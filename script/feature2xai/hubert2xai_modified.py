import os
import json
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import urllib.request
import fairseq
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel
from huggingface_hub import PyTorchModelHubMixin
from captum.attr import Saliency, InputXGradient, IntegratedGradients
from sklearn.linear_model import LinearRegression
import warnings
import gc

# 忽略警告
warnings.filterwarnings("ignore")

# === 1. Configuration ===
# 你的音频根目录
database_root = "database/dev"
# 你的文件名列表 txt
file_list_path = "./common_valid_correct_files.txt"

# 输出目录配置
output_root = "XAI_Image/hubert/dev"
folders = {
    "IG": os.path.join(output_root, "IG"),
    "lime": os.path.join(output_root, "lime"),
    "saliency": os.path.join(output_root, "saliency"),
    "consensus": os.path.join(output_root, "consensus"),
    "agreement": os.path.join(output_root, "agreement"),
    "refined_ig": os.path.join(output_root, "refined_ig"),
    "refined_saliency": os.path.join(output_root, "refined_saliency"),
    "refined_lime": os.path.join(output_root, "refined_lime"),
    "numpy_ig": os.path.join(output_root, "numpy", "ig"),
    "numpy_saliency": os.path.join(output_root, "numpy", "saliency"),
    "numpy_lime": os.path.join(output_root, "numpy", "lime"),
    # Additive: raw float arrays for CGXA outputs, needed so the evidence
    # extraction stage works on real numbers instead of reverse-engineering
    # values from colormapped JPEG pixels.
    "numpy_consensus": os.path.join(output_root, "numpy", "consensus"),
    "numpy_agreement": os.path.join(output_root, "numpy", "agreement"),
    "numpy_refined_ig": os.path.join(output_root, "numpy", "refined_ig"),
    "numpy_refined_saliency": os.path.join(output_root, "numpy", "refined_saliency"),
    "numpy_refined_lime": os.path.join(output_root, "numpy", "refined_lime"),
    "numpy_mask": os.path.join(output_root, "numpy", "mask"),
    # Additive: per-file physical scale metadata (duration_sec, max_freq_hz,
    # array shape) so a later, independent script can convert pixel
    # coordinates back into seconds/Hz without re-reading the audio.
    "meta": os.path.join(output_root, "meta"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 确保输出目录存在
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)
print(f"Output directories created at: {output_root}")
##############################################################
# Proposed Method: Consensus-Guided XAI Aggregation (CGXA)
##############################################################

USE_CGXA = True

CONSENSUS_BLEND = 0.30

EPS = 1e-8

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

# ============================================================
# Consensus-Guided XAI Aggregation (CGXA)
# ============================================================

class CGXA:

    def __init__(
        self,
        w_ig=0.34,
        w_sal=0.33,
        w_lime=0.33,
        percentile=85
    ):

        self.w_ig = w_ig
        self.w_sal = w_sal
        self.w_lime = w_lime

        self.percentile = percentile

    ########################################################

    def normalize(self, x):

        x = x.astype(np.float32)

        x -= x.min()

        if x.max() > 0:
            x /= x.max()

        return x

    ########################################################

    def lime_to_2d(self, lime_weights, target_shape):
        """
        Convert 1D LIME weights into a 2D map with the same
        shape as the IG and Saliency spectrograms.
        """

        h, w = target_shape

        x_old = np.linspace(0, 1, len(lime_weights))
        x_new = np.linspace(0, 1, w)

        interp = np.interp(x_new, x_old, lime_weights)

        lime_map = np.tile(interp, (h, 1))

        return self.normalize(lime_map)

    ########################################################

    def align_maps(self, ig, sal, lime):

        h = min(
            ig.shape[0],
            sal.shape[0],
            lime.shape[0]
        )

        w = min(
            ig.shape[1],
            sal.shape[1],
            lime.shape[1]
        )

        ig = ig[:h, :w]
        sal = sal[:h, :w]
        lime = lime[:h, :w]

        return ig, sal, lime

    ########################################################

    def agreement_map(self, ig, sal, lime):
        """
        Agreement-aware confidence map.

        High agreement → value close to 1.
        High disagreement OR no signal → value close to 0.

        NOTE: (1 - std) alone is degenerate — three maps that all read
        zero at a pixel have zero std, which scores as "perfect
        agreement" even though nothing important is there. We fix
        this by requiring both low disagreement (low std) AND actual
        signal (nonzero mean magnitude) before calling it "agreement".
        """

        stack = np.stack([ig, sal, lime], axis=0)

        disagreement = np.std(stack, axis=0)
        signal = np.mean(stack, axis=0)

        agreement = (1.0 - disagreement) * signal

        agreement = np.clip(agreement, 0, 1)

        return agreement

    ########################################################

    def agreement_weighted_consensus(self, ig, sal, lime):
        """
        Build the agreement-aware consensus map.
        """

        weighted = (
            self.w_ig * ig +
            self.w_sal * sal +
            self.w_lime * lime
        )

        agreement = self.agreement_map(ig, sal, lime)

        consensus = weighted * agreement

        consensus = self.normalize(consensus)

        return consensus

    ########################################################

    def refine_maps(self, ig, sal, lime, consensus):
        """
        Refine all attribution maps using the consensus mask.
        """

        threshold = np.percentile(
            consensus,
            self.percentile
        )

        mask = consensus >= threshold

        refined_ig = self.normalize(ig * mask)

        refined_sal = self.normalize(sal * mask)

        refined_lime = self.normalize(lime * mask)

        return (
            refined_ig,
            refined_sal,
            refined_lime,
            mask
        )

    ########################################################

    def process(
        self,
        spec_ig,
        spec_sal,
        lime_weights
    ):
        """
        Complete CGXA pipeline.
        """

        spec_ig = self.normalize(spec_ig)

        spec_sal = self.normalize(spec_sal)

        lime_map = self.lime_to_2d(
            lime_weights,
            spec_ig.shape
        )

        spec_ig, spec_sal, lime_map = self.align_maps(
            spec_ig,
            spec_sal,
            lime_map
        )

        consensus = self.agreement_weighted_consensus(
            spec_ig,
            spec_sal,
            lime_map
        )

        refined_ig, refined_sal, refined_lime, mask = (
            self.refine_maps(
                spec_ig,
                spec_sal,
                lime_map,
                consensus
            )
        )

        return {

            "consensus": consensus,

            "agreement": self.agreement_map(
                spec_ig,
                spec_sal,
                lime_map
            ),

            "mask": mask,

            "refined_ig": refined_ig,

            "refined_saliency": refined_sal,

            "refined_lime": refined_lime

        }

# === 4. Initialization ===

print("Loading HuBERT model...")
model = DeepfakeDetector.from_pretrained("nii-yamagishilab/hubert-xlarge-anti-deepfake")
model.to(device)
model.eval()

# CGXA is stateless w.r.t. individual files, so instantiate once.
cgxa = CGXA()

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
    npy_name = base_name + ".npy"

    # 真·断点续传：如果在 load audio 之前发现所有输出都齐了，直接跳过
    path_ig = os.path.join(folders["IG"], jpg_name)
    path_sal = os.path.join(folders["saliency"], jpg_name)
    path_lime = os.path.join(folders["lime"], jpg_name)
    path_consensus = os.path.join(folders["consensus"], jpg_name)
    path_agreement = os.path.join(folders["agreement"], jpg_name)
    path_refined_ig = os.path.join(folders["refined_ig"], jpg_name)
    path_refined_sal = os.path.join(folders["refined_saliency"], jpg_name)
    path_refined_lime = os.path.join(folders["refined_lime"], jpg_name)
    path_npy_ig = os.path.join(folders["numpy_ig"], npy_name)
    path_npy_sal = os.path.join(folders["numpy_saliency"], npy_name)
    path_npy_lime = os.path.join(folders["numpy_lime"], npy_name)
    path_npy_consensus = os.path.join(folders["numpy_consensus"], npy_name)
    path_npy_agreement = os.path.join(folders["numpy_agreement"], npy_name)
    path_npy_refined_ig = os.path.join(folders["numpy_refined_ig"], npy_name)
    path_npy_refined_sal = os.path.join(folders["numpy_refined_saliency"], npy_name)
    path_npy_refined_lime = os.path.join(folders["numpy_refined_lime"], npy_name)
    path_npy_mask = os.path.join(folders["numpy_mask"], npy_name)
    path_meta = os.path.join(folders["meta"], base_name + ".json")

    cgxa_outputs_required = (
        os.path.exists(path_consensus)
        and os.path.exists(path_agreement)
        and os.path.exists(path_refined_ig)
        and os.path.exists(path_refined_sal)
        and os.path.exists(path_refined_lime)
        and os.path.exists(path_npy_ig)
        and os.path.exists(path_npy_sal)
        and os.path.exists(path_npy_lime)
        and os.path.exists(path_npy_consensus)
        and os.path.exists(path_npy_agreement)
        and os.path.exists(path_npy_refined_ig)
        and os.path.exists(path_npy_refined_sal)
        and os.path.exists(path_npy_refined_lime)
        and os.path.exists(path_npy_mask)
        and os.path.exists(path_meta)
    ) if USE_CGXA else True

    if (
        os.path.exists(path_ig)
        and os.path.exists(path_sal)
        and os.path.exists(path_lime)
        and cgxa_outputs_required
    ):
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

        # Load Audio (无截断) — cheap decode, needed regardless of what's cached
        input_wav = load_wav_and_preprocess(full_path, requires_grad=True)
        duration_sec = input_wav.shape[-1] / 16000
        max_freq_hz = 8000

        # === A. Generate IG (skip if already computed and saved) ===
        if os.path.exists(path_npy_ig):
            spec_ig = np.load(path_npy_ig)
        else:
            ig = IntegratedGradients(model)
            # 【关键修改】internal_batch_size=1：防止IG显存爆炸
            attr_ig = ig.attribute(input_wav, target=target_class_idx, n_steps=200, internal_batch_size=1)
            spec_ig = compute_attribution_spectrogram(attr_ig)

            del attr_ig, ig
            torch.cuda.empty_cache()
            gc.collect()

	# === B. Generate Saliency (skip if already computed and saved) ===
	if os.path.exists(path_npy_sal):
    	    spec_sal = np.load(path_npy_sal)
	else:
            saliency = Saliency(model)
    	    nt = NoiseTunnel(saliency)

    	    attr_sal = nt.attribute(
        	input_wav,
        	target=target_class_idx,
        	nt_type="smoothgrad",
        	nt_samples=25,
        	stdevs=0.02,
    	)

    	spec_sal = compute_attribution_spectrogram(attr_sal)

    	del attr_sal, nt, saliency
    	torch.cuda.empty_cache()
    	gc.collect()

        # === C. Generate LIME (skip if already computed and saved) ===
        if os.path.exists(path_npy_lime):
            lime_weights = np.load(path_npy_lime)
        else:
            lime_weights = run_audio_lime(model, input_wav, target_class_idx, num_segments=40, num_samples=1000)

        # === D. Save raw arrays (skip re-saving what's already on disk) ===
        if not os.path.exists(path_npy_ig):
            np.save(path_npy_ig, spec_ig)
        if not os.path.exists(path_npy_sal):
            np.save(path_npy_sal, spec_sal)
        if not os.path.exists(path_npy_lime):
            np.save(path_npy_lime, lime_weights)

        # === E. Run CGXA (needs spec_ig, spec_sal, lime_weights together) ===
        results = cgxa.process(spec_ig, spec_sal, lime_weights) if USE_CGXA else None

        # === F. Save original images (only if missing) ===
        if not os.path.exists(path_ig):
            plot_and_save_spectrogram(spec_ig, f"IG - {filename} ({label_str})", path_ig, duration_sec, max_freq_hz)

        if not os.path.exists(path_sal):
            plot_and_save_spectrogram(spec_sal, f"Saliency - {filename} ({label_str})", path_sal, duration_sec,
                                      max_freq_hz)

        if not os.path.exists(path_lime):
            plot_and_save_lime(lime_weights, f"LIME - {filename} ({label_str})", path_lime, duration_sec)

        if USE_CGXA:
            # === G. Save consensus + agreement ===
            if not os.path.exists(path_consensus):
                plot_and_save_spectrogram(results["consensus"], f"Consensus - {filename} ({label_str})",
                                          path_consensus, duration_sec, max_freq_hz)

            if not os.path.exists(path_agreement):
                plot_and_save_spectrogram(results["agreement"], f"Agreement - {filename} ({label_str})",
                                          path_agreement, duration_sec, max_freq_hz)

            # === H. Save refined maps ===
            if not os.path.exists(path_refined_ig):
                plot_and_save_spectrogram(results["refined_ig"], f"Refined IG - {filename} ({label_str})",
                                          path_refined_ig, duration_sec, max_freq_hz)

            if not os.path.exists(path_refined_sal):
                plot_and_save_spectrogram(results["refined_saliency"], f"Refined Saliency - {filename} ({label_str})",
                                          path_refined_sal, duration_sec, max_freq_hz)

            if not os.path.exists(path_refined_lime):
                plot_and_save_spectrogram(results["refined_lime"], f"Refined LIME - {filename} ({label_str})",
                                          path_refined_lime, duration_sec, max_freq_hz)

            # === H2. Save raw CGXA arrays + physical-scale metadata (additive) ===
            # These are what the evidence extraction stage actually consumes —
            # the JPEGs above are for human eyes only.
            if not os.path.exists(path_npy_consensus):
                np.save(path_npy_consensus, results["consensus"])
            if not os.path.exists(path_npy_agreement):
                np.save(path_npy_agreement, results["agreement"])
            if not os.path.exists(path_npy_refined_ig):
                np.save(path_npy_refined_ig, results["refined_ig"])
            if not os.path.exists(path_npy_refined_sal):
                np.save(path_npy_refined_sal, results["refined_saliency"])
            if not os.path.exists(path_npy_refined_lime):
                np.save(path_npy_refined_lime, results["refined_lime"])
            if not os.path.exists(path_npy_mask):
                np.save(path_npy_mask, results["mask"])
            if not os.path.exists(path_meta):
                meta = {
                    "filename": filename,
                    "base_name": base_name,
                    "prediction": label_str,
                    "target_class_idx": target_class_idx,
                    "duration_sec": float(duration_sec),
                    "max_freq_hz": float(max_freq_hz),
                    "map_shape": list(results["consensus"].shape),  # [freq_bins, time_frames]
                }
                with open(path_meta, "w") as f:
                    json.dump(meta, f, indent=2)

        # === I. Cleanup (after CGXA, not before) ===
        if 'results' in locals():
            del results
        if 'spec_ig' in locals():
            del spec_ig
        if 'spec_sal' in locals():
            del spec_sal
        if 'lime_weights' in locals():
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
