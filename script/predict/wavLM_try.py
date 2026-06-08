import os
import torch
import torchaudio
import numpy as np
from sklearn.metrics import roc_curve, f1_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ================================
# Config
# ================================
folder_paths = [
    "../database/dev/con_wav",
]

audio_formats = (".mp3", ".wav", ".flac", ".m4a")
MODEL_NAME = "DavidCombei/wavLM-base-Deepfake_V2"  # 这是一个基于wavlm的deepfake检测模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================================
# Load wavLM deepfake model
# ================================
# 加载特征提取器和模型
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# ================================
# Audio loader
# ================================
def load_wav_and_preprocess(wav_path, target_sr=16000):
    try:
        wav, sr = torchaudio.load(wav_path)
        # 如果是多声道，取平均值转为单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # 重采样
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        # 确保是单声道一维数组 (虽然feature extractor通常处理它，但保持一致性较好)
        return wav.squeeze()
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None


# ================================
# Inference
# ================================
results = {"train": [], "valid": [], "test": []}

for folder_path in folder_paths:
    # 确定数据集划分
    if "train" in folder_path:
        split = "train"
    elif "dev" in folder_path:
        split = "valid"
    elif "eval" in folder_path:
        split = "test"
    else:
        # 如果路径中不包含关键字，可以默认设为valid或者打印警告
        print(f"Warning: Could not determine split for {folder_path}, skipping.")
        continue

    print(f"\nProcessing {split}: {folder_path}")

    cnt = 0  # 计数器重置
    for root, _, files in os.walk(folder_path):
        for file in files:
            # === 修改处：跳过以 . 开头的临时文件 ===
            if file.startswith("."):
                continue

            if file.lower().endswith(audio_formats):
                input_path = os.path.join(root, file)

                wav = load_wav_and_preprocess(input_path)

                if wav is None: continue

                with torch.no_grad():
                    # feature_extractor处理输入
                    inputs = feature_extractor(
                        wav,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    )

                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    logits = model(**inputs).logits
                    # 获取概率
                    probs = torch.softmax(logits, dim=1)
                    # probs 通常是 [fake_prob, real_prob] 或相反，取决于模型配置
                    # 需要确认该模型的label id映射。假设 0: fake, 1: real 是常见的，
                    # 但DavidCombei这个模型通常 0是fake, 1是real。下面逻辑是基于此假设。

                # 存储文件名和概率(Tensor转numpy)
                results[split].append((file, probs.cpu().numpy()[0]))

                cnt += 1
                if cnt > 30:  # 仅用于测试，正式跑请注释掉
                    break
        if cnt > 30: break


# ================================
# Evaluation
# ================================
def evaluate_split(name, data):
    if not data:
        print(f"\n{name} results: No data found.")
        return None

    y_true, y_pred, y_score = [], [], []
    correct_files = []

    for file_name, prob in data:
        # 假设模型输出 prob[0] 是 label 0 的概率, prob[1] 是 label 1 的概率
        # 通常 deepfake 模型 label 0 是 fake/spoof, label 1 是 real/bonafide
        # 具体要看模型 config.id2label

        prob_0 = prob[0]
        prob_1 = prob[1]

        # 预测逻辑：如果 prob_1 (Real) > prob_0 (Fake) 则预测为 1 (Real)
        pred_label = 1 if prob_1 > prob_0 else 0

        # 真实标签逻辑：根据文件名判断
        if "CON" in file_name:  # Conversion -> Fake
            true_label = 0
        elif "LA" in file_name:  # Real (通常LA数据集中有LA_T_...即Real)
            # 注意：LA 数据集可能有 spoof 和 bonafide，文件名规则需确认
            # 假设这里只要不是CON就是Real，或者根据具体数据集调整
            true_label = 1
        else:
            # 无法从文件名判断标签，跳过
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_score.append(prob_1)  # 使用 Real 类的概率作为分数计算 ROC

        if pred_label == true_label:
            correct_files.append(file_name)

    if len(y_true) == 0:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)  # 防止除零警告

    # 计算EER
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fnr - fpr))
        eer = fpr[eer_index]
    else:
        eer = 0.0  # 只有一类样本无法计算EER

    print(f"\n{name} results")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"EER: {eer:.4f}")

    # print(f"\n{name} 正确预测文件 ({len(correct_files)}):")
    # for f in correct_files:
    #     print(f)

    return y_true, y_pred, y_score


# ================================
# Evaluate splits & Overall
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
if len(all_true) > 0:
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_score = np.array(all_score)

    acc = np.mean(all_true == all_pred)
    f1 = f1_score(all_true, all_pred, zero_division=0)

    if len(np.unique(all_true)) > 1:
        fpr, tpr, _ = roc_curve(all_true, all_score, pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    else:
        eer = 0.0

    print("\n==============================")
    print("OVERALL RESULTS")
    print("==============================")
    print(f"Total samples: {len(all_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"EER: {eer:.4f}")
else:
    print("\nNo valid samples processed.")