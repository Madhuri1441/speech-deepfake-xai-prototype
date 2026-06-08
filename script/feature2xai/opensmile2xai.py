# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import shap
# import opensmile
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset
#
# # ==========================================
# # 0. 路径配置 (锁定 DEV)
# # ==========================================
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# FEATURE_ROOT = os.path.join(CURRENT_DIR, "features")
# DEV_FOLDER = os.path.join(FEATURE_ROOT, "dev")
# MODEL_PATH = os.path.join(CURRENT_DIR, "best_mlp_opensmile.pt")
#
# print(f"📍 脚本目录: {CURRENT_DIR}")
#
#
# # ==========================================
# # 1. 模型结构
# # ==========================================
# class MLPDetector(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(88, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
#             nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(128, 64), nn.ReLU(),
#             nn.Linear(64, 2)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# # ==========================================
# # 2. 数据集加载
# # ==========================================
# class NpyDataset(Dataset):
#     def __init__(self, folder, mean=None, std=None):
#         if not os.path.exists(folder): raise FileNotFoundError(f"❌ 找不到: {folder}")
#         self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")]
#         if not self.files: raise ValueError("文件夹为空")
#         self.mean = mean
#         self.std = std
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         path = self.files[idx]
#         feat = np.load(path).astype(np.float32)
#         if self.mean is not None: feat = (feat - self.mean) / self.std
#         feat = torch.tensor(feat)
#         filename = os.path.basename(path)
#         label = 0 if "CON" in filename else 1
#         return feat, torch.tensor(label), filename
#
#
# def compute_stats(folder):
#     files = [f for f in os.listdir(folder) if f.endswith(".npy")]
#     feats = [np.load(os.path.join(folder, f)) for f in files[:2000]]
#     feats = np.stack(feats)
#     return feats.mean(axis=0), feats.std(axis=0) + 1e-6
#
#
# # ==========================================
# # 主程序
# # ==========================================
# if __name__ == "__main__":
#     # --- A. 初始化特征名 ---
#     try:
#         temp_smile = opensmile.Smile(
#             feature_set=opensmile.FeatureSet.eGeMAPSv02,
#             feature_level=opensmile.FeatureLevel.Functionals,
#         )
#         feature_names = temp_smile.feature_names
#     except:
#         feature_names = [f"Feature_{i}" for i in range(88)]
#
#     # --- B. 准备数据 ---
#     print("📂 计算统计量...")
#     mean, std = compute_stats(DEV_FOLDER)
#     dataset = NpyDataset(DEV_FOLDER, mean, std)
#
#     # --- C. 加载模型 ---
#     device = torch.device("cpu")
#     model = MLPDetector().to(device)
#     if not os.path.exists(MODEL_PATH):
#         print("❌ 模型未找到")
#         exit()
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()
#
#     # --- D. 筛选样本 (找 5 个 High-Confidence Fake) ---
#     print("\n🔍 正在筛选 5 个高置信度 Fake 样本...")
#     loader = DataLoader(dataset, batch_size=1, shuffle=True)
#
#     real_samples_list = []
#     target_samples = []  # 存 (data_numpy, filename, confidence)
#
#     for x, y, fname in loader:
#         label = y.item()
#         # 1. 收集 Real 背景
#         if label == 1 and len(real_samples_list) < 20:
#             real_samples_list.append(x.numpy())
#         # 2. 收集 Fake 目标
#         if label == 0 and len(target_samples) < 5:
#             with torch.no_grad():
#                 probs = torch.softmax(model(x), dim=1)
#                 if probs[0, 0] > 0.95:
#                     target_samples.append((x.numpy(), fname[0], probs[0, 0].item()))
#                     print(f"  ✅ 找到目标 {len(target_samples)}: {fname[0]} (Conf: {probs[0, 0]:.4f})")
#         if len(real_samples_list) >= 20 and len(target_samples) >= 5:
#             break
#
#     if len(target_samples) < 5:
#         print(f"⚠️ 样本不足 5 个，将只处理 {len(target_samples)} 个。")
#
#     background_data = np.concatenate(real_samples_list, axis=0)
#
#     # --- E. 批量运行 Permutation SHAP ---
#     print(f"\n🚀 启动 PermutationExplainer...")
#
#
#     def predict_fn(data_np):
#         data_tensor = torch.tensor(data_np, dtype=torch.float32)
#         with torch.no_grad():
#             outputs = model(data_tensor)
#         return outputs[:, 0].numpy()
#
#
#     explainer = shap.explainers.Permutation(predict_fn, background_data)
#
#     # 循环处理 5 个样本
#     for i, (sample_data, fname, conf) in enumerate(target_samples):
#         plot_filename = f"test{5 + i}.png"
#         print(f"\n[{i + 1}/{len(target_samples)}] 分析 {fname} -> {plot_filename}...")
#
#         # 计算 SHAP
#         shap_values_raw = explainer(sample_data, max_evals=1500)
#
#         # 提取核心数据
#         vals = shap_values_raw.values[0]  # SHAP值 (88,)
#         base = shap_values_raw.base_values[0]  # 基准值 (标量)
#
#         # --- 打印 Top 5 (终端显示) ---
#         print(f"   🔍 Top 5 特征 ({fname}):")
#         feature_importance = list(zip(feature_names, vals))
#         feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
#         for k, (name, val) in enumerate(feature_importance[:5]):
#             direction = "🔴Fake" if val > 0 else "🔵Real"
#             print(f"     {k + 1}. {name:<40} : {val:+.3f} {direction}")
#
#         # --- 绘图 (关键修改部分) ---
#         plt.figure(figsize=(14, 9))  # 画布稍微大一点以便显示文字
#
#         # 【核心修复】手动构建完整的 Explanation 对象
#         # 这样才能保证图上显示特征名和原始数值
#         explanation_obj = shap.Explanation(
#             values=vals,  # 红蓝条长度
#             base_values=base,  # 起点
#             data=sample_data[0],  # 【关键】原始数据值，用于在 Y 轴显示具体大小
#             feature_names=feature_names  # 【关键】特征名称列表
#         )
#
#         shap.waterfall_plot(
#             explanation_obj,
#             max_display=20,
#             show=False
#         )
#         plt.title(f"Why is {fname} Fake?\n(Conf: {conf:.2%})", fontsize=14)
#         plt.tight_layout()
#
#         save_path = os.path.join(CURRENT_DIR, plot_filename)
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"   ✅ 图片已保存: {plot_filename}")
#         plt.close()
#
#     print("\n🎉 全部完成！test5.png 到 test9.png 现在应该包含特征名和数值了。")

#
# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import shap
# import opensmile
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset
#
# # ==========================================
# # 0. 路径配置
# # ==========================================
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# FEATURE_ROOT = os.path.join(CURRENT_DIR, "features")
# DEV_FOLDER = os.path.join(FEATURE_ROOT, "dev")
# MODEL_PATH = os.path.join(CURRENT_DIR, "best_mlp_opensmile.pt")
#
# print(f"📍 脚本目录: {CURRENT_DIR}")
#
#
# # ==========================================
# # 1. 模型结构
# # ==========================================
# class MLPDetector(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(88, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
#             nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(128, 64), nn.ReLU(),
#             nn.Linear(64, 2)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# # ==========================================
# # 2. 数据集加载
# # ==========================================
# class NpyDataset(Dataset):
#     def __init__(self, folder, mean=None, std=None):
#         if not os.path.exists(folder): raise FileNotFoundError(f"❌ 找不到: {folder}")
#         self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")]
#         if not self.files: raise ValueError("文件夹为空")
#         self.mean = mean
#         self.std = std
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         path = self.files[idx]
#         feat = np.load(path).astype(np.float32)
#         # 归一化
#         if self.mean is not None: feat = (feat - self.mean) / self.std
#         feat = torch.tensor(feat)
#         filename = os.path.basename(path)
#         label = 0 if "CON" in filename else 1
#         return feat, torch.tensor(label), filename
#
#
# def compute_stats(folder):
#     files = [f for f in os.listdir(folder) if f.endswith(".npy")]
#     feats = [np.load(os.path.join(folder, f)) for f in files[:2000]]
#     feats = np.stack(feats)
#     return feats.mean(axis=0), feats.std(axis=0) + 1e-6
#
#
# # ==========================================
# # 主程序
# # ==========================================
# if __name__ == "__main__":
#     # --- A. 初始化特征名 ---
#     try:
#         temp_smile = opensmile.Smile(
#             feature_set=opensmile.FeatureSet.eGeMAPSv02,
#             feature_level=opensmile.FeatureLevel.Functionals,
#         )
#         feature_names = temp_smile.feature_names
#     except:
#         feature_names = [f"Feature_{i}" for i in range(88)]
#
#     # --- B. 准备数据 ---
#     print("📂 计算统计量...")
#     mean, std = compute_stats(DEV_FOLDER)
#     dataset = NpyDataset(DEV_FOLDER, mean, std)
#
#     # --- C. 加载模型 ---
#     device = torch.device("cpu")
#     model = MLPDetector().to(device)
#     if not os.path.exists(MODEL_PATH):
#         print("❌ 模型未找到")
#         exit()
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()
#
#     # --- D. 筛选样本 ---
#     print("\n🔍 正在筛选 5 个高置信度 Fake 样本...")
#     loader = DataLoader(dataset, batch_size=1, shuffle=True)
#     real_samples_list = []
#     target_samples = []
#
#     for x, y, fname in loader:
#         label = y.item()
#         if label == 1 and len(real_samples_list) < 20:
#             real_samples_list.append(x.numpy())
#         if label == 0 and len(target_samples) < 5:
#             with torch.no_grad():
#                 probs = torch.softmax(model(x), dim=1)
#                 if probs[0, 0] > 0.95:
#                     target_samples.append((x.numpy(), fname[0], probs[0, 0].item()))
#         if len(real_samples_list) >= 20 and len(target_samples) >= 5:
#             break
#
#     background_data = np.concatenate(real_samples_list, axis=0)
#
#     # --- E. 批量运行 SHAP ---
#     print(f"\n🚀 启动 PermutationExplainer...")
#
#
#     def predict_fn(data_np):
#         data_tensor = torch.tensor(data_np, dtype=torch.float32)
#         with torch.no_grad():
#             outputs = model(data_tensor)
#         return outputs[:, 0].numpy()
#
#
#     explainer = shap.explainers.Permutation(predict_fn, background_data)
#
#     print("\n" + "=" * 100)
#     print("📋 TOP 5 特征详细数据报告")
#     print("=" * 100)
#
#     for i, (sample_data, fname, conf) in enumerate(target_samples):
#         # 计算 SHAP
#         shap_values_raw = explainer(sample_data, max_evals=1500)
#         vals = shap_values_raw.values[0]
#         base = shap_values_raw.base_values[0]
#
#         # 反归一化：获取真实的物理数值
#         original_values = (sample_data[0] * std) + mean
#
#         # 打包数据：(特征名, SHAP值, 原始物理值)
#         combined_data = list(zip(feature_names, vals, original_values))
#         # 按 SHAP 绝对值排序
#         combined_data.sort(key=lambda x: abs(x[1]), reverse=True)
#
#         # --- 打印表格 ---
#         print(f"\n样本 [{i + 1}/5]: {fname} (Fake置信度: {conf:.2%})")
#         print(f"{'-' * 95}")
#         print(
#             f"{'Rank':<4} | {'Feature Name':<45} | {'SHAP(贡献)':<12} | {'Original Value(物理值)':<22} | {'Direction'}")
#         print(f"{'-' * 95}")
#
#         for k, (name, s_val, o_val) in enumerate(combined_data[:5]):
#             direction = "🔴Fake" if s_val > 0 else "🔵Real"
#             # 打印格式化：SHAP保留4位小数，原始值保留4位
#             print(f"{k + 1:<4} | {name:<45} | {s_val:<12.4f} | {o_val:<22.4f} | {direction}")
#         print(f"{'-' * 95}")
#
#         # --- 绘图 ---
#         plt.figure(figsize=(14, 9))
#         explanation_obj = shap.Explanation(
#             values=vals,
#             base_values=base,
#             data=original_values,
#             feature_names=feature_names
#         )
#         shap.waterfall_plot(explanation_obj, max_display=20, show=False)
#         plt.title(f"Why is {fname} Fake?\n(Conf: {conf:.2%})", fontsize=14)
#         plt.tight_layout()
#         save_path = os.path.join(CURRENT_DIR, f"test{5 + i}.png")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
#
#     print(f"\n✅ 所有数据已打印，图片已保存至 test5.png - test9.png")

import os
import torch
import torch.nn as nn
import numpy as np
import shap
import opensmile
import warnings
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_ROOT = os.path.join(CURRENT_DIR, "features")
DEV_FOLDER = os.path.join(FEATURE_ROOT, "eval")
MODEL_PATH = os.path.join(CURRENT_DIR, "best_mlp_opensmile.pt")
FILE_LIST_PATH = "common_test_correct_files.txt"

# 输出目录配置
OUTPUT_ROOT = os.path.join(CURRENT_DIR, "../XAI_text/opensmile/eval/shap")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"📍 脚本目录: {CURRENT_DIR}")
print(f"📂 输出目录: {os.path.abspath(OUTPUT_ROOT)}")


# ==========================================
# 1. 模型结构
# ==========================================
class MLPDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(88, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. 辅助函数
# ==========================================
def compute_stats(folder):
    """计算均值和方差用于归一化和反归一化"""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ 找不到: {folder}")

    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    # 读取前 2000 个样本计算统计量
    feats = [np.load(os.path.join(folder, f)) for f in files[:2000]]
    feats = np.stack(feats)
    return feats.mean(axis=0), feats.std(axis=0) + 1e-6


def get_background_data(folder, n_samples=50):
    """获取 Real (LA) 样本作为 SHAP 背景"""
    all_files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    # 筛选文件名包含 LA 的 (Real)
    la_files = [f for f in all_files if "LA" in f]

    if len(la_files) < n_samples:
        print(f"⚠️ 警告: Real 样本不足 {n_samples} 个，使用全部 {len(la_files)} 个。")
        selected_files = la_files
    else:
        selected_files = la_files[:n_samples]

    feats = [np.load(os.path.join(folder, f)) for f in selected_files]
    return np.stack(feats)


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # --- A. 初始化特征名 ---
    try:
        temp_smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        feature_names = temp_smile.feature_names
    except:
        feature_names = [f"Feature_{i}" for i in range(88)]

    # --- B. 准备数据与统计量 ---
    print("📂 计算统计量 (用于归一化)...")
    mean, std = compute_stats(DEV_FOLDER)

    # --- C. 加载模型 ---
    device = torch.device("cpu")
    model = MLPDetector().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型未找到: {MODEL_PATH}")
        exit()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --- D. 准备 SHAP Explainer ---
    # 1. 获取背景数据 (Raw numpy)
    print("🔍 准备 SHAP 背景数据 (Real样本)...")
    background_raw = get_background_data(DEV_FOLDER, n_samples=50)
    # 2. 归一化背景数据 (用于喂给模型)
    background_norm = (background_raw - mean) / std


    # 3. 定义预测函数 (输入归一化数据 -> 输出 Fake Logits)
    def predict_fn(data_np):
        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(data_tensor)
        return outputs[:, 0].numpy()  # Index 0 is Fake


    # 4. 初始化 Explainer
    print("🚀 初始化 PermutationExplainer...")
    explainer = shap.explainers.Permutation(predict_fn, background_norm)

    # --- E. 读取目标文件列表 ---
    if not os.path.exists(FILE_LIST_PATH):
        print(f"❌ 找不到文件列表: {FILE_LIST_PATH}")
        exit()

    with open(FILE_LIST_PATH, 'r') as f:
        # 过滤掉空行和临时文件
        target_files = [line.strip() for line in f if line.strip() and not line.startswith("._")]

    print(f"📄 待处理文件数: {len(target_files)}")

    # --- F. 批量处理循环 ---
    for wav_filename in tqdm(target_files, desc="Processing SHAP"):
        # 1. 构造文件名和路径
        npy_filename = wav_filename.replace(".wav", ".npy")  # CON_D_xxx.wav -> CON_D_xxx.npy
        npy_path = os.path.join(DEV_FOLDER, npy_filename)

        # 构造输出路径
        txt_filename = wav_filename.replace(".wav", ".txt")  # CON_D_xxx.wav -> CON_D_xxx.txt
        output_path = os.path.join(OUTPUT_ROOT, txt_filename)

        # 跳过已存在的 (可选)
        if os.path.exists(output_path):
            continue

        if not os.path.exists(npy_path):
            print(f"⚠️ 跳过: 找不到特征文件 {npy_filename}")
            continue

        try:
            # 2. 加载并预处理单个样本
            feat_raw = np.load(npy_path).astype(np.float32)  # (88,)
            feat_norm = (feat_raw - mean) / std  # (88,)

            # 增加 batch 维度 (1, 88)
            feat_norm_batch = feat_norm.reshape(1, -1)

            # 3. 计算 SHAP
            # max_evals=1500 平衡速度与精度
            shap_values_obj = explainer(feat_norm_batch, max_evals=1500)

            vals = shap_values_obj.values[0]  # (88,)

            # 4. 筛选 Top 3 正贡献特征
            # 打包数据：(特征名, SHAP值, 原始物理值)
            combined_data = []
            for i in range(len(feature_names)):
                s_val = vals[i]
                if s_val > 0:  # ⚠️ 只保留正贡献 (Positive Contribution)
                    # 反归一化获取原始物理值
                    o_val = (feat_norm[i] * std[i]) + mean[i]
                    combined_data.append((feature_names[i], s_val, o_val))

            # 按 SHAP 值从大到小排序
            combined_data.sort(key=lambda x: x[1], reverse=True)

            # 取前 3 个
            top3_data = combined_data[:3]

            # 5. 保存到 TXT
            with open(output_path, "w", encoding="utf-8") as f:
                if not top3_data:
                    f.write("No positive contribution features found.\n")
                else:
                    for name, s_val, o_val in top3_data:
                        f.write(f"{name}\n")
                        f.write(f"shap value：{s_val:.4f}\n")
                        f.write(f"Original Value：{o_val:.4f}\n")
                        f.write("\n")  # 空行分隔

        except Exception as e:
            print(f"❌ Error processing {wav_filename}: {e}")

    print("\n✅ 所有 Opensmile SHAP 文本提取完成！")