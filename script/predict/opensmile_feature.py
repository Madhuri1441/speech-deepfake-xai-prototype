# import os
# import numpy as np
# import opensmile
#
# audio_root = "../database"
# save_root = "./features"
#
# audio_formats = (".wav", ".flac", ".mp3", ".m4a")
#
# splits = {
#     "train": "train/con_wav",
#     "dev": "dev/con_wav",
#     "eval": "eval/con_wav",
# }
#
# # openSMILE (88维)
# smile = opensmile.Smile(
#     feature_set=opensmile.FeatureSet.eGeMAPSv02,
#     feature_level=opensmile.FeatureLevel.Functionals,
# )
#
# for split, rel_path in splits.items():
#     input_dir = os.path.join(audio_root, rel_path)
#     output_dir = os.path.join(save_root, split)
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     print(f"\nProcessing {split}...")
#
#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.lower().endswith(audio_formats):
#                 wav_path = os.path.join(root, file)
#
#                 try:
#                     feat_df = smile.process_file(wav_path)
#                     feat = feat_df.values.squeeze().astype(np.float32)
#
#                     save_name = os.path.splitext(file)[0] + ".npy"
#                     save_path = os.path.join(output_dir, save_name)
#
#                     np.save(save_path, feat)
#
#                 except Exception as e:
#                     print("Error:", wav_path, e)
#
# print("\nDone.")
#
import os
import numpy as np
import opensmile
from tqdm import tqdm  # 【新增】导入 tqdm

# === 配置路径 ===
audio_root = "../database"
save_root = "./features"

audio_formats = (".wav", ".flac", ".mp3", ".m4a")

splits = {
    # "train": "train/con_wav",
#    "dev": "dev/con_wav",
  "eval": "eval/con_wav",
}

# === 初始化 OpenSMILE ===
print("Initializing OpenSMILE...")
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# === 开始处理 ===
for split, rel_path in splits.items():
    input_dir = os.path.join(audio_root, rel_path)
    output_dir = os.path.join(save_root, split)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nScanning files for {split} set in {input_dir}...")

    # 【步骤 1】先收集所有待处理的文件路径
    # 这样 tqdm 才能知道总数，从而显示进度条
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            # 过滤掉非音频文件和 MacOS 隐藏文件
            if file.lower().endswith(audio_formats) and not file.startswith("._"):
                full_path = os.path.join(root, file)
                wav_files.append(full_path)

    if len(wav_files) == 0:
        print(f"⚠️ Warning: No audio files found in {input_dir}")
        continue

    print(f"Found {len(wav_files)} files. Extracting features...")

    # 【步骤 2】使用 tqdm 包装文件列表
    # desc=split 显示当前处理的数据集，unit="wav" 显示单位
    for wav_path in tqdm(wav_files, desc=f"Processing {split}", unit="wav"):
        try:
            # 提取文件名用于保存
            file_name = os.path.basename(wav_path)
            save_name = os.path.splitext(file_name)[0] + ".npy"
            save_path = os.path.join(output_dir, save_name)

            # 如果文件已存在，可以选择跳过 (可选)
            # if os.path.exists(save_path):
            #     continue

            # 提取特征
            feat_df = smile.process_file(wav_path)
            # 转换为 numpy (88维)
            feat = feat_df.values.squeeze().astype(np.float32)

            # 保存
            np.save(save_path, feat)

        except Exception as e:
            # 使用 tqdm.write 而不是 print，防止打断进度条显示
            tqdm.write(f"Error processing {wav_path}: {e}")

print("\nAll Done.")