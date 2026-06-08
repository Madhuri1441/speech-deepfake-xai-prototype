import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm  # 进度条库

# === 1. 配置路径 ===
# change here for different ssl feature
INPUT_ROOT = "./XAI_Image/{wav2vec}/eval"

OUTPUT_ROOT = "./XAI_text/wav2vec/eval"

# 需要处理的子文件夹名称
SUB_FOLDERS = ["IG", "saliency", "lime"]

# === 2. 模型初始化 ===
print("正在加载 Qwen2.5-VL 模型...")
model_img = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor_img = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("模型加载完成。")


# === 3. 定义 Prompt 模板 ===

def get_ig_prompt(image_path):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": '''
You are an expert in audio deepfake detection and explainable AI.
Rules:
- Do not hallucinate information not supported by XAI
- Use approximate numeric ranges when needed
- Follow the output format strictly
'''}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": '''An XAI results for a deepfake audio is provided.

This is an Integrated Gradients attribution map.
The x-axis is time (seconds).
The y-axis is frequency (Hz).
Red indicates attribution strength.
Darker red means stronger attribution.

Task:
Find the SINGLE strongest attribution region in the image.

Output format:
TIME_REGION: [start-end] seconds
FREQUENCY_REGION: [low-high] Hz

Rules:
- Only output one region
- Use approximate ranges
- If uncertain, choose the most visually intense red area
- Do not explain
'''}
            ],
        },
    ]


def get_saliency_prompt(image_path):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": '''
You are an expert in audio deepfake detection and explainable AI.
Rules:
- Do not hallucinate information not supported by XAI
- Use approximate numeric ranges when needed
- Follow the output format strictly
'''}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": '''An XAI results for a deepfake audio is provided.

Describe only the attribution evidence visible in this Saliency map. Red indicates positive attribution. Darker red indicates higher importance.

This is an Saliency attribution map.
The x-axis is time (seconds).
The y-axis is frequency (Hz).
Red indicates attribution strength.
Darker red means stronger attribution.

Task:
Find the SINGLE strongest attribution region in the image.

Output format:
TIME_REGION: [start-end] seconds
FREQUENCY_REGION: [low-high] Hz

Rules:
- Only output one region
- Use approximate ranges
- If uncertain, choose the most visually intense red area
- Do not explain
'''}
            ],
        },
    ]


def get_lime_prompt(image_path):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": '''
You are an expert in audio deepfake detection and explainable AI.
Rules:
- Do not hallucinate information not supported by XAI
- Use approximate numeric ranges when needed
- Follow the output format strictly
'''}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": '''An XAI results for a deepfake audio is provided.

This is a LIME figure.
Each bar represents a time segment.

Task:
Find the time segment with the highest positive importance value.

Output:
TIME_REGION: [start-end] seconds

Rules:
- Only choose the highest positive bar
- Ignore negative bars
- Do not explain
'''}
            ],
        },
    ]


# === 4. 推理函数 ===
def run_inference(image_path, folder_type):
    # 根据文件夹类型选择对应的 Prompt
    if folder_type == "IG":
        messages = get_ig_prompt(image_path)
    elif folder_type == "saliency":  # 注意这里可能是小写或大写，取决于你的文件夹命名
        messages = get_saliency_prompt(image_path)
    elif folder_type == "lime":
        messages = get_lime_prompt(image_path)
    else:
        return None

    # 准备输入
    text = processor_img.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor_img(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成
    generated_ids = model_img.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor_img.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


# === 5. 主循环 ===

def main():
    # 遍历三个类型的文件夹
    for folder_name in SUB_FOLDERS:
        # 构建输入和输出的具体路径
        # 注意：这里做了一个不区分大小写的匹配，因为有时候文件夹叫 Saliency 有时候叫 saliency
        # 我们假设 INPUT_ROOT 下存在这几个文件夹

        # 寻找实际的文件夹名（处理大小写问题）
        actual_folder_name = None
        if os.path.exists(os.path.join(INPUT_ROOT, folder_name)):
            actual_folder_name = folder_name
        elif os.path.exists(os.path.join(INPUT_ROOT, folder_name.capitalize())):  # Saliency
            actual_folder_name = folder_name.capitalize()
        elif os.path.exists(os.path.join(INPUT_ROOT, folder_name.upper())):  # IG
            actual_folder_name = folder_name.upper()

        if not actual_folder_name:
            print(f"⚠️ 警告: 找不到文件夹 {folder_name}，跳过。")
            continue

        input_dir = os.path.join(INPUT_ROOT, actual_folder_name)
        output_dir = os.path.join(OUTPUT_ROOT, actual_folder_name)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有图片文件
        valid_extensions = ('.jpg', '.png', '.jpeg')
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
        image_files.sort()

        print(f"\n📂 正在处理文件夹: {actual_folder_name} (共 {len(image_files)} 张图片)")

        # 使用 tqdm 显示进度条
        for img_file in tqdm(image_files, desc=f"Processing {actual_folder_name}"):
            # 构建完整路径
            img_path = os.path.join(input_dir, img_file)

            # 构建输出 txt 路径
            base_name = os.path.splitext(img_file)[0]
            txt_file = base_name + ".txt"
            txt_path = os.path.join(output_dir, txt_file)

            # 检查是否已存在（断点续传）
            if os.path.exists(txt_path):
                continue

            try:
                # 运行推理
                # 传入 folder_name (小写归一化后的键值，或者是我们定义的 SUB_FOLDERS 中的值) 来决定用哪个 prompt
                # 注意：这里需要映射回 "IG", "saliency", "lime"
                prompt_type = folder_name
                if prompt_type == "saliency": prompt_type = "saliency"  # 保持一致

                result_text = run_inference(img_path, prompt_type)

                if result_text:
                    # 保存结果
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(result_text)
                else:
                    print(f"❌ Error: 未生成文本 - {img_file}")

            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")
                # 可选：写入错误日志
                with open("error_log.txt", "a") as ef:
                    ef.write(f"{img_path}: {str(e)}\n")

    print("\n✅ 所有任务处理完成！")


if __name__ == "__main__":
    main()