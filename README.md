# XAI-Grounded Explanation Generation for Speech Deepfake Detection with Training-Free Multimodal Large Language Models

Official implementation for the paper **"XAI-Grounded Explanation Generation for Speech Deepfake Detection with Training-Free Multimodal Large Language Models"**.

This repository provides scripts for speech deepfake detection, XAI feature attribution generation, and training-free multimodal large language model based explanation generation.

## Overview

The pipeline contains three main stages:

1. **Prediction**: run speech deepfake detection models and obtain correctly predicted samples.
2. **Feature-to-XAI**: generate explainable AI visualizations from model features, including Integrated Gradients, Saliency, LIME, and SHAP-style explanations.
3. **XAI-to-Text**: use a multimodal large language model to convert XAI images into grounded natural-language explanation regions.

## Repository Structure

```text
.
|-- database/                  # Dataset directory. Download from Zenodo before running scripts.
|-- res/                       # Generated explanation results.
|   |-- one_model_4XAI/        # Results based on one-model XAI settings.
|   `-- three_model_4XAI/      # Results based on three-model XAI settings.
`-- script/
    |-- predict/               # Model feature extraction and prediction scripts.
    |   |-- environment.yaml
    |   |-- opensmile_feature.py
    |   |-- MLP_opensmile.py
    |   |-- hubert_try.py
    |   |-- wav2vec_try.py
    |   `-- wavLM_try.py
    |-- feature2xai/           # Convert model features/predictions to XAI images.
    |   |-- opensmile2xai.py
    |   |-- hubert2xai.py
    |   |-- wav2vec2xai.py
    |   `-- wavlm2xai.py
    `-- xai2txt.py             # Convert XAI images to textual explanations.
```

## Dataset

This project uses the **PartialSpoof Database - Partially Spoofed Audio Dataset for Anti-spoofing**.

Download the dataset from Zenodo:

[https://zenodo.org/records/4817532](https://zenodo.org/records/4817532)

After downloading and extracting the dataset, place it under `database/` with the expected split structure:

```text
database/
|-- dev/
|-- eval/
`-- train/
```

For the evaluation archive, Zenodo provides split files. Concatenate them before extraction:

```bash
cat database_eval.tar.gz.a* > database_eval.tar.gz
tar -zxvf database_eval.tar.gz
```

## Environment

A Conda environment file is provided in `script/predict/environment.yaml`.

```bash
conda env create -f script/predict/environment.yaml
conda activate antideepfake
```

The scripts use common speech, deep learning, XAI, and multimodal LLM packages, including:

- `torch`
- `torchaudio`
- `transformers`
- `fairseq`
- `huggingface_hub`
- `captum`
- `librosa`
- `scikit-learn`
- `matplotlib`
- `qwen-vl-utils`
- `opensmile`

Depending on your CUDA/PyTorch setup, some packages may need to be installed manually.

## Usage

### 1. Run Speech Deepfake Detection

Prediction scripts are stored in `script/predict/`.

Examples:

```bash
python script/predict/wav2vec_try.py
python script/predict/hubert_try.py
python script/predict/wavLM_try.py
```

For OpenSMILE-based prediction, first extract OpenSMILE features and then run the MLP detector:

```bash
python script/predict/opensmile_feature.py
python script/predict/MLP_opensmile.py
```

These scripts evaluate deepfake detection performance and can be used to identify correctly predicted samples for downstream XAI generation.

### 2. Generate XAI Images

XAI image generation scripts are stored in `script/feature2xai/`.

Examples:

```bash
python script/feature2xai/wav2vec2xai.py
python script/feature2xai/hubert2xai.py
python script/feature2xai/wavlm2xai.py
python script/feature2xai/opensmile2xai.py
```

The scripts generate attribution visualizations such as:

- Integrated Gradients
- Saliency
- LIME
- SHAP/OpenSMILE-based explanations

Before running, check and update the path variables near the top of each script, such as:

- `database_root`
- `file_list_path`
- `output_root`

### 3. Convert XAI Images to Text

The XAI-to-text stage is implemented in:

```bash
python script/xai2txt.py
```

This script uses **Qwen2.5-VL-7B-Instruct** to read generated XAI images and produce grounded textual explanation regions, such as the most important time and frequency ranges.

Before running, update the input and output paths in `script/xai2txt.py`:

```python
INPUT_ROOT = "./XAI_Image/wav2vec/eval"
OUTPUT_ROOT = "./XAI_text/wav2vec/eval"
```

## Results

Generated results are stored in `res/`.

- `res/one_model_4XAI/`: explanation outputs from the one-model XAI setting.
- `res/three_model_4XAI/`: explanation outputs from the three-model XAI setting.

Each text file contains the explanation output generated from XAI-grounded multimodal reasoning.
