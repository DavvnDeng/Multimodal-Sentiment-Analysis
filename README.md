# Multimodal Sentiment Analysis (Experiment 5)

> 华东师范大学 数据科学与工程学院 深度学习实验五：多模态情感分类

This repository contains the implementation of a Multimodal Sentiment Analysis model based on **Late Fusion** strategy. It combines **BERT** (for text) and **ResNet-50** (for image) to classify social media posts into three sentiment categories: `Positive`, `Neutral`, and `Negative`.

## 📌 Project Overview

*   **Task**: Multimodal Sentiment Classification (3 classes).
*   **Model Architecture**: BERT + ResNet50 + Concatenation + MLP.
*   **Key Features**:
    *   **Robust Data Loading**: Fixed GUID parsing bugs for inconsistent CSV formats.
    *   **Data Augmentation**: Random Horizontal Flip & Rotation implemented for training.
    *   **Evaluation**: Comprehensive ablation studies and bad case analysis.
*   **Performance**: Achieved **72.12%** accuracy on the validation set.

## 📂 File Structure

```text
Multimodal-Sentiment-Analysis/
├── data/                   # Data folder (Excluded from git)
│   ├── train.txt           # Training labels
│   ├── test_without_label.txt
│   ├── *.jpg               # Image files
│   └── *.txt               # Text files
├── output/                 # Model checkpoints and logs
│   ├── best_model.pth      # Best trained weights
│   └── bad_cases.csv       # Analysis of error samples
├── src/                    # Source code
│   ├── dataset.py          # Data loader with robust parsing logic
│   ├── model.py            # Model architecture (BERT + ResNet)
│   ├── train.py            # Training loop with Data Augmentation
│   ├── predict.py          # Inference script for test set
│   ├── ablation.py         # Ablation study script
│   └── analyze_bad_cases.py # Error analysis tool
├── requirements.txt        # Python dependencies
├── test_result.txt         # Final submission file
└── README.md               # Project documentation
```

## 🛠️ Environment Requirements

To set up the environment, run:

```
pip install -r requirements.txt
```

**Main Dependencies:**
*   Python 3.8+
*   PyTorch
*   Torchvision
*   Transformers (HuggingFace)
*   Pandas, Pillow, Scikit-learn, Tqdm

## 🚀 Execution Flow

### 1. Data Preparation
Please unzip the provided dataset `实验五数据.zip` and place all files into the `data/` directory.

### 2. Training
Train the model from scratch. This script includes data augmentation and automatically saves the best model to `output/best_model.pth`.

```
cd src
python train.py
```

### 3. Inference (Prediction)
Generate predictions for `test_without_label.txt`. The result will be saved as `test_result.txt` in the root directory.

```
python predict.py
```

### 4. Ablation Study
Evaluate the contribution of each modality (Text-Only vs Image-Only vs Multimodal).

```
python ablation.py
```

### 5. Error Analysis
Identify and save misclassified samples from the validation set to `output/bad_cases.csv` for analysis.

```
python analyze_bad_cases.py
```

## 📊 Results

### Validation Performance
| Model Setting | Text Input | Image Input | Accuracy |
| :--- | :---: | :---: | :---: |
| Image Only | ❌ | ✅ | 60.38% |
| Text Only | ✅ | ❌ | 66.25% |
| **Multimodal (Ours)** | **✅** | **✅** | **72.12%** |

## 📝 References

This project is implemented based on the following papers and repositories:

**Papers:**
*   **BERT**: Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." (NAACL 2019).
*   **ResNet**: He, K., et al. "Deep Residual Learning for Image Recognition." (CVPR 2016).

**Repositories:**
*   [HuggingFace Transformers](https://github.com/huggingface/transformers)
*   [PyTorch Vision](https://github.com/pytorch/vision)
*   [GloGNN (Readme Style Reference)](https://github.com/RecklessRonan/GloGNN)
```
