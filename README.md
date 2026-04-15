# 🔍 Fake Face Detection using Deep Learning & Transformers

> Binary face authenticity classification using EfficientNet-B0, Vision Transformer, and CLIP with LoRA fine-tuning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-91.5%25-brightgreen?style=flat-square)

---

## 📋 Overview

This project compares three deep learning architectures for detecting AI-generated (fake) human faces:

| Model | Test Accuracy | F1-Score | Trainable Params |
|-------|:---:|:---:|:---:|
| EfficientNet-B0 | 68.4% | 70.3% | 5.3M (100%) |
| ViT-S/16 | 75.2% | 75.2% | 22.0M (100%) |
| **CLIP-ViT + LoRA** | **91.5%** | **92.4%** | **2.1M (2.4%)** |

The CLIP-ViT + LoRA approach achieves the best results while training only **2.4% of total parameters**, demonstrating that parameter-efficient fine-tuning of large pretrained models is highly effective in low-data regimes.

---

## 🗂️ Dataset

- **Source**: [Ciplab Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) (Kaggle)
- **Size**: 2,041 images — 1,081 real faces + 960 synthetic faces
- **Splits**: 70% train / 15% validation / 15% test (stratified)

---

## 🏗️ Architecture Overview

### 1. EfficientNet-B0 (Baseline CNN)
- MBConv blocks with squeeze-and-excitation attention
- 5.3M parameters, pretrained on ImageNet
- Custom classification head (1280 → 2)

### 2. Vision Transformer — ViT-S/16
- 224×224 images split into 196 patches (16×16 each)
- 384-dimensional embeddings, 12 transformer encoder layers
- Trained with Mixup, CutMix, and label smoothing (α=0.1)

### 3. CLIP Vision Transformer + LoRA ⭐
- CLIP ViT-B/16 backbone pretrained on 400M image-text pairs (frozen)
- LoRA adapters (rank=8) injected into Q, K, V, and output projections
- Only 2.1M parameters trained out of 86M total
- AUC: **0.98**

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install transformers peft
pip install albumentations opencv-python
pip install scikit-learn matplotlib seaborn
```

### Clone the repository

```bash
git clone https://github.com/<your-username>/fake-face-detection.git
cd fake-face-detection
```

### Download the dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) and organize it as:

```
data/
├── real/
│   ├── image1.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    └── ...
```

### Train a model

```bash
# EfficientNet-B0
python train.py --model efficientnet

# Vision Transformer
python train.py --model vit

# CLIP + LoRA (recommended)
python train.py --model clip_lora
```

### Evaluate

```bash
python evaluate.py --model clip_lora --checkpoint checkpoints/best_clip_lora.pt
```

---

## 🧪 Training Configuration

All models are trained with:
- **Optimizer**: AdamW (weight decay = 0.01)
- **Learning Rate**: 1e-4 (EfficientNet, ViT) / 5e-5 (CLIP-LoRA)
- **Scheduler**: Cosine annealing
- **Early stopping**: patience = 5 epochs (on validation accuracy)
- **Loss**: Cross-entropy
- **Hardware**: Tesla T4 GPU (Google Colab), batch size 32, mixed precision (FP16)

### Data Augmentation (training only)

| Type | Details |
|------|---------|
| Geometric | Random flip (p=0.5), rotation ±15°, scale/translate ±10% |
| Photometric | Brightness/contrast jitter, saturation/hue shifts |
| Quality | Gaussian blur, motion blur, JPEG compression simulation |
| Normalization | ImageNet stats — mean: [0.485, 0.456, 0.406] |

---

## 📊 Results

### ROC Curves

| Model | AUC |
|-------|:---:|
| EfficientNet-B0 | 0.74 |
| ViT-S/16 | 0.82 |
| CLIP-ViT + LoRA | **0.98** |

### Convergence Speed

- CLIP-LoRA reaches 90% validation accuracy by **epoch 6**
- ViT-S/16 reaches it by epoch 12
- EfficientNet reaches it by epoch 15

### Common Failure Cases

| Category | % of Errors |
|---|:---:|
| Dark / poor lighting | 32% |
| Heavy JPEG compression | 28% |
| Extreme expressions / unusual poses | 24% |

---

## 🔬 Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) and self-attention visualization are used to interpret model decisions:

- **Real faces**: Models focus on eyes, nose, and mouth regions
- **Fake faces**: Models attend to transition boundaries between facial features and texture artifacts typical of generative models
- CLIP attention maps are more precise and consistent than ViT, particularly around eye regions and facial edges

---

## 📁 Project Structure

```
fake-face-detection/
├── data/                   # Dataset directory
├── models/
│   ├── efficientnet.py
│   ├── vit.py
│   └── clip_lora.py
├── utils/
│   ├── dataset.py
│   ├── augmentations.py
│   └── gradcam.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 🔮 Future Work

- [ ] Evaluate on larger, more diverse datasets (StyleGAN, diffusion models, face-swapping)
- [ ] Integrate frequency-domain analysis for periodic artifact detection
- [ ] Develop ensemble architectures combining CNN and transformer strengths
- [ ] Extend to video-based deepfake detection using temporal inconsistencies
- [ ] Explore adversarial robustness against adaptive attacks

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{acharya2024fakeface,
  title={Fake and Real Face Detection Using Deep Learning and Transformer-Based Architectures with Parameter-Efficient Fine-Tuning},
  author={Acharya, Suhani},
  institution={Sardar Vallabhbhai National Institute of Technology, Surat},
  year={2024}
}
```

---

## 🙏 Acknowledgements

- Department of Artificial Intelligence, NIT Surat, for computational resources
- Dr. Nitesh Funde for guidance and support
- [Ciplab at Yonsei University](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) for the dataset
- [OpenAI CLIP](https://github.com/openai/CLIP) and [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685) for the foundational techniques used

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.