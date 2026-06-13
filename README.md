# 🩺 Facial Skin Diseases Classification using Deep Learning

> **GSSoC 2026 Contribution** | Deep Learning Simplified Repository

---

## 📌 Project Overview

This project classifies facial/skin diseases using multiple deep learning architectures and compares their performance to find the best-fitted model. It includes Exploratory Data Analysis (EDA), data preprocessing, model training, and a final comparison of all models.

---

## 📂 Dataset

**Original Issue Dataset:** [Facial Skin Diseases Dataset – Kaggle](https://www.kaggle.com/datasets/osmankagankurnaz/facial-skin-diseases-dataset)

> ⚠️ The original dataset contained only one class (Acne), which is insufficient for multi-class classification and model comparison. After discussion with the maintainer ([@abhisheks008](https://github.com/abhisheks008)), permission was granted to use a richer dataset.

**Dataset Used:** [DermNet – Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)

### Why DermNet?
- Contains 23 skin disease categories with thousands of real dermatology images
- Widely used in medical image classification research
- Enables meaningful multi-class comparison across architectures

### Subset Created
To keep training feasible and balanced, a **7-class subset** was created:

| Class | Train Images | Test Images |
|-------|-------------|-------------|
| Actinic Keratosis / Basal Cell Carcinoma | 300 | 60 |
| Eczema | 300 | 60 |
| Melanoma / Skin Cancer / Nevi & Moles | 300 | 60 |
| Nail Fungus & other Nail Disease | 300 | 60 |
| Psoriasis / Lichen Planus | 300 | 60 |
| Tinea Ringworm / Candidiasis / Fungal Infections | 300 | 60 |
| Warts Molluscum & other Viral Infections | 300 | 60 |
| **Total** | **2100** | **420** |

---

## 🔍 Exploratory Data Analysis (EDA)

Before training, the following EDA was performed:
- Class distribution analysis to verify balance across 7 categories
- Sample image visualization per class
- Image dimension and pixel distribution analysis
- Data augmentation strategy planning

---

## 🧠 Models Implemented

All 7 models were trained and evaluated:

| # | Model | Type |
|---|-------|------|
| 1 | Custom CNN | Built from scratch |
| 2 | VGG16 | Transfer Learning + Fine-tuning |
| 3 | ResNet50 | Transfer Learning + Fine-tuning |
| 4 | EfficientNetB0 | Transfer Learning + Fine-tuning |
| 5 | MobileNetV2 | Transfer Learning + Fine-tuning |
| 6 | DenseNet121 | Transfer Learning + Fine-tuning |
| 7 | Xception | Transfer Learning + Fine-tuning |

### Training Strategy
- **Pretrained weights:** ImageNet
- **Fine-tuning:** Last 30 layers unfrozen for all transfer learning models
- **Data Augmentation:** Rotation, width/height shift, shear, zoom, horizontal flip
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy

---

## 📊 Results

### Model Accuracy Comparison

| Model | Test Accuracy | Test Loss |
|-------|:------------:|:---------:|
| 🥇 VGG16 | **56.19%** | 1.4381 |
| 🥈 Xception | 47.62% | 1.4501 |
| 🥉 MobileNetV2 | 38.81% | 1.7851 |
| ResNet50 | 38.10% | 1.7304 |
| DenseNet121 | 35.48% | 2.1343 |
| Custom CNN | 24.05% | 2.9582 |
| EfficientNetB0 | 14.29% | 1.9900 |

### 🏆 Best Model: VGG16 (56.19% Test Accuracy)

VGG16 with fine-tuning outperformed all other architectures on this dataset. Its deep convolutional structure combined with ImageNet pretrained weights proved most effective for skin disease feature extraction.

> 📁 All training history plots, confusion matrices, and classification reports are available in the `Images/` folder.

---

## 🗂️ Project Structure

```
Facial Skin Diseases Classification using DL/
│
├── Images/                          
│   ├── all_models_comparison.png    # Final accuracy comparison bar chart
│   ├── all_models_loss.png          # Loss comparison chart
│   ├── model_comparison.csv         # Results summary table
│   ├── VGG16_history.png
│   ├── VGG16_confusion_matrix.png
│   ├── VGG16_classification_report.txt
│   └── ... (plots for all 7 models)
│
├── Dataset/
│   └── dataset_info.md              # Dataset source and description
│
├── Model/
│   └── FacialSkinDetection_Complete.ipynb   # Full training notebook
│── README.md    
└── requirements.txt
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended — GPU required)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `FacialSkinDetection_Complete.ipynb`
3. Go to `Runtime > Change runtime type > T4 GPU`
4. Run all cells **top to bottom** (14 cells total)

**What each cell does:**

| Cell | Description |
|------|-------------|
| 1 | Install packages & verify GPU |
| 2 | Mount Google Drive |
| 3 | Download DermNet via kagglehub & create 7-class subset |
| 4 | Config, hyperparameters & data generators |
| 5 | Shared helper functions (train, plot, evaluate) |
| 6–12 | Train each of the 7 models |
| 13 | Generate final comparison charts & summary table |
| 14 | Save all results & models to Google Drive |

> ⏱️ Expected runtime: ~2–3 hours on T4 GPU

### Option 2: Local Machine

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Deep-Learning-Simplified.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook FacialSkinDetection_Complete.ipynb
```
> ⚠️ A GPU is strongly recommended. CPU training will be very slow.

---

## 📦 Requirements

See `requirements.txt` for full list. Key dependencies:
- `tensorflow >= 2.12`
- `kagglehub`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `opencv-python`

---

## 🔮 Future Work

- Increase training images per class (500+) for better generalization
- Tune EarlyStopping patience and learning rates for each model individually
- Experiment with ensemble methods combining VGG16 + Xception
- Try Vision Transformers (ViT) for comparison
- Deploy best model (VGG16) as a web app using Streamlit or Flask

---

## 👩‍💻 Contributor

**Radhika**
- GitHub: [@Radhika-789](https://github.com/Radhika-789)
- GSSoC 2026 Contributor

---

## 📜 License

This project is part of the [Deep Learning Simplified](https://github.com/abhisheks008/DL-Simplified) open source repository under GSSoC 2026.
