# Liver Disease Detection using Ultrasound Imaging

A deep learning project that classifies **NAFLD (Non-Alcoholic Fatty Liver Disease) vs. Non-NAFLD** from ultrasound images by benchmarking four CNN architectures, using a two-phase transfer-learning strategy with class-imbalance handling and threshold tuning to maximise diagnostic reliability.

## Overview

Liver disease is often diagnosed through manual interpretation of ultrasound scans, a process that can be time-consuming and dependent on radiologist expertise. This project builds and compares four CNN-based classifiers — DenseNet121, ResNet50, EfficientNetB0, and VGG — trained on the **BEHSOF ultrasound dataset** (Kaggle) to automatically distinguish NAFLD from Non-NAFLD cases, and evaluates which architecture generalises best to a real-world, class-imbalanced medical imaging dataset.

## Dataset & Preprocessing

- **Source:** BEHSOF dataset (Kaggle: `mahendarbyra/behsof-dataset`), ultrasound liver images.
- **Classes:** Binary — `NAFLD` vs. `Non-NAFLD`.
- **Class imbalance:** ~1,517 NAFLD vs. ~152 Non-NAFLD images (~10:1) for the DenseNet121/ResNet50 runs; a balanced/oversampled split (1,061 vs. 1,061) with class weighting was used for the EfficientNetB0 run.
- **Split:** Stratified 70/15/15 train/dev/test split.
- **Augmentation:** Rotation (±8°), horizontal flip, brightness jitter (0.9–1.1×), zoom (±5%), applied only to the training set.
- **Input size:** 224×224, preprocessed using each backbone's own `preprocess_input` function.

## Models Compared

| Notebook | Architecture |
|---|---|
| `DenseNet121_and_cnn.ipynb` | DenseNet121 (transfer learning) + custom CNN head |
| `EfficientNetB0_and_cnn.ipynb` | EfficientNetB0 (transfer learning) + custom CNN head |
| `ResNet50_and_cnn.ipynb` | ResNet50 (transfer learning) + custom CNN head |
| `using_vgg_and_feedforward.ipynb` | VGG (transfer learning) + feedforward classifier |

Each backbone was trained with a **two-phase transfer-learning approach**:
1. **Phase 1 (frozen backbone):** ImageNet-pretrained backbone frozen; a custom head (Conv2D + BatchNorm + MaxPooling → GlobalAveragePooling → Dense(128)/Dense(64) with L2 regularisation and 0.5 dropout → sigmoid output) trained for up to 20 epochs with Adam.
2. **Phase 2 (fine-tuning):** Last 20 layers of the backbone unfrozen and fine-tuned for up to 15 more epochs at a low learning rate (1e-5) with gradient clipping (`clipnorm=1.0`).

**Loss & imbalance handling:** A custom **binary focal loss** (γ=2.0, α=0.75) was used in place of standard binary cross-entropy to down-weight easy examples and up-weight the minority (Non-NAFLD) class.

**Callbacks:** `EarlyStopping` (monitored on `val_auc`) and `ReduceLROnPlateau`, plus a custom callback to halt training once training loss dropped below 0.01.

**Threshold tuning:** Instead of a fixed 0.5 cutoff, the classification threshold was selected on the dev set to maximise F1-score, then applied to the held-out test set.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Architectures:** DenseNet121, ResNet50, EfficientNetB0, VGG
- **Environment:** Google Colab
- **Data/Eval:** NumPy, Pandas, Matplotlib, scikit-learn (classification report, ROC-AUC, confusion matrix)

## Repository Structure

```
├── DenseNet121_and_cnn.ipynb        # DenseNet121-based classifier
├── EfficientNetB0_and_cnn.ipynb     # EfficientNetB0-based classifier
├── ResNet50_and_cnn.ipynb           # ResNet50-based classifier
├── using_vgg_and_feedforward.ipynb  # VGG + feedforward classifier
└── README.md
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Ishanya0/Liver-disease-detection-using-ultrasound-imaging.git
   ```
2. Open any notebook in [Google Colab](https://colab.research.google.com/) or a local Jupyter environment.
3. Install dependencies (TensorFlow, NumPy, Pandas, Matplotlib, scikit-learn) if running locally.
4. Run the notebook cells sequentially to reproduce dataset splitting, training, and evaluation.

## Results

On the held-out test set, using the dev-tuned decision threshold:

| Model | Test Accuracy | AUC | Weighted F1 | Sensitivity (Non-NAFLD) | Specificity (NAFLD) |
|---|---|---|---|---|---|
| **ResNet50** | **99.14%** | **0.999** | **0.99** | 90.5% | 99.5% |
| DenseNet121 | 98.06% | 0.996 | 0.97 | 76.2% | 99.5% |
| EfficientNetB0 | 95.62% | 0.963 | 0.96 | 73.9% | 97.8% |

**ResNet50 was the top performer**, reaching **99%+ accuracy** with near-perfect specificity, making it the strongest candidate architecture for this classification task.

## Future Scope

- Consolidate the best-performing architecture (ResNet50) into a single deployable pipeline.
- Add explainability (Grad-CAM) to visualize which regions of the ultrasound image drive predictions.
- Expand the dataset and evaluate generalization across different ultrasound machines/sources.

## Author

**Ishanya Sharma**
[GitHub](https://github.com/Ishanya0)
