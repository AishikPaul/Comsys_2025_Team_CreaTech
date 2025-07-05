# COMSYS: Lightweight Visual Intelligence using Multi-Scale Features and Attention
** **Test Scripts are provided in folder "Model Weights Files"**
## Training & validation results (Accuracy, Precision, Recall, F1-Score)
### Task A
#### Training
- Accuracy: 0.9709
- Precision: 0.9383
- Recall: 0.9546
- F1-Score: 0.9462
#### Validation
- Accuracy: 0.9479
- Precision: 0.9293
- Recall: 0.8949
- F1-Score: 0.9108

### Task B
#### Training
- Accuracy:
- Precision:
- Recall:
- F1-Score:
#### Validation
- Accuracy:
- Precision:
- Recall:
- F1-Score:

## Overview

This repository presents a two-part visual intelligence system:

- **Task A**: Multi-class image classification using a MobileNetV2 backbone fused with multiscale features and window-based attention.
- **Task B**: Visual similarity detection using a Siamese network with shared encoders to generate robust embeddings.

Both models are designed to be efficient, distortion-resilient, and effective on real-world data, including challenging image conditions.

---

## Task A: Image Classification with Multi-Scale Features

### 🔧 Model Architecture

![Task A Architecture](Diagrams/Task_A_diagram.jpg)

- **Backbone**: MobileNetV2 extracts hierarchical features from 4 stages: `[0:4]`, `[4:7]`, `[7:14]`, `[14:]`.
- **1x1 Convolutions**: Unify the output of each stage to 256 channels.
- **Global Avg Pool + Flatten**: Generates a compact feature vector per stage.
- **Feature Stacking**: Combines all stage vectors into a 3D tensor `(B, 4, 256)`.
- **Window Attention x2**: Applies self-attention across stacked features.
- **Flatten → Classifier**: Outputs class probabilities using a fully connected classifier with softmax activation.

### Training Details

- **Loss**: Categorical CrossEntropy  
- **Optimizer**: Adam  
- **Scheduler**: Learning rate scheduling  
- **Early Stopping**: Enabled to prevent overfitting  
- **Evaluation**: Accuracy & Validation Loss

### Data Augmentation (Albumentations)

- **Geometric**: Resize, Flip, Rotate  
- **Environmental**: Fog, Rain, Sun Flare, Motion Blur  
- **Photometric**: CLAHE, Brightness/Contrast, Gaussian Noise  
- **Normalization**: Mean = 0.5, Std = 0.5

---

## Task B: Visual Similarity Detection with Siamese Network

### Model Architecture

![Task B Architecture](Diagrams/Task_B_diagram.jpg)

- **Backbone**: Same encoder as Task A (`MSFF_WinAttn_MobileNet_Embedding`)
- **Feature Generation**:
  - MobileNetV2 stages (same splits)
  - 1×1 Conv to unify channel depth
  - Global Average Pool + Flatten
  - Stack → Window Attention (x2)
  - DropPath + ReLU → BatchNorm → Linear Layer → L2 Normalization
- **Output**: Compact embedding vector (e.g., 128-D) for each input image

### How the Siamese Network Works

- **Input**: Image pairs (x1, x2)  
- **Twin Encoders**: Shared-weight `MSFF_WinAttn_MobileNet_Embedding`  
- **Distance**: Absolute difference (L1) between embeddings  
- **Classifier**: Fully connected layer with Sigmoid output  
- **Output**: Probability that the images are of the same identity

### Dataset Preparation

- **Positive Pairs**: Same person, possibly one distorted (e.g., foggy, blurry)  
- **Negative Pairs**: Randomly selected from different identities  
- **CSV Logging**: Each pair and its label (1 or 0) is saved

### Training Strategy

- **Loss**: Binary CrossEntropy  
- **Optimizer**: Adam  
- **Class Balance**: Ensured between positive and negative pairs

### Augmentation for Robustness

- Simulates fog, rain, motion blur, sun flare, low-light noise  
- Used selectively to maintain pairwise similarity integrity

---

## Results and Use-Cases

- **Task A**: Robust multi-class classification under various distortions  
- **Task B**: Embedding generation and verification under challenging conditions (e.g., fog, blur)

---

## Applications

- Smart Surveillance  
- Biometric Authentication  
- Image Retrieval  
- Noise-Robust Visual Classification

---

## Folder Structure

