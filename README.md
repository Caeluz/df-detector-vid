# Deepfake Detector

## Overview

This project provides a deepfake detection solution that leverages a pre-trained model to determine whether videos are genuine or manipulated. It features an intuitive interface where users can upload videos and quickly receive authenticity results.

## Objectives

- Acquire and preprocess the dataset.
- Develop and train a model for deepfake detection.
- Assess the model's accuracy and effectiveness.
- Save the trained model for later use.

## Tools Used

- Python
- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- Scikit-learn
- OpenCV
- PyTorch

## Dataset

The dataset used for training and testing the model is the [DeepFake Detection Challenge Dataset](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2).

## Setup

[See setup instructions](documentation/setup.md) for detailed steps on setting up the environment.

## Model

For this project, a convolutional neural network (CNN) is employed. CNNs are well-suited for image classification problems and can effectively learn patterns in image data to distinguish between REAL and FAKE labels.

## Formulas

The model uses the following formulas for training and evaluation:

### **Input Layer**

- Defines the input shape: RGB image of size **128×128×3**

$$
\text{Input shape} = (128, 128, 3)
$$

---

### **Conv2D Layer**

- Applies learnable filters to extract features
- Uses kernel size = 3, padding = 1, stride = 1

$$
\text{Output size} = \left( \frac{\text{Input size} + 2 \times \text{padding} - \text{kernel size}}{\text{stride}} \right) + 1
$$

$$
\text{Output size per conv block} = \text{same as input due to padding}
$$

---

### **ReLU Activation**

- Applies non-linearity

$$
\text{Output} = \max(0, \text{input})
$$

---

### **BatchNormalization2D Layer**

- Normalizes feature maps to stabilize training

$$
\text{Output} = \gamma \cdot \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:

- $\mu$: mean of batch
- $\sigma^2$: variance
- $\gamma, \beta$: learned scale and shift parameters
- $\epsilon$: small constant for numerical stability

---

### **Dropout2D Layer (p = 0.2)**

- Randomly zeros out entire channels in feature maps during training

$$
\text{Output} = \text{input} \times \text{mask}, \quad \text{mask} \sim \text{Bernoulli}(1 - p)
$$

---

### **MaxPool2D Layer (2×2)**

- Downsamples spatial dimensions by taking max value over 2×2 regions

$$
\text{Output size} = \left( \frac{\text{Input size}}{\text{pool size}} \right)
$$

Applied after each Conv block:

- After Block 1: $128 \rightarrow 64$
- After Block 2: $64 \rightarrow 32$
- After Block 3: $32 \rightarrow 16$
- After Block 4: $16 \rightarrow 8$

---

### **Flatten Layer**

- Converts the final feature map to a 1D vector:

$$
\text{Flattened size} = 512 \times 8 \times 8 = 32,768
$$

---

### **Dense (Linear) Layers**

#### 1. **Linear(32768 → 1024)**

$$
\text{Output} = \text{ReLU}(W_1 \cdot x + b_1)
$$

Followed by BatchNorm1d and Dropout(0.5)

#### 2. **Linear(1024 → 512)**

$$
\text{Output} = \text{ReLU}(W_2 \cdot x + b_2)
$$

Followed by BatchNorm1d and Dropout(0.5)

#### 3. **Linear(512 → 256)**

$$
\text{Output} = \text{ReLU}(W_3 \cdot x + b_3)
$$

Followed by BatchNorm1d and Dropout(0.5)

#### 4. **Linear(256 → 2)**

$$
\text{Output} = \text{Logits (Real vs. Fake)}
$$

---

### **Optional: Temperature Scaling**

Used during inference to calibrate logits:

$$
\text{Logits} = \frac{\text{Output}}{\text{Temperature}}, \quad \text{Temperature} \in \mathbb{R}^{+}
$$

## Evaluation Metrics

1. **Accuracy**: Measures the percentage of correct predictions.

   $$ \text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}} $$

2. **Precision**: Measures the proportion of true positive predictions among all positive predictions.

   $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$

3. **Recall**: Measures the proportion of true positive predictions among all actual positive samples.

   $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$
