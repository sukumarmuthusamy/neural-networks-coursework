# CIFAR-10 Classification using CNN – Image Recognition

This assignment implements a Convolutional Neural Network (CNN) for classifying color images from the CIFAR-10 dataset. The implementation demonstrates the superior performance of CNNs over traditional neural networks for image recognition tasks, leveraging spatial hierarchies and feature extraction capabilities.

## Files Included

- **cifar10_classification_cnn.ipynb** — Complete Jupyter Notebook with code, outputs, visualizations, and detailed comparative analysis for the CNN model and Neural Network baseline models.

## Assignment Overview

This assignment covers advanced concepts in deep learning for computer vision through convolutional neural networks:
- **Dataset:** CIFAR-10 (60,000 32×32 color images across 10 classes)
- **Task:** Build a CNN for multi-class image classification and compare performance with traditional Neural Networks
- **Framework:** Python with TensorFlow/Keras, NumPy, Matplotlib, Scikit-Learn, and Seaborn
- **Focus:** Understanding CNN architecture, feature learning, and performance comparison with fully connected networks

## Dataset Details

The [CIFAR-10](https://keras.io/api/datasets/cifar10/) dataset consists of:
- **Total Images:** 60,000 color images (32×32 pixels, RGB)
- **Classes:** 10 categories — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images
- **Image Format:** 32×32×3 (width × height × color channels)

## Tasks Implemented

### Task 1: Load Dataset and Scale the Pixels
- Load CIFAR-10 dataset using Keras built-in dataset loader (`cifar10.load_data()`)
- Display dataset structure and verify dimensions (training: 50,000 images, test: 10,000 images)
- Normalize pixel values from [0, 255] to [0, 1] by dividing by 255
- Prepare data in the proper format for CNN input (height, width, channels)
- Print pixel statistics before and after scaling for verification

### Task 2: Display Some Images from the Dataset
- Visualize 25 random sample images from the training set in a 5×5 grid layout
- Display images with their corresponding class labels
- Verify correct data loading and understand the visual characteristics of each category
- Set random seed (13) for reproducibility

### Task 3: Define CNN Design and Fit the Model
The CNN architecture consists of:

#### **Convolutional Feature Extraction:**

**Conv Block 1:**
- Conv2D Layer 1: 32 filters, 3×3 kernel, ReLU activation, same padding
- Conv2D Layer 2: 32 filters, 3×3 kernel, ReLU activation, same padding
- MaxPooling2D: 2×2 pool size (downsampling)
- Dropout: 25% (regularization)

**Conv Block 2:**
- Conv2D Layer 3: 64 filters, 3×3 kernel, ReLU activation, same padding
- Conv2D Layer 4: 64 filters, 3×3 kernel, ReLU activation, same padding
- MaxPooling2D: 2×2 pool size (downsampling)
- Dropout: 25% (regularization)

#### **Fully Connected Classification:**
- Flatten Layer: Convert 2D feature maps to 1D vector
- Dense Layer: 512 neurons with ReLU activation
- Dropout: 50% (regularization)
- Output Layer: 10 neurons with SoftMax activation (multi-class classification)

#### **Training Configuration:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 25 (with EarlyStopping monitoring validation accuracy, patience=5)
- Batch Size: 64
- Validation Split: 10% of training data
- Random Seed: 13 (for reproducibility)

**Total Parameters:** 2,168,362 (8.27 MB)

### Task 4: Analyze the Model by Computing Test Accuracy
- Evaluate the trained CNN model on the hold-out test set (10,000 images)
- Compute test accuracy to measure generalization performance
- Calculate training and validation accuracy to assess overfitting
- Compare validation and test accuracies to verify model consistency
- Display detailed evaluation results including test loss

### Task 5: Display Confusion Matrix
- Generate confusion matrix for test set predictions (10,000 samples)
- Visualize classification performance across all 10 classes using heatmap (Seaborn)
- Identify which classes are frequently confused with each other
- Analyze error patterns to understand model strengths and weaknesses
- Display class-wise prediction accuracy along diagonal

## Additional Analysis: CNN vs. Neural Network Comparison

To demonstrate the superiority of CNNs for image classification, three traditional fully connected Neural Network (NN) architectures from Assignment 2 were retrained on the same CIFAR-10 dataset:

### Neural Network Designs Tested:

**Design 1: Single Hidden Layer (128 neurons)**
- Input Layer: 3,072 neurons (flattened 32×32×3 pixels)
- Hidden Layer: 128 neurons with Sigmoid activation
- Output Layer: 10 neurons with SoftMax activation
- Epochs: 100 with batch size (full dataset)

**Design 2: Two Hidden Layers (128 → 64 neurons)**
- Input Layer: 3,072 neurons
- Hidden Layer 1: 128 neurons with Sigmoid activation
- Hidden Layer 2: 64 neurons with Sigmoid activation
- Output Layer: 10 neurons with SoftMax activation
- Epochs: 100

**Design 3: Three Hidden Layers (256 → 128 → 64 neurons)**
- Input Layer: 3,072 neurons
- Hidden Layer 1: 256 neurons with Sigmoid activation
- Hidden Layer 2: 128 neurons with Sigmoid activation
- Hidden Layer 3: 64 neurons with Sigmoid activation
- Output Layer: 10 neurons with SoftMax activation
- Epochs: 100

## Model Analysis

### CNN Performance Summary

The Convolutional Neural Network (CNN) model implemented for this assignment consists of two convolutional blocks, each with two convolutional layers using ReLU activation and max pooling, followed by dropout layers for regularization. After these feature extraction stages, the network includes a dense layer with 512 neurons (ReLU) and an output layer with 10 neurons using SoftMax activation. This architecture allows the model to capture spatial hierarchies such as edges, textures, and shapes, making it well-suited for image recognition tasks like CIFAR-10.

**CNN Model Performance:**
- **Training Accuracy:** 88.02%
- **Validation Accuracy (Best):** 79.32% (at epoch 22)
- **Test Accuracy:** 77.76%
- **Test Loss:** 0.7427

The close alignment between validation accuracy (79.32%) and test accuracy (77.76%) indicates that the model generalizes well without overfitting. The confusion matrix also supports this observation, showing that most predictions lie along the diagonal, with only minor misclassifications between visually similar categories such as cats and dogs or automobiles and trucks. These results demonstrate that the CNN successfully learned meaningful visual features from the dataset rather than memorizing the samples.

### Neural Network (NN) Performance Summary

| Model Architecture | Train Accuracy | Test Accuracy |
|-------------------|----------------|---------------|
| **NN Design 1** (128) | 59.51% | 48.38% |
| **NN Design 2** (128, 64) | 59.87% | 47.73% |
| **NN Design 3** (256, 128, 64) | 58.78% | 46.08% |

### Performance Comparison

When compared to the Neural Network (NN) models designed in Assignment 2 and retrained on the same CIFAR-10 dataset, the difference is significant:

| Model Type | Architecture | Test Accuracy | Improvement |
|------------|--------------|---------------|-------------|
| **Best NN** (Design 1) | Single hidden layer (128 neurons) | 48.38% | Baseline |
| **CNN** (This Assignment) | Two conv blocks + dense layers | 77.76% | **+29.38%** (absolute)<br>**+60.73%** (relative) |

### Key Findings

**Why CNN Outperforms Traditional NN:**

1. **Spatial Feature Learning:** Convolutional layers preserve spatial relationships between pixels, capturing local patterns like edges, corners, and textures that are crucial for image understanding. Traditional fully connected networks flatten the image and lose this spatial structure.

2. **Parameter Efficiency:** CNNs use shared weights in convolutional filters, reducing the number of parameters while still learning robust features. Fully connected networks require many more parameters to achieve similar representation capacity.

3. **Hierarchical Feature Extraction:** The two convolutional blocks learn increasingly complex features:
   - **Early layers (Conv Block 1):** Detect simple edges, colors, and basic shapes
   - **Deeper layers (Conv Block 2):** Recognize textures, patterns, and object parts
   - **Dense layers:** Combine features for final classification

4. **Translation Invariance:** Through pooling layers, CNNs learn features that are invariant to small translations and distortions, making them more robust to variations in object position and scale.

5. **Better Generalization:** The dropout layers (25%, 25%, 50%) and architectural design prevent overfitting, as evidenced by the small gap between validation (79.32%) and test accuracy (77.76%). In contrast, the NN models show signs of overfitting with larger gaps between training and test accuracies.

6. **Efficient Learning:** CNNs converge faster and learn more meaningful representations with fewer training examples compared to fully connected networks.

### Interpretation

The CNN model proved to be **more accurate, stable, and better suited for handling complex image data** than the fully connected NN baseline. The nearly **30% absolute improvement** in test accuracy (from 48.38% to 77.76%) and **61% relative gain** highlight the effectiveness of convolutional layers in learning spatial dependencies and feature hierarchies that a traditional dense NN cannot capture.

Overall, the CNN model proved to be more accurate, stable, and better suited for handling complex image data than the fully connected NN baseline. This validates the fundamental principle that architecture matters significantly in deep learning, especially for vision tasks where spatial structure is essential.

## How to View Results

Open the Jupyter Notebook file (`cifar10_classification_cnn.ipynb`) to see:
- All Python code with proper documentation and comments
- Model architecture summary (2.17M parameters)
- Training history showing convergence over 25 epochs (early stopping at epoch 22)
- Test evaluation metrics and loss curves
- Confusion matrix visualization with heatmap
- Sample image visualizations from the dataset
- Comparative analysis with NN models from Assignment 2
- Detailed performance comparison tables

## Requirements

- **Python 3.x**
- **TensorFlow / Keras** — For building and training the CNN model
- **NumPy** — For numerical operations and array manipulation
- **Matplotlib** — For plotting sample images and training history
- **Seaborn** — For confusion matrix heatmap visualization
- **Scikit-Learn** — For confusion matrix computation and NN baseline models (MLPClassifier)
- **Google Colab / Jupyter Notebook** — Execution environment

## Notes

- **Random Seed:** Set to `13` for reproducibility across dataset sampling, model initialization, and training
- **Early Stopping:** Implemented with patience of 5 epochs monitoring validation accuracy to prevent overfitting; training completed all 25 epochs and restored best weights from epoch 22
- **Data Preprocessing:** Pixel normalization to [0, 1] range is critical for faster convergence and stable training
- **Validation Split:** 10% of training data (5,000 images) reserved for validation during training
- **Dropout Regularization:** Applied at 25% (conv blocks) and 50% (dense layer) to prevent overfitting
- **Batch Size:** 64 images per batch for efficient GPU utilization and stable gradient updates
- **Output Format:** Notebook includes all code outputs and visualizations as required for grading
- **Comparative Analysis:** NN models from Assignment 2 retrained on CIFAR-10 to demonstrate CNN superiority using scikit-learn's MLPClassifier
- **Class Labels:** The 10 CIFAR-10 classes are: airplane (0), automobile (1), bird (2), cat (3), deer (4), dog (5), frog (6), horse (7), ship (8), and truck (9)
- **Model Checkpointing:** Best model saved based on validation accuracy during training as `cnn_cifar10_model.keras`
- **Confusion Matrix Insights:** Most common misclassifications occur between visually similar categories (e.g., cat/dog, automobile/truck)
- **GPU Acceleration:** Training performed on Google Colab with T4 GPU for faster computation
