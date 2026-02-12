# MNIST Classification using MLP – Handwritten Digit Recognition

This assignment implements a multilayer perceptron (MLP) neural network for classifying handwritten digits from the MNIST dataset. The implementation includes three neural network architectures with varying complexity to explore the relationship between model depth, training efficiency, and generalization performance.

## Files Included

- **mnist_classification_mlp.ipynb** — Complete Jupyter Notebook with code, outputs, visualizations, and detailed analysis for all three neural network designs.
- **Assignment2_MNIST.csv** — Dataset containing 1,000 MNIST handwritten digit images (100 samples per digit 0-9) with 784 pixel features and corresponding digit labels.

## Assignment Overview

This assignment covers fundamental concepts in neural network design and training through multi-class classification:
- **Dataset:** Assignment2_MNIST.csv (Handwritten Digits 0-9)
- **Task:** Build and compare multiple neural network architectures for digit classification
- **Framework:** Python with Pandas, NumPy, Scikit-Learn, TensorFlow/Keras, and Matplotlib
- **Focus:** Understanding how network architecture affects model performance, training efficiency, and generalization

## Tasks Implemented

### Task 1: Load Dataset and Scale the Pixels
- Upload and import Assignment2_MNIST.csv into a pandas DataFrame
- Display first few rows and dataset summary statistics (shape: 1,000 samples × 785 columns)
- Verify data types and dataset dimensions
- Normalize pixel values from [0, 255] to [0, 1] by dividing by 255 for optimal neural network training

### Task 2: Split Data into Training and Testing Sets
- Partition the dataset into 80% training (800 samples) and 20% testing (200 samples)
- Ensure stratified split to maintain class distribution across both sets
- Prepare data in the proper format (features matrix X and labels vector y)

### Task 3: Define and Train Multiple Neural Network Designs
Three neural network architectures are implemented and compared:

#### **Design 1: Single Hidden Layer (128 neurons)**
- Input Layer: 784 neurons (pixel features)
- Hidden Layer: 128 neurons with Sigmoid activation
- Output Layer: 10 neurons with Softmax activation (multi-class classification)
- Epochs: 100 with batch size 32

#### **Design 2: Two Hidden Layers (128 → 64 neurons)**
- Input Layer: 784 neurons (pixel features)
- Hidden Layer 1: 128 neurons with Sigmoid activation
- Hidden Layer 2: 64 neurons with Sigmoid activation
- Output Layer: 10 neurons with Softmax activation
- Epochs: 100 with batch size 32

#### **Design 3: Three Hidden Layers (256 → 128 → 64 neurons)**
- Input Layer: 784 neurons (pixel features)
- Hidden Layer 1: 256 neurons with Sigmoid activation
- Hidden Layer 2: 128 neurons with Sigmoid activation
- Hidden Layer 3: 64 neurons with Sigmoid activation
- Output Layer: 10 neurons with Softmax activation
- Epochs: 100 with batch size 32

### Task 4: Evaluate Models Using Test Accuracy
- Compute test accuracy for each design on the hold-out test set
- Calculate training accuracy to assess generalization gap (training vs. testing performance)
- Generate detailed performance metrics for comparison

### Task 5: Visualize Training and Test Errors
- Plot training and test error curves across epochs for all three designs
- Create error trajectories showing convergence behavior
- Visualize confusion matrices for each model design
- Generate comparative visualizations to highlight differences in learning dynamics

## Model Analysis

After training and evaluating all three neural network designs, clear differences emerge in how architecture complexity affects performance on the MNIST digit classification task:

**Performance Summary:**

| Metric | Design 1 | Design 2 | Design 3 |
|--------|----------|----------|----------|
| **Training Accuracy** | 99.25% | 99.37% | 100% |
| **Test Accuracy** | 89.5% | 89% | 83% |
| **Generalization Gap** | 9.75% | 10.37% | 17% |
| **Training Time** | 3.25 sec | 3.72 sec | 9.16 sec |

### Key Findings:

**Design 1 (Single Hidden Layer - Best Performer):**
- Achieved the highest test accuracy of 89.5% with a reasonable generalization gap of 9.75%
- Demonstrates that the model learned meaningful patterns without memorizing training data
- Fastest training time at just 3.25 seconds
- Most efficient in terms of computational cost and simplicity
- Handles most digit classifications well with some expected confusion between visually similar digits (2 and 8, 4 and 9)

**Design 2 (Two Hidden Layers - Competitive):**
- Achieved 89% test accuracy, very close to Design 1
- Slightly larger generalization gap (10.37%) suggests beginning signs of overfitting
- Training time moderately increased to 3.72 seconds
- Performance comparable to Design 1 but without clear performance advantage despite added complexity

**Design 3 (Three Hidden Layers - Severe Overfitting):**
- Despite achieving perfect 100% training accuracy, test accuracy drops to only 83%
- A dramatic 17% generalization gap is a clear indicator of severe overfitting
- The model essentially memorized the training data instead of learning generalizable features
- Training time jumped to 9.16 seconds (nearly 3x longer than Design 1) with worse results
- Confusion matrix shows more scattered errors across different digit pairs, indicating poor generalization
- Error curves show test error plateauing around 0.17 from epoch 50-60 while training error continues to decrease

### Interpretation:

The analysis demonstrates that **more architectural complexity is not always better**, especially with smaller datasets like this 1,000-sample MNIST subset. Design 1's single hidden layer architecture strikes the optimal balance between:
- **Learning Capacity:** Sufficient complexity to capture digit patterns
- **Generalization:** Avoiding overfitting with reasonable training-test gap
- **Efficiency:** Minimal computational overhead
- **Maintainability:** Simpler architecture easier to debug and optimize

The overfitting exhibited by Design 3 illustrates a fundamental principle in machine learning: as models become more complex, they require more data to generalize effectively. With limited training data (800 samples), the deeper network finds shortcuts to fit the training set perfectly but fails to learn generalizable features for unseen test data.

### Practical Recommendation:

For MNIST digit classification on this dataset, **Design 1 is the recommended architecture** as it delivers the best balance of accuracy, efficiency, and generalization. The performance improvement from simpler architectures demonstrates that domain knowledge and careful architecture design are more valuable than simply adding more layers.

## How to View Results

Open the Jupyter Notebook file to see:
- All Python code with syntax highlighting
- Execution outputs for each task
- Data visualizations (error curves, confusion matrices, training dynamics)
- Summary statistics and comparative metrics
- Detailed model interpretation and analysis
- Side-by-side comparison of all three neural network designs

## Requirements

- Python 3.x
- Pandas (data manipulation)
- NumPy (numerical computations)
- Scikit-Learn (preprocessing and metrics)
- TensorFlow/Keras (neural network implementation)
- Matplotlib (data visualization)

## Notes

This notebook was developed in Google Colab and includes all required outputs, visualizations, and analysis as per assignment specifications. The three-design comparison provides valuable insights into the trade-offs between model complexity, generalization, and computational efficiency in neural network design.
