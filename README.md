# Quantum Transfer Learning with ResNet18 and Dressed Quantum Circuits

## Description

This project explores the concept of Classical-to-Quantum Transfer Learning, where features extracted by a classical deep learning model are passed into a quantum machine learning model for final classification. Specifically, we use the pre-trained ResNet18 architecture to extract abstract features from high-resolution images, and feed these features into a 4-qubit variational quantum circuit (VQC) classifier.

The goal of this project is to demonstrate the feasibility and performance of hybrid classical-quantum models in image classification tasks, especially on limited datasets like Hymenoptera (which contains only two classes: ants and bees).

## Key Concepts

### 1. Transfer Learning

Transfer learning involves using a model trained on one task (usually on a large dataset) and adapting it for another task with limited data. In our case, we use the feature extractor part of ResNet18, a well-known convolutional neural network trained on ImageNet, and adapt it for a different task: binary classification of insect images.

### 2. Classical Feature Extraction

ResNet18, pre-trained on the ImageNet dataset, is used to convert raw image data into 512-dimensional feature vectors. This is done by removing the last classification layer and freezing the weights of the remaining network. The resulting model outputs abstract representations of images that capture important visual patterns.

### 3. Quantum Classification

The abstract features from ResNet18 are input into a hybrid classifier. The core of this classifier is a variational quantum circuit (VQC) implemented using 4 qubits. This VQC is “dressed” between classical layers to improve its expressivity and performance. The quantum circuit is trained while keeping ResNet18 frozen.

### 4. Dataset

We use the Hymenoptera dataset, a subset of the ImageNet dataset, which includes labeled images of ants and bees. This dataset is chosen for its simplicity, making it suitable for testing hybrid models.

## File Descriptions

- `transfer_learning.ipynb`: Jupyter Notebook containing the full Python implementation of the hybrid model.
- `3D float design.pptx`: Presentation outlining the concept, architecture, implementation, and results.

## How to Run

1. Install required Python packages:
   ```bash
   pip install torch torchvision qiskit matplotlib numpy
   ```

2. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook transfer_learning.ipynb
   ```

3. Run all cells in order to:
   - Load and preprocess the dataset
   - Extract features using ResNet18
   - Pass the features to a dressed quantum circuit
   - Train the hybrid model
   - Evaluate and visualize the results

## Outcomes

- The hybrid model successfully classifies ants vs. bees using classical-to-quantum transfer learning.
- Demonstrates how quantum-enhanced models can be used for real-world machine learning tasks.
- Encourages further exploration of quantum neural networks and quantum-classical hybrid systems.

## Potential Extensions

- Expand to multi-class datasets
- Try other CNN architectures like VGG or EfficientNet
- Experiment with deeper or alternative quantum circuit designs
- Apply the method to non-image tasks (e.g., finance, medical data)

## Contributors

- Satwik Chaubey  
- Jagrit  
- Sajal Jain
