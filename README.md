# Quantum-Transfer-Learning
This project demonstrates a hybrid Classical-to-Quantum Transfer Learning pipeline for image classification using ResNet18 as a feature extractor and a 4-qubit variational quantum circuit for classification.

**Overview**

This work explores how classical deep learning models can be combined with quantum circuits for enhanced performance on specific tasks. We use:

ResNet18 (pre-trained on ImageNet) to extract 512-dimensional feature vectors.

A dressed quantum circuit (variational quantum circuit with classical layers) to classify image features.

The Hymenoptera dataset (ants vs. bees) as the target task for classification.

**What is Transfer Learning?**

Transfer learning leverages a model trained on one task and adapts it to a different but related task. We use a frozen ResNet18 model to convert high-resolution images into abstract features, which are then classified using a trainable quantum circuit.

**Architecture**

Feature Extractor: Pre-trained ResNet18 (final layer removed)

Classifier: 4-qubit variational quantum circuit with classical layers (Hybrid Model)

Input Image → ResNet18 → 512 Features → Quantum Circuit → Class → Output

**Dataset**

Hymenoptera Dataset (subset of ImageNet): Binary classification of ant and bee images.

**How to Run**

Install required libraries:

pip install torch torchvision qiskit matplotlib

Open and run the Jupyter notebook transfer_learning.ipynb

View the predictions and visualize model performance.

**Contributors**

Satwik Chaubey
Jagrit
Sajal Jain
