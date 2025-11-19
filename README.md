# Brahmi Script Character Recognition

A deep learning-based image classification system for recognizing Brahmi script characters using transfer learning with ResNet50.

## Overview

Brahmi is one of the oldest writing systems of India, dating back to the 3rd century BCE. This project implements a Convolutional Neural Network (CNN) using transfer learning to classify and recognize Brahmi script characters from images.

## Features

- **Transfer Learning**: Utilizes pre-trained ResNet50 model for feature extraction
- **Data Augmentation**: Implements comprehensive image augmentation techniques to improve model generalization
- **Class Balancing**: Automatic computation of class weights to handle imbalanced datasets
- **Early Stopping**: Prevents overfitting with intelligent training interruption
- **Learning Rate Scheduling**: Adaptive learning rate adjustment for optimal convergence
- **Model Checkpointing**: Saves the best performing model during training

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-learn
- Keras

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Bibek478/Brahmi.git
cd Brahmi
```

2. Install required dependencies:
```bash
pip install tensorflow numpy scikit-learn
```

3. Ensure you have the Brahmi dataset (BRAHMI.zip) available in your working directory.

## Usage

### Training the Model

1. Open the Jupyter notebook:
```bash
jupyter notebook Brahmi_code.ipynb
```

2. Update the configuration parameters if needed:
```python
ZIP_PATH = "/content/BRAHMI.zip"  # Path to your dataset
EXTRACT_DIR = "/content/BRAHMI_dataset"
MODEL_SAVE_PATH = "/content/brahmi_model.h5"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
```

3. Run all cells to train the model.

### Quick Start

The training pipeline automatically:
1. Extracts the dataset from the ZIP file
2. Creates training and validation data generators with augmentation
3. Builds the ResNet50-based model
4. Computes class weights for balanced training
5. Trains the model with callbacks
6. Saves the best model

## Model Architecture

The model uses **ResNet50** as the base architecture with the following modifications:

- **Base Model**: ResNet50 pre-trained on ImageNet (top layers removed)
- **Fine-tuning**: Last 10 layers of ResNet50 are trainable
- **Custom Top Layers**:
  - Global Average Pooling 2D
  - Dropout (0.5)
  - Dense layer (1024 units, ReLU activation)
  - Dropout (0.3)
  - Output Dense layer (softmax activation)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224x224 pixels |
| Batch Size | 16 |
| Epochs | 100 (with early stopping) |
| Optimizer | Adam (learning rate: 0.001) |
| Loss Function | Categorical Crossentropy |
| Validation Split | 15% |

## Data Augmentation

The following augmentation techniques are applied during training:

- Rotation: ±20 degrees
- Width/Height Shift: 10%
- Shear Transformation: 10%
- Zoom: ±20%
- Horizontal Flip
- ResNet50 preprocessing

## Callbacks

- **Early Stopping**: Monitors validation loss with patience of 7 epochs
- **ReduceLROnPlateau**: Reduces learning rate by factor of 0.2 when validation loss plateaus (patience: 3 epochs)
- **ModelCheckpoint**: Saves model with best validation accuracy

## Dataset

The project uses a custom Brahmi character dataset (`Brahmi-Dataset.zip`). The dataset should be organized in a directory structure where each subdirectory represents a different character class:

```
BRAHMI_dataset/
    ├── character_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── character_2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

## Results

The model achieves classification of Brahmi script characters through:
- Transfer learning from ImageNet-trained ResNet50
- Balanced training using computed class weights
- Robust data augmentation pipeline
- Optimized hyperparameters with adaptive learning
