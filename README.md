# TextOCR
# OCR with CRNN and ResNet Feature Extraction

This project implements an Optical Character Recognition (OCR) system using a hybrid Convolutional Recurrent Neural Network (CRNN) with ResNet as the feature extractor. The model is trained to classify images extracted from PDFs.

## Features
- Converts PDFs to images using `pdf2image`.
- Preprocesses images with augmentations and normalization.
- Uses a ResNet-based feature extractor followed by a Bi-LSTM for sequence modeling.
- Implements training and evaluation functions for classification.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy torch torchvision scikit-learn pdf2image
```
Additionally, install `poppler` for `pdf2image`:

- **Linux**: `sudo apt install poppler-utils`
- **Windows**: Download and add `poppler` to PATH from [here](https://github.com/oschwartz10612/poppler-windows/releases).

## Usage
### 1. Convert PDFs to Images
Run the script to convert PDFs in the `Pdfs/` folder to images in the `images/` folder:

```python
pdfs_to_images("Pdfs", "images")
```

### 2. Train the OCR Model
Run the script to preprocess the dataset, initialize the model, and train it:

```python
train(model, dataloader, criterion, optimizer, num_epochs=30)
```

### 3. Evaluate the Model
Once trained, evaluate the model using:

```python
evaluate(model, dataloader)
```

## Model Architecture
- **Feature Extractor**: ResNet-18 with modified input channels.
- **Sequence Model**: Bi-LSTM with 3 layers.
- **Classification Head**: Fully connected layer.

## Data Processing Pipeline
1. Convert PDFs to images (JPG format).
2. Apply image transformations (resize, grayscale, rotation, jitter, normalization).
3. Encode labels using `LabelEncoder`.
4. Train the CRNN model using the dataset.

## Expected Output
- After training, the model should provide accurate character recognition from the dataset.
- The evaluation function returns an accuracy score based on classification performance.



