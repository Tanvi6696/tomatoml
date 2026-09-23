# 🍅 Tomato Leaf Disease Classification using EfficientNetB0

A deep learning project for classifying tomato leaf diseases from images using
EfficientNetB0 transfer learning.

The project also uses Grad-CAM to explain which regions of the leaf influenced
the model's prediction.

## Features

- Tomato leaf disease classification
- EfficientNetB0 transfer learning
- Image augmentation
- Top-3 disease predictions
- Confidence scores
- Grad-CAM visual explanations
- Supports single or multiple image prediction
- Evaluation using confusion matrix and classification metrics

## Model

The model is based on:

- EfficientNetB0
- ImageNet pretrained weights
- Global Average Pooling
- Dropout
- Softmax classification layer

Input image size:

224 × 224 pixels

## Prediction

The notebook allows the user to upload one or multiple tomato leaf images.

For every image, the model provides:

1. Top prediction
2. Top-3 predictions
3. Confidence scores
4. Grad-CAM visualization

Example:

Top 1 → Tomato___Late_blight  
Top 2 → Tomato___Early_blight  
Top 3 → Tomato___Leaf_Mold

## Grad-CAM

Grad-CAM is used to visualize the regions of the tomato leaf that contributed
to the model's prediction.

The visualization contains:

- Original image
- Grad-CAM heatmap
- Heatmap overlay with prediction

## Technologies

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- OpenCV
- Matplotlib
- Pandas
- Google Colab

## Repository Structure

```text
tomato-leaf-disease-ai/
│
├── Tomato_Leaf_Disease_Classification.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── models/
    ├── tomato_disease_efficientnet.keras
    └── class_names.json
