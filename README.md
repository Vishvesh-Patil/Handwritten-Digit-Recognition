# Handwritten Digit Recognition (MNIST)

**Project:** Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset.

**Tech stack:** Python, TensorFlow, Keras, NumPy, Matplotlib

**Features**
- Train a CNN on MNIST with Keras (train.py)
- Save and load the trained model (TensorFlow SavedModel / HDF5)
- Inference script to predict digits from images (predict.py)
- Notebook for exploratory data analysis and step-by-step training (notebooks/MNIST_CNN_colab.ipynb)

## Files
- `train.py` — Train the CNN and save model and training history.
- `predict.py` — Load saved model and predict on input images (PNG/JPG).
- `visualize.py` — Visualize predictions on sample images.
- `utils.py` — Helper functions for preprocessing.
- `requirements.txt` — Python dependencies.
- `notebooks/MNIST_CNN_colab.ipynb` — runnable Colab notebook (cells included in repo).

## Quick start (local)
1. Clone repo:
```bash
git clone https://github.com/<your-username>/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
