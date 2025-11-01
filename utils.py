# utils.py
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

def preprocess_mnist_images(x):
    # x expected shape (n, 28, 28) or (n, 28, 28, 1)
    x = x.astype('float32') / 255.0
    if x.ndim == 3:
        x = np.expand_dims(x, -1)
    return x

def prepare_image_for_prediction(path, img_size=(28,28)):
    # Read image from path, convert to grayscale, invert if needed and resize to 28x28
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    # resize keeping aspect ratio then pad to 28x28
    h, w = img.shape
    scale = max(h, w)
    # Fit into 20x20 box like classic preprocessing (optional)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    # In MNIST, background is black and digit white. If background is white, invert.
    if np.mean(img) > 0.5:
        img = 1.0 - img
    img = img.reshape(1, img_size[0], img_size[1], 1)
    return img
