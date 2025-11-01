# predict.py
import sys
import numpy as np
from tensorflow.keras.models import load_model
from utils import prepare_image_for_prediction
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    model_path = 'best_model.h5'
    if not os.path.exists(model_path):
        model_path = 'model.h5'  # fallback
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    model = load_model(model_path)
    img_path = sys.argv[1]
    img = prepare_image_for_prediction(img_path)
    preds = model.predict(img)
    label = np.argmax(preds, axis=1)[0]
    prob = np.max(preds)
    print(f"Prediction: {label} (confidence: {prob:.3f})")

if __name__ == '__main__':
    main()
