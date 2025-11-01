# visualize.py
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from utils import preprocess_mnist_images

def show_samples():
    (x_train, y_train), _ = mnist.load_data()
    plt.figure(figsize=(8,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(str(y_train[i]))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    show_samples()
