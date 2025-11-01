# train.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from utils import preprocess_mnist_images

def build_model(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

def main(args):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = preprocess_mnist_images(x_train)
    x_test = preprocess_mnist_images(x_test)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model(input_shape=x_train.shape[1:])
    model.summary()

    checkpoint_cb = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    early_cb = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    history = model.fit(x_train, y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_split=0.1,
                        callbacks=[checkpoint_cb, early_cb])

    # Save final model in both HDF5 and SavedModel format
    model.save('model.h5')
    model.save('saved_model')

    plot_history(history)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    main(args)
