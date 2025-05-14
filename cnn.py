import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
num_classes = 10

# Preprocessing untuk CNN
x_train_cnn = x_train.astype("float32") / 255.0
x_test_cnn = x_test.astype("float32") / 255.0

# Ubah ukuran gambar ke 96x96 untuk MobileNetV2 (karena MobileNetV2 memerlukan gambar lebih besar)
x_train_mnet = tf.image.resize(x_train[..., np.newaxis], (96, 96))  # Ubah ke dimensi (28, 28, 1) menjadi (28, 28, 3)
x_test_mnet = tf.image.resize(x_test[..., np.newaxis], (96, 96))
x_train_mnet = tf.repeat(x_train_mnet, 3, axis=-1)  # Mengulangi saluran warna (grayscale ke RGB)
x_test_mnet = tf.repeat(x_test_mnet, 3, axis=-1)
x_train_mnet = preprocess_input(x_train_mnet)
x_test_mnet = preprocess_input(x_test_mnet)

# Label one-hot
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# CNN Sederhana
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# MobileNetV2 Transfer Learning
def build_mobilenetv2():
    base_model = MobileNetV2(include_top=False, input_shape=(96, 96, 3), weights='imagenet', pooling='avg')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Compile & Train
def train_model(model, x_train, y_train, x_val, y_val, name):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Training {name}...")
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val), verbose=2)
    return model, history

# Jalankan training
cnn_model, cnn_hist = train_model(build_cnn(), x_train_cnn, y_train_cat, x_test_cnn, y_test_cat, "CNN")
mobilenet_model, mobilenet_hist = train_model(build_mobilenetv2(), x_train_mnet, y_train_cat, x_test_mnet, y_test_cat, "MobileNetV2")

# Evaluasi
cnn_acc = cnn_model.evaluate(x_test_cnn, y_test_cat, verbose=0)[1]
mnet_acc = mobilenet_model.evaluate(x_test_mnet, y_test_cat, verbose=0)[1]
print(f"\nAkurasi CNN: {cnn_acc:.4f}")
print(f"Akurasi MobileNetV2: {mnet_acc:.4f}")

# Plot hasil akurasi validasi
plt.plot(cnn_hist.history['val_accuracy'], label='CNN')
plt.plot(mobilenet_hist.history['val_accuracy'], label='MobileNetV2')
plt.title("Perbandingan Akurasi Validasi")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
