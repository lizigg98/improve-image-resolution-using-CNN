import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# Đường dẫn dữ liệu
TRAIN_LR_PATH = r"C:\Users\ADMIN\PycharmProjects\BTLAVTGMT\CauTrucMoHinh\Dataset\DIV2K_train_LR_bicubic"
TRAIN_HR_PATH = r"C:\Users\ADMIN\PycharmProjects\BTLAVTGMT\CauTrucMoHinh\Dataset\DIV2K_train_HR"

# Hàm tải ảnh
def load_images(path, target_size=(128, 128)):
    if not os.path.exists(path):
        print(f"Error: Path not found -> {path}")
        return []

    images = []
    for img_file in tqdm(sorted(os.listdir(path)), desc=f"Loading images from {path}"):
        img_path = os.path.join(path, img_file)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

# Xây dựng mô hình CNN cơ bản
def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), padding="same", activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        UpSampling2D(size=(2, 2)),
        Conv2D(3, (3, 3), padding="same", activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["accuracy"])
    return model

# Hàm huấn luyện mô hình
def train_model(epochs=5, batch_size=16):
    print("Loading training data...")
    train_lr = load_images(TRAIN_LR_PATH, target_size=(128, 128))
    train_hr = load_images(TRAIN_HR_PATH, target_size=(128, 128))

    if len(train_lr) == 0 or len(train_hr) == 0:
        print("Training data could not be loaded. Please check paths!")
        return

    # Đồng bộ số lượng ảnh giữa train_lr và train_hr
    min_len = min(len(train_lr), len(train_hr))
    train_lr = train_lr[:min_len]
    train_hr = train_hr[:min_len]

    # Xây dựng và biên dịch mô hình
    model = build_cnn_model(input_shape=(128, 128, 3))

    # Huấn luyện
    history = model.fit(
        train_lr, train_hr,
        batch_size=batch_size,
        epochs=epochs
    )

    # Hiển thị biểu đồ loss
    plt.plot(history.history['loss'], label="Loss", color='red')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model(epochs=20, batch_size=32)  # Có thể thay đổi số epochs và batch size tại đây