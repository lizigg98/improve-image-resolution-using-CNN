import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Đường dẫn đến dữ liệu huấn luyện và xác thực
TRAIN_LR_PATH = r"C:\Users\ADMIN\PycharmProjects\BTLAVTGMT\CauTrucMoHinh\Dataset\DIV2K_train_LR_bicubic"
TRAIN_HR_PATH = r"C:\Users\ADMIN\PycharmProjects\BTLAVTGMT\CauTrucMoHinh\Dataset\DIV2K_train_HR"
VALID_LR_PATH = r"C:\Users\ADMIN\PycharmProjects\BTLAVTGMT\CauTrucMoHinh\Dataset\DIV2K_valid_LR_bicubic"
VALID_HR_PATH = r"C:\Users\ADMIN\PycharmProjects\BTLAVTGMT\CauTrucMoHinh\Dataset\DIV2K_valid_HR"


# Hàm tải ảnh từ một thư mục
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


# Hàm xây dựng mô hình CNN cải tiến
def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.3),
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),

        UpSampling2D(size=(2, 2)),
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),

        UpSampling2D(size=(2, 2)),
        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),

        UpSampling2D(size=(2, 2)),
        Conv2D(3, (3, 3), padding="same"),
        Activation("sigmoid")
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["accuracy"])
    return model


# Callback để theo dõi tiến trình huấn luyện và các chỉ số
class TrainingProgress(Callback):
    def __init__(self, total_epochs, valid_lr, valid_hr):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        self.total_time = 0
        self.epoch_bar = None
        self.valid_lr = valid_lr
        self.valid_hr = valid_hr
        self.history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.total_epochs, desc="Training Progress", position=0)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.total_time += epoch_time
        self.epoch_bar.update(1)

        pred_hr = self.model.predict(self.valid_lr)
        pred_hr_binary = (pred_hr > 0.5).astype(np.float32)
        valid_hr_binary = (self.valid_hr > 0.5).astype(np.float32)

        precision = precision_score(valid_hr_binary.flatten(), pred_hr_binary.flatten(), average='macro',
                                    zero_division=0)
        recall = recall_score(valid_hr_binary.flatten(), pred_hr_binary.flatten(), average='macro', zero_division=0)
        f1 = f1_score(valid_hr_binary.flatten(), pred_hr_binary.flatten(), average='macro')

        self.history['loss'].append(logs['loss'])
        self.history['accuracy'].append(logs['accuracy'])
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)
        self.history['f1_score'].append(f1)

        self.epoch_bar.set_postfix({
            "Epoch": epoch + 1,
            "loss": f"{logs['loss']:.4f}",
            "accuracy": f"{logs['accuracy']:.4f}",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1_score": f"{f1:.4f}",
            "epoch_time (s)": f"{epoch_time:.2f}"
        })

    def on_train_end(self, logs=None):
        self.epoch_bar.close()
        print(f"\nTotal training time: {self.total_time:.2f} seconds")
        self.print_final_metrics()
        self.plot_metrics()
        self.save_training_history()

    def print_final_metrics(self):
        print("\nFinal Metrics:")
        print(f"Loss: {self.history['loss'][-1]:.4f}")
        print(f"Accuracy: {self.history['accuracy'][-1]:.4f}")
        print(f"Precision: {self.history['precision'][-1]:.4f}")
        print(f"Recall: {self.history['recall'][-1]:.4f}")
        print(f"F1 Score: {self.history['f1_score'][-1]:.4f}")

    def plot_metrics(self):
        epochs = range(1, len(self.history['loss']) + 1)

        plt.figure(figsize=(18, 12))

        # Biểu đồ Loss
        plt.subplot(3, 2, 1)
        plt.plot(epochs, self.history['loss'], label="Loss", color='red', marker='o')
        plt.title("Loss Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.xticks(epochs)

        # Biểu đồ Accuracy
        plt.subplot(3, 2, 2)
        plt.plot(epochs, self.history['accuracy'], label="Accuracy", color='blue', marker='o')
        plt.title("Accuracy Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.xticks(epochs)

        # Biểu đồ Precision
        plt.subplot(3, 2, 3)
        plt.plot(epochs, self.history['precision'], label="Precision", color='green', marker='o')
        plt.title("Precision Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.xticks(epochs)

        # Biểu đồ Recall
        plt.subplot(3, 2, 4)
        plt.plot(epochs, self.history['recall'], label="Recall", color='orange', marker='o')
        plt.title("Recall Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Recall", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.xticks(epochs)

        # Biểu đồ F1 Score
        plt.subplot(3, 2, 5)
        plt.plot(epochs, self.history['f1_score'], label="F1 Score", color='purple', marker='o')
        plt.title("F1 Score Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("F1 Score", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.xticks(epochs)

        plt.tight_layout()
        output_dir = "training_plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "training_metrics.png"))
        plt.show()

    def save_training_history(self):
        history_df = pd.DataFrame(self.history)
        history_df.to_csv("training_history.csv", index=False)
        print("Training history saved to training_history.csv")


# Hàm huấn luyện mô hình
def train_model(epochs=50, batch_size=32):
    print("Loading training data...")
    train_lr = load_images(TRAIN_LR_PATH, target_size=(128, 128))
    train_hr = load_images(TRAIN_HR_PATH, target_size=(128, 128))

    if len(train_lr) == 0 or len(train_hr) == 0:
        print("Training data could not be loaded. Please check paths!")
        return

    min_len = min(len(train_lr), len(train_hr))
    train_lr = train_lr[:min_len]
    train_hr = train_hr[:min_len]

    print("Loading validation data...")
    valid_lr = load_images(VALID_LR_PATH, target_size=(128, 128))
    valid_hr = load_images(VALID_HR_PATH, target_size=(128, 128))

    if len(valid_lr) == 0 or len(valid_hr) == 0:
        print("Validation data could not be loaded. Please check paths!")
        return

    model = build_cnn_model(input_shape=(128, 128, 3))

    overall_start_time = time.time()

    history = model.fit(
        train_lr,
        train_hr,
        validation_data=(valid_lr, valid_hr),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            TrainingProgress(total_epochs=epochs, valid_lr=valid_lr, valid_hr=valid_hr),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
    )

    model.save("trained_model.h5")
    total_time = time.time() - overall_start_time
    print(f"Training completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    train_model(epochs=30, batch_size=32)