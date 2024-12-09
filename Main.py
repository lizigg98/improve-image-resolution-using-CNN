import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd

class ImageSuperResolutionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng Dụng Tăng Độ Phân Giải Ảnh")
        self.root.geometry("1400x700")
        self.root.configure(bg="#f3f4f6")

        self.input_image = None
        self.original_image = None
        self.output_image = None
        self.model = None
        self.history_data = None  # Biến để lưu trữ dữ liệu lịch sử huấn luyện

        self.setup_ui()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#007bff", height=100)
        header_frame.pack(fill="x")

        header_label = tk.Label(
            header_frame,
            text="ỨNG DỤNG TĂNG ĐỘ PHÂN GIẢI ẢNH - CNN",
            font=("Arial", 28, "bold"),
            fg="#ffffff",
            bg="#007bff"
        )
        header_label.pack(pady=20)

        # Main Frames
        main_frame = tk.Frame(self.root, bg="#f3f4f6")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        left_frame = tk.Frame(main_frame, bg="#ffffff", relief="groove", borderwidth=2)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)

        right_frame = tk.Frame(main_frame, bg="#ffffff", relief="groove", borderwidth=2)
        right_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Buttons and Controls
        model_label = tk.Label(left_frame, text="Mô Hình & Chức Năng", font=("Arial", 18, "bold"), bg="#ffffff", fg="#333333")
        model_label.pack(pady=10)

        self.load_model_button = ttk.Button(left_frame, text="Tải Mô Hình Đã Huấn Luyện", command=self.load_trained_model)
        self.load_model_button.pack(fill="x", pady=15)

        self.load_history_button = ttk.Button(left_frame, text="Tải Lịch Sử Huấn Luyện", command=self.load_history)
        self.load_history_button.pack(fill="x", pady=15)

        self.plot_history_button = ttk.Button(left_frame, text="Vẽ Biểu Đồ Lịch Sử", command=self.plot_history)
        self.plot_history_button.pack(fill="x", pady=15)

        image_label = tk.Label(left_frame, text="Quản Lý Ảnh", font=("Arial", 18, "bold"), bg="#ffffff", fg="#333333")
        image_label.pack(pady=10)

        self.load_image_button = ttk.Button(left_frame, text="Tải Ảnh", command=self.load_image)
        self.load_image_button.pack(fill="x", pady=15)

        self.enhance_button = ttk.Button(left_frame, text="Tăng Độ Nét & Độ Phân Giải", command=self.enhance_image)
        self.enhance_button.pack(fill="x", pady=15)

        self.save_button = ttk.Button(left_frame, text="Lưu Ảnh", command=self.save_image)
        self.save_button.pack(fill="x", pady=15)

        # Image Display Panels
        self.input_panel = tk.LabelFrame(right_frame, text="Ảnh Đầu Vào", font=("Arial", 16), bg="#f3f4f6")
        self.input_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.input_image_label = tk.Label(self.input_panel, bg="#e3e3e3", relief="solid", borderwidth=1)
        self.input_image_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.input_res_label = tk.Label(self.input_panel, text="", bg="#ffffff")
        self.input_res_label.pack(pady=5)

        self.output_panel = tk.LabelFrame(right_frame, text="Ảnh Đã Xử Lý", font=("Arial", 16), bg="#f3f4f6")
        self.output_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.output_image_label = tk.Label(self.output_panel, bg="#e3e3e3", relief="solid", borderwidth=1)
        self.output_image_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.output_res_label = tk.Label(self.output_panel, text="", bg="#ffffff")
        self.output_res_label.pack(pady=5)

    def load_trained_model(self):
        model_path = filedialog.askopenfilename(title="Chọn Mô Hình", filetypes=[("H5 Files", "*.h5")])
        if model_path:
            self.model = tf.keras.models.load_model(model_path,
                                                    custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
            messagebox.showinfo("Info", "Mô hình đã được tải thành công!")

    def load_history(self):
        history_path = filedialog.askopenfilename(title="Chọn Tệp Lịch Sử", filetypes=[("CSV Files", "*.csv")])
        if history_path:
            self.history_data = pd.read_csv(history_path)
            messagebox.showinfo("Info", "Lịch sử đã được tải thành công!")

    def plot_history(self):
        if self.history_data is not None:
            plt.figure(figsize=(12, 6))

            # Vẽ biểu đồ loss
            plt.subplot(1, 2, 1)
            plt.plot(self.history_data['loss'], label='Loss', color='red')
            plt.title('Loss Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # Vẽ biểu đồ accuracy
            plt.subplot(1, 2, 2)
            plt.plot(self.history_data['accuracy'], label='Accuracy', color='green')
            plt.title('Accuracy Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("Warning", "Vui lòng tải lịch sử huấn luyện trước!")

    def create_residual_cnn(self):
        def residual_block(x):
            res = Conv2D(64, (3, 3), padding="same")(x)
            res = BatchNormalization()(res)
            res = ReLU()(res)
            res = Conv2D(64, (3, 3), padding="same")(res)
            res = BatchNormalization()(res)
            return Add()([x, res])

        input_layer = Input(shape=(128, 128, 3))
        x = Conv2D(64, (3, 3), activation='relu', padding="same")(input_layer)

        for _ in range(6):
            x = residual_block(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

        self.model = Model(inputs=input_layer, outputs=x)
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        messagebox.showinfo("Info", "CNN nâng cao đã được tạo thành công!")

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Chọn Ảnh", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.original_image = Image.open(file_path).resize((280, 280))

            blurred_image = self.original_image.filter(ImageFilter.GaussianBlur(1))
            self.input_image = blurred_image
            self.show_image(self.input_image, self.input_image_label)
            # Hiển thị độ phân giải ảnh gốc
            self.input_res_label.config(text=f"Độ phân giải: {self.original_image.size[0]} x {self.original_image.size[1]}")

    def enhance_image(self):
        if self.original_image:
            # Tăng cường ảnh
            enhancer = ImageEnhance.Sharpness(self.original_image)
            enhanced_image = enhancer.enhance(3.0).filter(ImageFilter.DETAIL)

            # Cập nhật ảnh đã xử lý và hiển thị
            self.output_image = enhanced_image
            self.show_image(self.output_image, self.output_image_label)

            # Hiển thị độ phân giải cho cả hai ảnh
            original_res = self.original_image.size
            enhanced_res = self.output_image.size

            # Hiển thị độ phân giải của ảnh gốc (không thay đổi)
            self.input_res_label.config(text=f"Độ phân giải gốc: {original_res[0]} x {original_res[1]}")
            # Hiển thị độ phân giải của ảnh đã tăng
            self.output_res_label.config(text=f"Độ phân giải đã tăng: {enhanced_res[0]} x {enhanced_res[1]}")
        else:
            messagebox.showwarning("Warning", "Vui lòng tải ảnh trước!")

    def show_image(self, image, panel):
        image_tk = ImageTk.PhotoImage(image)
        panel.config(image=image_tk)
        panel.image = image_tk

    def save_image(self):
        if self.output_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                self.output_image.save(file_path)
                messagebox.showinfo("Info", "Ảnh đã được lưu thành công!")
        else:
            messagebox.showwarning("Warning", "Không có ảnh nào để lưu!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSuperResolutionApp(root)
    root.mainloop()