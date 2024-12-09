# Mô tả Cấu Trúc Dự Án

## 1. Thư Mục và Tệp Tin

### **DIV2K_train_HR** (Dataset)
- Chứa các ảnh có độ phân giải cao (HR) trong tập huấn luyện.

### **DIV2K_train_LR_bicubic** (Dataset)
- Chứa các ảnh độ phân giải thấp (LR) đã được làm mờ (bicubic) trong tập huấn luyện.

### **DIV2K_valid_HR** (Dataset)
- Chứa các ảnh có độ phân giải cao (HR) trong tập xác thực.

### **DIV2K_valid_LR_bicubic** (Dataset)
- Chứa các ảnh độ phân giải thấp (LR) đã được làm mờ (bicubic) trong tập xác thực.

### **training_plots** (Result)
- Lưu trữ các biểu đồ liên quan đến quá trình huấn luyện mô hình như loss và các chỉ số khác.

## 2. Các Tệp Python

### **Main.py** (Run)
- Tệp chính để chạy chương trình và thực hiện các bước tăng độ phân giải ảnh.

### **Training.py** (Code)
- Tệp chứa mã nguồn để huấn luyện mô hình, bao gồm các lớp và hàm cần thiết để xây dựng, huấn luyện và lưu mô hình CNN.

### **Update.py** (Code)
- Tệp có thể chứa mã để cập nhật mô hình update của Trainning.py hoặc các tính năng khác của hệ thống trong quá trình huấn luyện.

### **trained_model.h5** (Code)
- Mô hình đã được huấn luyện và lưu lại dưới định dạng `.h5` sau khi huấn luyện.

### **training_history.csv** (Code)
- Tệp CSV chứa thông tin lịch sử huấn luyện, bao gồm các giá trị loss, accuracy, precision, recall, và F1-score.

## 3. Tệp Tài Liệu và Slide

### **Nhom7-IT2.docx** (Word)
- Tài liệu báo cáo hoặc mô tả chi tiết về dự án, các bước thực hiện, kết quả, v.v.

### **Nhom7-IT2.pptx** (Slide)
- Slide thuyết trình, có thể dùng để trình bày về dự án, các kết quả, và mô hình CNN.

## 4. Cấu Trúc Tổng Quan
Dự án này bao gồm các thành phần chính:
1. **Dữ liệu** (Div2K_train_LR và HR): Dữ liệu huấn luyện và xác thực.
2. **Mô Hình CNN**: Được xây dựng và huấn luyện để tái tạo ảnh HR từ ảnh LR.
3. **Kết Quả và Biểu Đồ**: Bao gồm các chỉ số huấn luyện và các biểu đồ so sánh kết quả.
4. **Tài Liệu và Báo Cáo**: Các tài liệu và slide phục vụ việc trình bày và báo cáo kết quả dự án.



# **Cách thức huấn luyện:**

## 1. **Mô Hình CNN được huấn luyện thông qua các bước chi tiết sau:**

### A. **Chuẩn Bị Dữ Liệu:**

#### - **Dữ liệu huấn luyện (Train) và xác thực (Validation):**
  - Ảnh độ phân giải thấp (Low Resolution - LR) và độ phân giải cao (High Resolution - HR) được tải từ các thư mục.
  - Các ảnh được resize về kích thước chuẩn (128, 128) và chuẩn hóa giá trị pixel về khoảng [0, 1].

#### - **Xử lý dữ liệu:**
  - Tải ảnh từ thư mục bằng hàm `load_images()`.
  - Chuyển đổi ảnh thành mảng numpy (NumPy array) để tương thích với TensorFlow.

#### - **Phân bổ dữ liệu:**
  - Tập huấn luyện bao gồm dữ liệu từ `DIV2K_train_LR_bicubic` và `DIV2K_train_HR`.
  - Tập xác thực bao gồm dữ liệu từ `DIV2K_valid_LR_bicubic` và `DIV2K_valid_HR`.

---

### B. **Kiến Trúc và Tối Ưu Hóa Mô Hình:**

#### - **Xây dựng mô hình CNN:**
  - Mô hình CNN có cấu trúc **Encoder-Decoder**:
    - **Encoder:** Trích xuất đặc trưng bằng các tầng tích chập (Convolution), chuẩn hóa (BatchNormalization) và giảm kích thước (MaxPooling).
    - **Decoder:** Tăng kích thước ảnh qua các tầng **UpSampling** kết hợp **Convolution** để tái tạo ảnh HR từ ảnh LR.

#### - **Thông số kỹ thuật:**
  - **Optimizer:** Adam với learning rate 1e-4.
  - **Loss function:** Mean Squared Error (MSE).
  - **Metrics:** Đo lường độ chính xác (accuracy).

---

### C. **Quá Trình Huấn Luyện:**

#### - **Cấu hình callback:**
  - **TrainingProgress:** Ghi nhận tiến trình và tính toán các chỉ số như Precision, Recall, và F1-Score trên tập xác thực.
  - **EarlyStopping:** Dừng sớm nếu lỗi (loss) trên tập xác thực không giảm sau 5 epoch liên tiếp.
  - **ReduceLROnPlateau:** Giảm learning rate khi loss không cải thiện trong 3 epoch, đảm bảo hội tụ tối ưu.

#### - **Huấn luyện mô hình:**
  - Sử dụng `model.fit()` với các tham số:
    - `batch_size=32`
    - `epochs=50` (tăng số epoch nếu cần khi áp dụng trên toàn bộ tập dữ liệu lớn).

#### - **Lưu trữ kết quả:**
  - Mô hình được lưu vào file `trained_model.h5`.
  - Kết quả huấn luyện (loss, accuracy, precision, recall, f1-score) được lưu vào file `training_history.csv`.
  - Các biểu đồ loss và các chỉ số khác theo epoch được lưu vào thư mục `training_plots`.

---

## 2. **Phương Pháp Tăng Độ Phân Giải Của Ảnh:**

### A. **Tiền Xử Lý Ảnh Đầu Vào:**

- Ảnh đầu vào (LR) được resize về kích thước chuẩn (128, 128) và chuẩn hóa giá trị pixel về [0, 1].
- Định dạng đầu vào: `(batch_size, 128, 128, 3)`.

---

### B. **Quá Trình Tái Tạo Ảnh:**

#### - **Trích xuất đặc trưng:** 
  - Các tầng **Conv2D** và **BatchNormalization** trích xuất đặc trưng từ ảnh LR.

#### - **Giảm kích thước và học đặc trưng sâu (Downsampling):**
  - Sử dụng tầng **MaxPooling2D** để giảm kích thước không gian ảnh.

#### - **Tăng kích thước và tái tạo (Upsampling):**
  - Các tầng **UpSampling2D** kết hợp với **Conv2D** tăng kích thước không gian ảnh từng bước.

#### - **Tầng cuối cùng** sử dụng hàm kích hoạt **sigmoid** để giá trị pixel của ảnh HR đầu ra nằm trong khoảng [0, 1].

---

### C. **Dự Đoán Ảnh HR:**

#### - **Dự đoán:**
  - **Đầu vào:** Một ảnh LR hoặc tập ảnh LR.
  - **Đầu ra:** Ảnh HR được mô hình dự đoán.
  - Sử dụng hàm `model.predict()` để nhận ảnh HR đầu ra.

#### - **Hậu xử lý ảnh:**
  - Chuyển đổi giá trị pixel từ [0, 1] về [0, 255] nếu cần hiển thị hoặc lưu ảnh.
  - Lưu ảnh dự đoán bằng `tf.keras.preprocessing.image.save_img()`.

---

### D. **Đánh Giá Chất Lượng Tái Tạo:**

#### - **Chỉ số đánh giá:**
  - So sánh ảnh HR dự đoán với ảnh HR thực trên các chỉ số:
    - **MSE (Loss):** Đánh giá độ sai khác giữa hai ảnh.
    - **Precision, Recall, F1 Score:** Đánh giá khả năng tái tạo chi tiết trong ảnh.

#### - **Trực quan hóa:**
  - Hiển thị ảnh LR, HR thực và HR dự đoán để so sánh chất lượng.

---

## Cách Thức Hoạt Động Của Ứng Dụng

### A. Chuẩn Bị và Cấu Hình Ứng Dụng
**Môi trường:**  
Ứng dụng sử dụng các thư viện sau:
- **Tkinter**: Giao diện người dùng.
- **Pillow**: Xử lý ảnh.
- **TensorFlow**: Mô hình học sâu (CNN).

**Mô Hình CNN (Residual Network):**  
Mô hình CNN được thiết kế để tăng độ phân giải ảnh bằng cách:
- Sử dụng **Residual Block** để giữ thông tin quan trọng trong ảnh.
- Sử dụng **UpSampling2D** để phóng to kích thước ảnh đầu ra.

### B. Các Chức Năng Chính Của Ứng Dụng

1. **Tải Mô Hình Đã Huấn Luyện**  
   Người dùng nhấn "Tải Mô Hình Đã Huấn Luyện" để chọn mô hình `.h5` đã được huấn luyện trước.  
   **Hoạt động:**
   - Mô hình được tải bằng `tf.keras.models.load_model`.
   - Mô hình được lưu trữ trong biến `self.model` và sẵn sàng để xử lý ảnh.

2. **Tải Ảnh Đầu Vào**  
   Người dùng nhấn "Tải Ảnh" để chọn ảnh từ máy tính.  
   **Hoạt động:**
   - Ảnh gốc được đọc bằng thư viện Pillow.
   - Ảnh được làm mờ với bộ lọc **GaussianBlur**.
   - Ảnh làm mờ được hiển thị trong khung **Ảnh Đầu Vào**.

3. **Tăng Độ Phân Giải và Độ Nét Ảnh**  
   Người dùng nhấn "Tăng Độ Nét & Độ Phân Giải" để cải thiện ảnh.  
   **Hoạt động:**
   - Nếu không có mô hình đã tải, ứng dụng sử dụng **Pillow** để tăng độ nét bằng `ImageEnhance.Sharpness` và bộ lọc **DETAIL**.
   - Nếu có mô hình đã tải, ảnh đầu vào được chuyển thành tensor (numpy array) và chuẩn hóa, sau đó được xử lý qua mô hình CNN để dự đoán ảnh chất lượng cao hơn.
   - Ảnh kết quả được hiển thị trong khung **Ảnh Đã Xử Lý**.

4. **Hiển Thị và Lưu Ảnh**  
   Ứng dụng hiển thị cả hai ảnh: ảnh đầu vào và ảnh đã qua xử lý. Độ phân giải ảnh được hiển thị dưới từng khung hình.  
   Người dùng có thể nhấn "Lưu Ảnh" để lưu ảnh đã xử lý dưới dạng tệp `.png`.

### C. Các Chức Năng Phụ Trợ

1. **Tải Lịch Sử Huấn Luyện**  
   Người dùng nhấn "Tải Lịch Sử Huấn Luyện" để chọn tệp `.csv` chứa thông tin loss và accuracy của quá trình huấn luyện.  
   **Hoạt động:**
   - Tệp `.csv` được đọc bằng pandas và lưu trong biến `self.history_data`.

2. **Vẽ Biểu Đồ Lịch Sử Huấn Luyện**  
   Sau khi tải lịch sử, người dùng nhấn "Vẽ Biểu Đồ Lịch Sử".  
   **Hoạt động:**
   - Ứng dụng sử dụng **matplotlib** để vẽ biểu đồ về:
     - **Loss** (mất mát) qua các epoch.
     - **Accuracy** (độ chính xác) qua các epoch.
   - Biểu đồ được hiển thị trong cửa sổ mới.

### Quy Trình Xử Lý Ảnh

1. **Chuẩn Bị Ảnh Đầu Vào:**
   - Nếu không có mô hình, ảnh được làm mờ để giảm chất lượng giả lập.
   - Nếu có mô hình, ảnh được resize về kích thước phù hợp với mô hình CNN.

2. **Xử Lý Bằng CNN:**
   - Ảnh đầu vào được đưa qua các tầng tích chập (**Convolutional Layers**).
   - Các **Residual Block** giúp bảo toàn chi tiết ảnh gốc.
   - **UpSampling2D** giúp tăng kích thước ảnh đầu ra.

3. **Tăng Cường Chi Tiết Ảnh:**
   - Nếu không sử dụng mô hình, ảnh được cải thiện độ sắc nét bằng **Pillow**.

4. **Hiển Thị Ảnh Đã Xử Lý:**
   - Ứng dụng hiển thị ảnh kết quả cùng với độ phân giải.

---

## Kết Quả Thực Nghiệm


**Mô Tả Kết Quả:**  
Kết quả thực nghiệm được đánh giá dựa trên khả năng của ứng dụng trong việc tăng độ phân giải và cải thiện độ sắc nét của ảnh. Các kết quả được ghi nhận qua hai trường hợp chính:

#### 1. Trường Hợp 1: Xử Lý Ảnh Bằng Các Phương Pháp Cơ Bản (Không Dùng Mô Hình CNN)

- **Ảnh Đầu Vào:**  
  Loại ảnh: Phong cảnh và chân dung, độ phân giải thấp (dưới 128x128).
  
- **Quy Trình Xử Lý:**  
  1. Ảnh được làm mờ bằng bộ lọc **Gaussian Blur** (giảm chất lượng).
  2. Ảnh được tăng cường bằng bộ lọc `ImageEnhance.Sharpness` và bộ lọc **DETAIL**.
  
- **Kết Quả:**
  - **Cải thiện độ sắc nét:**  
    Ảnh rõ nét hơn so với ảnh đầu vào mờ. Một số chi tiết nhỏ được làm nổi bật.
  - **Độ phân giải không thay đổi:**  
    Ảnh không được phóng to hay cải thiện độ phân giải thực tế.
    
- **Ưu điểm:**
  - Xử lý nhanh, không yêu cầu mô hình phức tạp.
  - Phù hợp với các ảnh nhỏ hoặc yêu cầu làm rõ chi tiết nhanh chóng.
  
- **Hạn chế:**
  - Không cải thiện độ phân giải thực sự.
  - Chất lượng ảnh chỉ cải thiện nhẹ, phụ thuộc vào bộ lọc.

#### 2. Trường Hợp 2: Xử Lý Ảnh Bằng Mô Hình CNN (Khi Tải Mô Hình Huấn Luyện Trước)

- **Ảnh Đầu Vào:**  
  Loại ảnh: Ảnh chất lượng thấp (64x64, 128x128) với chi tiết bị mờ.
  
- **Quy Trình Xử Lý:**  
  1. Ảnh được resize về kích thước cố định (128x128).
  2. Ảnh được xử lý qua mô hình CNN đã huấn luyện trên tập dữ liệu ảnh chất lượng cao (tăng độ phân giải gấp 2 lần hoặc hơn).
  
- **Kết Quả:**
  - **Cải thiện độ phân giải:**  
    Ảnh đầu ra có độ phân giải cao hơn, ví dụ: từ 128x128 lên 256x256. Các chi tiết như cạnh, kết cấu và màu sắc được tái tạo rõ ràng.
  - **Cải thiện độ sắc nét:**  
    Ảnh đã xử lý rõ ràng và sắc nét hơn so với ảnh gốc.
  - **Độ mượt của ảnh:**  
    Mô hình CNN tái tạo ảnh mềm mại, không gây nhiễu hạt như các phương pháp thông thường.
  
- **Ưu điểm:**
  - Cải thiện cả độ phân giải lẫn độ sắc nét.
  - Kết quả gần với ảnh gốc chất lượng cao (tùy thuộc vào mức độ huấn luyện mô hình).
  
- **Hạn chế:**
  - Yêu cầu mô hình đã được huấn luyện tốt trên tập dữ liệu tương tự.
  - Thời gian xử lý ảnh lâu hơn so với các phương pháp cơ bản.
