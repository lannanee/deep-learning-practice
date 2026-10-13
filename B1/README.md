# 📘 Bài 35: Lược sử Deep Learning

> **Chủ đề:** Lịch sử phát triển và các khái niệm cơ bản về Deep Learning  
> **Tài liệu tham khảo:** [Bài 35 – Lược sử Deep Learning](https://machinelearningcoban.com/2018/06/22/deeplearning/)  
> **Mục đích:** Tổng hợp kiến thức trọng tâm và tự kiểm tra mức độ hiểu bài.

---

## 1. Tổng quan bài học

Bài học giới thiệu mối quan hệ giữa Trí tuệ nhân tạo (AI), Học máy (Machine Learning) và Học sâu (Deep Learning), đồng thời trình bày các khái niệm nền tảng của mạng nơ-ron nhân tạo.

### 1.1. Mối quan hệ giữa AI, ML và DL

```text
Artificial Intelligence (AI)
            |
            v
   Machine Learning (ML)
            |
            v
      Deep Learning (DL)
```

- **Artificial Intelligence (AI) – Trí tuệ nhân tạo:** Lĩnh vực nghiên cứu và phát triển các hệ thống có khả năng thực hiện những nhiệm vụ thường đòi hỏi trí thông minh của con người.
- **Machine Learning (ML) – Học máy:** Nhánh của AI cho phép máy tính học các quy luật từ dữ liệu để dự đoán hoặc đưa ra quyết định mà không cần lập trình tường minh mọi quy tắc.
- **Deep Learning (DL) – Học sâu:** Nhánh của ML sử dụng mạng nơ-ron nhân tạo với nhiều lớp để học các đặc trưng và mối quan hệ phức tạp từ dữ liệu.

**Ghi nhớ:** AI -> ML -> DL

### 1.2. AI yếu và AI mạnh

| Khái niệm | Đặc điểm |
|---|---|
| Narrow AI (AI yếu) | Được thiết kế để thực hiện một nhiệm vụ hoặc một nhóm nhiệm vụ cụ thể, chẳng hạn nhận diện khuôn mặt hoặc dịch ngôn ngữ. |
| General AI (AI tổng quát/AI mạnh) | Hướng tới khả năng học hỏi, suy luận và thực hiện nhiều nhiệm vụ trí tuệ khác nhau tương tự con người. Đây vẫn là mục tiêu nghiên cứu, chưa đạt được theo nghĩa AI tổng quát thực sự. |

### 1.3. Ba loại Machine Learning

| Loại học | Đặc điểm | Ví dụ |
|---|---|---|
| Supervised Learning (Học có giám sát) | Học từ dữ liệu có nhãn. | Phân loại email spam. |
| Unsupervised Learning (Học không giám sát) | Tìm kiếm cấu trúc hoặc mẫu trong dữ liệu không có nhãn. | Phân nhóm khách hàng. |
| Reinforcement Learning (Học tăng cường) | Học thông qua tương tác với môi trường và phản hồi dưới dạng phần thưởng hoặc hình phạt. | AI chơi game, robot tự hành. |

### 1.4. Perceptron – Nơ-ron nhân tạo

Một nơ-ron nhân tạo cơ bản nhận dữ liệu đầu vào, kết hợp với trọng số và độ lệch, sau đó đưa kết quả qua hàm kích hoạt.

Công thức:

Output = f(Σ(xᵢ × wᵢ) + b)

Trong đó:

- `xᵢ` (Inputs): Các giá trị dữ liệu đầu vào.
- `wᵢ` (Weights): Trọng số thể hiện mức độ ảnh hưởng của từng đầu vào.
- `b` (Bias): Độ lệch, giúp điều chỉnh giá trị trước khi đưa qua hàm kích hoạt.
- `f` (Activation Function): Hàm kích hoạt, ví dụ Sigmoid, ReLU hoặc Tanh.
- `Output`: Giá trị đầu ra của nơ-ron.

### 1.5. Cấu trúc mạng nơ-ron cơ bản

Một mạng nơ-ron truyền thẳng (Feedforward Neural Network) có thể gồm ba loại lớp chính:

1. **Input Layer – Lớp đầu vào:** Tiếp nhận dữ liệu. Các nơ-ron đầu vào thường tương ứng với những đặc trưng của dữ liệu.
2. **Hidden Layer – Lớp ẩn:** Xử lý dữ liệu thông qua trọng số, bias và hàm kích hoạt để học các đặc trưng và mối quan hệ.
3. **Output Layer – Lớp đầu ra:** Tạo kết quả cuối cùng, chẳng hạn xác suất thuộc từng lớp trong bài toán phân loại.

Mạng được gọi là *truyền thẳng* vì thông tin truyền từ đầu vào qua các lớp xử lý đến đầu ra, không có vòng lặp phản hồi trong đường truyền tính toán của mạng.

Với bài toán phân loại đa lớp có `n` lớp, một thiết kế phổ biến là sử dụng `n` nơ-ron đầu ra, mỗi nơ-ron tương ứng với một lớp.

### 1.6. Ứng dụng Deep Learning trong y tế

Deep Learning có thể hỗ trợ nhiều lĩnh vực y tế:

- **Chẩn đoán hình ảnh:** Phân tích X-ray, CT, MRI, hình ảnh võng mạc và hình ảnh da liễu.
- **Phát triển thuốc:** Hỗ trợ khám phá thuốc, thiết kế phân tử và tái định vị thuốc.
- **Y học cá nhân hóa:** Hỗ trợ đưa ra dự đoán hoặc khuyến nghị dựa trên thông tin di truyền, lối sống, môi trường và tiền sử bệnh.

Các ứng dụng này hỗ trợ chuyên gia y tế chứ không mặc nhiên thay thế quyết định chuyên môn của bác sĩ.

---

## 2. Câu hỏi ôn tập và đáp án

### Câu 1. Mối quan hệ giữa AI, ML và DL

**Câu hỏi:** Mối quan hệ giữa ba khái niệm AI, Machine Learning (ML) và Deep Learning (DL) là gì?

**Trả lời:** Deep Learning là tập con của Machine Learning, còn Machine Learning là tập con của Artificial Intelligence. Có thể biểu diễn mối quan hệ là **AI → ML → DL**.

### Câu 2. AI yếu và AI tổng quát

**Câu hỏi:** AI yếu (Narrow AI) khác với AI tổng quát (General AI) ở điểm nào?

**Trả lời:** AI yếu được thiết kế để thực hiện một nhiệm vụ hoặc một nhóm nhiệm vụ cụ thể. AI tổng quát hướng tới khả năng học hỏi, suy luận và thực hiện nhiều nhiệm vụ trí tuệ khác nhau tương tự con người; hiện vẫn chưa đạt được theo nghĩa AI tổng quát thực sự.

### Câu 3. Nguyên lý cốt lõi của Machine Learning

**Câu hỏi:** Nguyên lý cốt lõi của Machine Learning là gì?

**Trả lời:** Machine Learning học các quy luật, mẫu và mối quan hệ từ dữ liệu để dự đoán hoặc đưa ra quyết định mà không cần lập trình tường minh mọi quy tắc.

### Câu 4. Thành phần Bias trong Perceptron

**Câu hỏi:** Trong công thức `Output = f(Σ(xᵢ × wᵢ) + b)`, thành phần `b` là gì?

**Trả lời:** `b` là Bias (độ lệch), một tham số được cộng vào tổng có trọng số để điều chỉnh giá trị trước khi đưa qua hàm kích hoạt.

### Câu 5. Mạng nơ-ron truyền thẳng

**Câu hỏi:** Vì sao mạng nơ-ron truyền thẳng (Feedforward Neural Network) được gọi là truyền thẳng?

**Trả lời:** Vì dữ liệu truyền từ lớp đầu vào qua các lớp xử lý đến lớp đầu ra theo một chiều, không có vòng lặp phản hồi trong đường truyền tính toán của mạng.

### Câu 6. Số nơ-ron ở lớp đầu ra

**Câu hỏi:** Với bài toán phân loại đa lớp có `n` lớp, số nơ-ron ở lớp đầu ra theo thiết kế phổ biến là bao nhiêu?

**Trả lời:** Có `n` nơ-ron đầu ra, mỗi nơ-ron tương ứng với một lớp cần phân loại.

### Câu 7. Ứng dụng Deep Learning trong y tế

**Câu hỏi:** Những ứng dụng nào của Deep Learning được đề cập trong lĩnh vực y tế?

**Trả lời:**
- Chẩn đoán hình ảnh y tế: X-ray, CT/MRI, hình ảnh võng mạc và da liễu.
- Phát triển thuốc: Khám phá thuốc, thiết kế phân tử và tái định vị thuốc.
- Y học cá nhân hóa: Hỗ trợ dự đoán hoặc khuyến nghị dựa trên thông tin di truyền, lối sống, môi trường và tiền sử bệnh.

### Câu 8. Thành phần của Perceptron

**Câu hỏi:** Những thành phần cơ bản nào cấu tạo nên một nơ-ron nhân tạo (Perceptron)?

**Trả lời:**
- **Inputs:** Các giá trị dữ liệu đầu vào.
- **Weights:** Trọng số thể hiện mức độ ảnh hưởng của từng đầu vào.
- **Bias:** Độ lệch giúp điều chỉnh giá trị trước hàm kích hoạt.
- **Activation Function:** Hàm kích hoạt, ví dụ Sigmoid, ReLU hoặc Tanh.

Công thức tổng quát:

`Output = f(Σ(xᵢ × wᵢ) + b)`

### Câu 9. Ba loại chính của Machine Learning

**Câu hỏi:** Trình bày ba loại chính của Machine Learning và ví dụ minh họa.

**Trả lời:**
- **Supervised Learning (Học có giám sát):** Học từ dữ liệu có nhãn. Ví dụ: phân loại email spam.
- **Unsupervised Learning (Học không giám sát):** Tìm kiếm cấu trúc hoặc mẫu trong dữ liệu không nhãn. Ví dụ: phân nhóm khách hàng.
- **Reinforcement Learning (Học tăng cường):** Học thông qua tương tác với môi trường và phản hồi dưới dạng phần thưởng hoặc hình phạt. Ví dụ: AI chơi game hoặc robot tự hành.

### Câu 10. Cấu trúc ba lớp của mạng nơ-ron (3 điểm)

**Câu hỏi:** Trình bày cấu trúc ba lớp cơ bản của một mạng nơ-ron gồm Input Layer, Hidden Layer và Output Layer. Nêu vai trò của từng lớp.

**Trả lời:**

Một mạng nơ-ron cơ bản có thể gồm ba lớp chính:

1. **Input Layer (Lớp đầu vào):** Tiếp nhận dữ liệu từ bên ngoài. Mỗi nơ-ron đầu vào thường tương ứng với một đặc trưng của dữ liệu.

2. **Hidden Layer (Lớp ẩn):** Thực hiện các phép tính dựa trên trọng số (weights), độ lệch (bias) và hàm kích hoạt (activation function). Lớp này giúp mạng học các đặc trưng và mối quan hệ phức tạp trong dữ liệu. Mạng có thể có một hoặc nhiều lớp ẩn.

3. **Output Layer (Lớp đầu ra):** Nhận kết quả xử lý từ các lớp trước và tạo ra kết quả cuối cùng, chẳng hạn nhãn phân loại, xác suất thuộc từng lớp hoặc giá trị dự đoán.

**Kết luận:** Dữ liệu được tiếp nhận tại lớp đầu vào, xử lý qua các lớp ẩn và tạo kết quả tại lớp đầu ra.


---

**Tài liệu học tập:** [Machine Learning cơ bản – Bài 35: Lược sử Deep Learning](https://machinelearningcoban.com/2018/06/22/deeplearning/)
