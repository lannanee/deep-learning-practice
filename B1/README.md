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

\[
Output=f\left(\sum_{i=1}^{n}x_iw_i+b\right)
\]

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

## 2. Quiz ôn tập

**Hướng dẫn:** Tự trả lời các câu hỏi trước khi mở phần đáp án. Câu hỏi có ghi *Chọn nhiều đáp án* yêu cầu chọn tất cả phương án đúng.

### Câu 1. Mối quan hệ giữa AI, ML và DL

Theo bài học, mối quan hệ đúng giữa ba khái niệm AI, Machine Learning (ML) và Deep Learning (DL) là gì?

- [ ] A. AI là tập con của ML, ML là tập con của DL.
- [ ] B. AI, ML và DL là ba lĩnh vực hoàn toàn độc lập.
- [ ] C. DL là tập con của ML, ML là tập con của AI.
- [ ] D. ML là tập con của DL, DL là tập con của AI.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: C.**

Deep Learning thuộc Machine Learning, còn Machine Learning thuộc lĩnh vực rộng hơn là Artificial Intelligence.

</details>

### Câu 2. AI yếu và AI tổng quát

AI yếu (Narrow AI) khác với AI tổng quát (General AI) ở điểm nào?

- [ ] A. AI yếu thực hiện nhiệm vụ chuyên biệt; AI tổng quát hướng tới khả năng thực hiện nhiều nhiệm vụ trí tuệ khác nhau.
- [ ] B. AI yếu luôn thông minh hơn con người trong mọi lĩnh vực.
- [ ] C. AI tổng quát chỉ thực hiện một nhiệm vụ duy nhất.
- [ ] D. AI yếu không sử dụng dữ liệu.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: A.**

AI yếu tập trung vào những nhiệm vụ cụ thể. AI tổng quát hướng tới khả năng học hỏi và giải quyết nhiều loại nhiệm vụ trí tuệ, nhưng vẫn chưa đạt được theo nghĩa tổng quát thực sự.

</details>

### Câu 3. Nguyên lý cốt lõi của Machine Learning

Nguyên lý cốt lõi của Machine Learning là gì?

- [ ] A. Viết thủ công tất cả quy tắc cho mọi trường hợp.
- [ ] B. Học các mẫu và mối quan hệ từ dữ liệu để dự đoán hoặc đưa ra quyết định.
- [ ] C. Chỉ lưu trữ dữ liệu mà không xử lý.
- [ ] D. Chỉ sử dụng các phép tính số học đơn giản.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: B.**

Machine Learning tìm ra quy luật từ dữ liệu. Tuy nhiên, mô hình vẫn cần được thiết kế, huấn luyện và đánh giá phù hợp; không có nghĩa là hoàn toàn không cần lập trình.

</details>

### Câu 4. Thành phần Bias trong Perceptron

Cho công thức:

`Output = f(Σ(xᵢ × wᵢ) + b)`

Thành phần `b` có ý nghĩa gì?

- [ ] A. Dữ liệu đầu vào.
- [ ] B. Trọng số của đầu vào.
- [ ] C. Độ lệch (Bias), giúp điều chỉnh giá trị trước hàm kích hoạt.
- [ ] D. Số lượng lớp trong mạng nơ-ron.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: C.**

Bias là một tham số được cộng vào tổng có trọng số, giúp mô hình linh hoạt hơn trong việc học mối quan hệ giữa đầu vào và đầu ra.

</details>

### Câu 5. Mạng nơ-ron truyền thẳng

Vì sao mạng nơ-ron truyền thẳng được gọi là *Feedforward Neural Network*?

- [ ] A. Vì dữ liệu chỉ truyền từ đầu ra về đầu vào.
- [ ] B. Vì dữ liệu truyền qua các lớp theo chiều từ đầu vào đến đầu ra, không có vòng lặp phản hồi trong đường truyền.
- [ ] C. Vì mạng không cần dữ liệu đầu vào.
- [ ] D. Vì mạng chỉ có một nơ-ron duy nhất.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: B.**

Trong quá trình truyền thẳng, dữ liệu đi qua các lớp theo một chiều từ đầu vào đến đầu ra. Điều này khác với các mạng có kết nối hồi tiếp.

</details>

### Câu 6. Số nơ-ron ở lớp đầu ra

Với bài toán phân loại đa lớp có `n` lớp, một thiết kế phổ biến là sử dụng bao nhiêu nơ-ron ở lớp đầu ra?

- [ ] A. Luôn luôn là 1.
- [ ] B. `n - 1` nơ-ron.
- [ ] C. `2n` nơ-ron.
- [ ] D. `n` nơ-ron.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: D.**

Một thiết kế phổ biến sử dụng `n` nơ-ron đầu ra, tương ứng với `n` lớp cần phân loại. Tuy nhiên, một số phương pháp mã hóa hoặc thiết kế mô hình khác có thể sử dụng cấu trúc đầu ra khác.

</details>

### Câu 7. Ứng dụng Deep Learning trong y tế

**Chọn nhiều đáp án.** Những ứng dụng nào sau đây phù hợp với Deep Learning trong y tế?

- [ ] A. Phân tích hình ảnh X-ray, CT/MRI, võng mạc và da liễu.
- [ ] B. Hỗ trợ khám phá thuốc, thiết kế phân tử và tái định vị thuốc.
- [ ] C. Hỗ trợ y học cá nhân hóa dựa trên thông tin di truyền, lối sống, môi trường và tiền sử bệnh.
- [ ] D. Đảm bảo mọi bệnh nhân đều được chữa khỏi hoàn toàn.
- [ ] E. Thay thế hoàn toàn bác sĩ trong mọi quyết định điều trị.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: A, B, C.**

Deep Learning có nhiều ứng dụng trong chẩn đoán hình ảnh, nghiên cứu thuốc và y học cá nhân hóa. Tuy nhiên, công nghệ này không đảm bảo chữa khỏi mọi bệnh và không mặc nhiên thay thế bác sĩ.

</details>

### Câu 8. Thành phần của Perceptron

**Chọn nhiều đáp án.** Những thành phần nào thuộc cấu tạo cơ bản của một nơ-ron nhân tạo?

- [ ] A. Inputs – dữ liệu đầu vào.
- [ ] B. Weights – trọng số.
- [ ] C. Bias – độ lệch.
- [ ] D. Activation Function – hàm kích hoạt.
- [ ] E. Database Server – máy chủ cơ sở dữ liệu.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: A, B, C, D.**

Các thành phần cơ bản gồm đầu vào, trọng số, bias và hàm kích hoạt. Máy chủ cơ sở dữ liệu không phải thành phần cấu tạo của một Perceptron.

</details>

### Câu 9. Ba loại chính của Machine Learning

**Chọn nhiều đáp án.** Phát biểu nào sau đây là đúng?

- [ ] A. Supervised Learning học từ dữ liệu có nhãn, ví dụ phân loại email spam.
- [ ] B. Unsupervised Learning tìm cấu trúc hoặc mẫu trong dữ liệu không nhãn, ví dụ phân nhóm khách hàng.
- [ ] C. Reinforcement Learning học thông qua tương tác với môi trường và phản hồi, ví dụ AI chơi game.
- [ ] D. Cả ba phương pháp đều bắt buộc phải sử dụng dữ liệu có nhãn.

<details>
<summary>💡 Xem đáp án và giải thích</summary>

**Đáp án: A, B, C.**

Ba phương pháp khác nhau chủ yếu ở cách học: từ dữ liệu có nhãn, từ dữ liệu không nhãn hoặc từ tương tác và phần thưởng. Phương án D không đúng.

</details>

### Câu 10. Cấu trúc ba lớp của mạng nơ-ron (3 điểm)

**Yêu cầu:** Trình bày cấu trúc ba lớp cơ bản của một mạng nơ-ron gồm Input Layer, Hidden Layer và Output Layer. Nêu vai trò của từng lớp.

<details>
<summary>💡 Xem đáp án tham khảo</summary>

Một mạng nơ-ron cơ bản có thể gồm ba lớp chính:

1. **Input Layer (Lớp đầu vào):** Tiếp nhận dữ liệu từ bên ngoài. Mỗi nơ-ron đầu vào thường tương ứng với một đặc trưng của dữ liệu, chẳng hạn tuổi, cân nặng hoặc chiều cao trong một bài toán dự đoán sức khỏe.

2. **Hidden Layer (Lớp ẩn):** Thực hiện các phép tính dựa trên trọng số (weights), độ lệch (bias) và hàm kích hoạt (activation function). Lớp này giúp mạng học các đặc trưng và mối quan hệ phức tạp từ dữ liệu. Mạng có thể có một hoặc nhiều lớp ẩn.

3. **Output Layer (Lớp đầu ra):** Nhận kết quả xử lý từ các lớp trước và tạo ra kết quả cuối cùng, chẳng hạn nhãn phân loại, xác suất thuộc từng lớp hoặc giá trị dự đoán.

**Kết luận:** Dữ liệu được tiếp nhận tại lớp đầu vào, xử lý qua các lớp ẩn và tạo kết quả tại lớp đầu ra.

</details>

---

**Tài liệu học tập:** [Machine Learning cơ bản – Bài 35: Lược sử Deep Learning](https://machinelearningcoban.com/2018/06/22/deeplearning/)
