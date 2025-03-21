# Xử lý và Tiền xử lý Dữ liệu: Đánh giá Nguy cơ Bệnh Tim bằng AI  

## Giới thiệu  

Notebook này tập trung vào nhiệm vụ quan trọng là **xử lý dữ liệu**, một bước nền tảng trong việc chuẩn bị tập dữ liệu cho phân tích và mô hình hóa. Quá trình này bao gồm việc chuyển đổi và ánh xạ dữ liệu thô sang định dạng phù hợp để phân tích, đảm bảo chất lượng, tính nhất quán và khả năng sử dụng của dữ liệu. Các bước thực hiện trong notebook này bao gồm:  

### Các bước chính trong Xử lý và Tiền xử lý Dữ liệu  

- **Thay thế các giá trị cụ thể bằng NaN**: Chuẩn hóa dữ liệu bị thiếu bằng cách thay thế các giá trị lỗi hoặc giá trị giữ chỗ bằng `NaN`.  
- **Tính toán phân bố giá trị**: Phân tích phân bố dữ liệu hiện có để hiểu trạng thái ban đầu của dữ liệu.  
- **Điền giá trị bị thiếu**: Sử dụng phân bố dữ liệu để điền giá trị bị thiếu, giúp duy trì tính toàn vẹn của tập dữ liệu.  
- **Ánh xạ giá trị danh mục**: Chuyển đổi mã số thành nhãn danh mục có ý nghĩa để dễ dàng diễn giải hơn.  
- **Đổi tên cột**: Cập nhật tên cột để phản ánh chính xác nội dung của chúng, giúp tập dữ liệu dễ đọc và sử dụng hơn.  

## Các bước thực hiện  

1. **Xử lý dữ liệu bị thiếu**:  
    - Xác định các giá trị bị thiếu trong tập dữ liệu.  
    - Thay thế các giá trị cụ thể (ví dụ: 7 và 9) bằng `NaN` để chuẩn hóa dữ liệu bị thiếu.  
    - Tính toán phân bố các giá trị hiện có để hiểu trạng thái ban đầu của dữ liệu.  
    - Điền giá trị bị thiếu dựa trên phân bố dữ liệu nhằm đảm bảo dữ liệu vẫn đại diện đúng cho đặc điểm ban đầu.  

2. **Ánh xạ dữ liệu**:  
    - Tạo một từ điển ánh xạ để chuyển đổi mã số thành nhãn danh mục có ý nghĩa (ví dụ: ánh xạ giới tính `{1: 'Nam', 2: 'Nữ', 3: 'Phi nhị nguyên'}`).  
    - Áp dụng ánh xạ cho các cột liên quan.  

3. **Đổi tên cột**:  
    - Cập nhật tên cột để phản ánh chính xác nội dung, giúp tập dữ liệu rõ ràng và dễ sử dụng hơn.  

4. **Tạo đặc trưng mới (Feature Engineering)**:  
    - Tạo các đặc trưng mới từ dữ liệu có sẵn để nâng cao sức mạnh dự đoán và giá trị phân tích của tập dữ liệu.  

## Kết luận  

Notebook này đảm bảo rằng tập dữ liệu sạch, nhất quán và sẵn sàng cho các bước phân tích tiếp theo cũng như các bài toán học máy. Bằng cách áp dụng có hệ thống các bước trên, chúng ta nâng cao chất lượng dữ liệu, giúp nó phù hợp hơn cho việc xây dựng các mô hình dự đoán chính xác và đáng tin cậy.