# **Phân tích dữ liệu khám phá: Đánh giá nguy cơ mắc bệnh tim bằng AI**

## **Giới thiệu**

Chào mừng bạn đến với sổ tay phân tích dữ liệu khám phá (EDA) cho dự án dự đoán bệnh tim. Đây là một bước quan trọng trong quy trình khoa học dữ liệu, giúp khám phá các thông tin và mô hình trong tập dữ liệu nhằm hướng dẫn quá trình xây dựng mô hình dự đoán.

Trong sổ tay này, chúng ta sẽ thực hiện:

* **Xác thực tập dữ liệu:** Đảm bảo dữ liệu sạch, nhất quán và sẵn sàng cho phân tích.  
* **Khám phá phân phối đặc trưng:** Phân tích cách các đặc trưng phân phối liên quan đến bệnh tim.  
* **Chuyển đổi dữ liệu phân loại:** Biến đổi các đặc trưng phân loại thành dạng số bằng bộ mã hóa CatBoost để phân tích và xây dựng mô hình tốt hơn.  
* **Phân tích tương quan:** Xem xét mối quan hệ tuyến tính và phi tuyến giữa các đặc trưng với biến mục tiêu (bệnh tim) bằng hệ số tương quan Pearson và thông tin tương hỗ (Mutual Information).  
* **Lựa chọn đặc trưng:** Xác định và chọn ra các đặc trưng có ảnh hưởng cao nhất đến nguy cơ mắc bệnh tim.  

Những bước này giúp chúng ta hiểu rõ hơn về dữ liệu, khám phá mối quan hệ quan trọng và chuẩn bị dữ liệu để xây dựng mô hình dự đoán chính xác.

---

## **Tập dữ liệu**

Tập dữ liệu trong phân tích này đã trải qua quá trình xử lý dữ liệu kỹ lưỡng. Quá trình xử lý dữ liệu đóng vai trò quan trọng trong khoa học dữ liệu, giúp biến đổi và chuẩn bị dữ liệu thô thành dạng phù hợp hơn. Các bước chính bao gồm:

* Xử lý giá trị thiếu  
* Ánh xạ dữ liệu  
* Làm sạch dữ liệu  
* Kỹ thuật đặc trưng (Feature Engineering)  

Những bước này đảm bảo rằng tập dữ liệu sẵn sàng để phân tích và xây dựng mô hình dự đoán bệnh tim đáng tin cậy.


## **Các đặc trưng liên quan đến bệnh tim**

Sau nhiều ngày nghiên cứu và phân tích tập dữ liệu, chúng tôi đã xác định các đặc trưng quan trọng trong đánh giá nguy cơ mắc bệnh tim:

* **Biến mục tiêu (Biến phụ thuộc):**  
    * **Heart_disease:** Đã từng được chẩn đoán mắc chứng đau thắt ngực hoặc bệnh động mạch vành.  

* **Nhân khẩu học:**  
    * **Gender:** Giới tính (Nam/Nữ).  
    * **Race:** Nhóm chủng tộc (theo bảng dữ liệu tính toán).  
    * **Age:** Tuổi (tính theo nhóm, trên 80 tuổi được gom chung).  

* **Tiền sử bệnh lý:**  
    * Tình trạng sức khỏe chung.  
    * Có bác sĩ chăm sóc sức khỏe cá nhân hay không.  
    * Có từng không đủ khả năng tài chính để gặp bác sĩ hay không.  
    * Thời gian kể từ lần kiểm tra sức khỏe định kỳ gần nhất.  
    * Đã từng được chẩn đoán đau tim.  
    * Đã từng bị đột quỵ.  
    * Đã từng được chẩn đoán mắc chứng rối loạn trầm cảm.  
    * Đã từng được chẩn đoán mắc bệnh thận.  
    * Đã từng được chẩn đoán mắc bệnh tiểu đường.  
    * Cân nặng (pound).  
    * Chiều cao (feet và inches).  
    * Chỉ số khối cơ thể (BMI).  
    * Khó khăn trong việc đi bộ hoặc leo cầu thang.  
    * Tình trạng sức khỏe thể chất (tính toán).  
    * Tình trạng sức khỏe tâm thần (tính toán).  
    * Tình trạng hen suyễn (tính toán).  

* **Lối sống:**  
    * Hoạt động thể chất trong thời gian rảnh rỗi.  
    * Đã từng hút ít nhất 100 điếu thuốc.  
    * Trạng thái hút thuốc (tính toán).  
    * Uống rượu bia quá mức (theo biến số tính toán).  
    * Số lượng đồ uống có cồn tiêu thụ mỗi tuần.  
    * Đã từng tập thể dục trong 30 ngày qua.  
    * Thời gian ngủ trung bình mỗi ngày.  

---

## **Chuyển đổi kiểu dữ liệu đặc trưng**

Trong **pandas**, kiểu dữ liệu `object` thường được sử dụng cho dữ liệu văn bản hoặc hỗn hợp. Khi một cột chứa dữ liệu phân loại, việc chuyển đổi sang kiểu **category** sẽ mang lại nhiều lợi ích:

**Lợi ích của việc chuyển đổi sang kiểu dữ liệu phân loại:**  
* **Tiết kiệm bộ nhớ:** Pandas lưu trữ các giá trị dưới dạng mã số thay vì chuỗi văn bản, giúp tối ưu bộ nhớ.  
* **Tăng hiệu suất:** Các thao tác trên dữ liệu phân loại nhanh hơn so với xử lý dữ liệu chuỗi thông thường.  
* **Định nghĩa rõ ràng:** Việc chuyển đổi giúp biểu thị rõ ràng dữ liệu phân loại, tránh hiểu nhầm là dữ liệu liên tục.  

---

## **Phân tích phân phối đặc trưng phân loại so với biến mục tiêu**

Trong phân tích dữ liệu, việc hiểu cách các đặc trưng phân loại phân bố theo biến mục tiêu giúp khám phá các xu hướng quan trọng. Một trong những phương pháp hữu ích là **biểu đồ cột ngang xếp chồng (horizontal stacked bar chart)**. Biểu đồ này giúp quan sát tỷ lệ giữa các nhóm trong một đặc trưng so với biến mục tiêu, làm rõ mối quan hệ giữa các yếu tố.

---

### **Bệnh tim: Biến mục tiêu**
#### **Phân tích phân phối**
* Có sự mất cân bằng đáng kể giữa hai nhóm.  
* **Phần lớn** cá nhân **không mắc bệnh tim** (`418.3K`), trong khi chỉ có một số ít **bị bệnh tim** (`26.8K`).  
* Sự mất cân bằng này được thể hiện rõ trong biểu đồ, với thanh màu xanh lá dài hơn rất nhiều so với thanh màu đỏ.  

---

### **Vấn đề mất cân bằng dữ liệu**
1. **Thiên lệch mô hình:** Khi huấn luyện mô hình phân loại trên tập dữ liệu này, mô hình có thể bị thiên lệch, chủ yếu dự đoán nhóm có số lượng lớn hơn (không mắc bệnh tim).  
2. **Chỉ số đánh giá sai lệch:** Độ chính xác (Accuracy) có thể gây hiểu nhầm. Ví dụ, một mô hình luôn dự đoán "không mắc bệnh tim" có thể có độ chính xác cao nhưng lại không thực sự hữu ích.  
3. **Độ nhạy (Recall) và Độ chính xác (Precision):**  
   * **Recall (Độ nhạy):** Đo lường khả năng phát hiện đúng các trường hợp mắc bệnh tim.  
   * **Precision (Độ chính xác):** Đánh giá mức độ chính xác khi mô hình dự đoán ai mắc bệnh tim.  
   * Nếu tập dữ liệu mất cân bằng, mô hình có thể có **recall thấp**, tức là không xác định chính xác những người thực sự mắc bệnh tim.  
### **Chiến lược Giải quyết Mất Cân Bằng**  
Bộ phân loại `BalancedRandomForestClassifier` từ thư viện `imbalanced-ensemble` xử lý hiệu quả vấn đề mất cân bằng lớp bằng cách sử dụng kỹ thuật lấy mẫu bootstrapped để cân bằng dữ liệu, giúp cải thiện khả năng phân loại các lớp thiểu số. Điều này giúp nâng cao hiệu suất mô hình, đặc biệt phù hợp với các tập dữ liệu mất cân bằng như dự đoán bệnh tim.  

---

### **Bệnh Tim theo Giới Tính**  

**Phân tích phân bố:**  
- Đa số người mắc bệnh tim là nam (`15.5K`), tiếp theo là nữ (`11.3K`).  
- Rất ít người phi nhị phân mắc bệnh tim (`15` người).  
- Sự chênh lệch đáng kể về số lượng ca mắc bệnh tim giữa nam và nữ so với người phi nhị phân cho thấy sự mất cân bằng đáng chú ý.  

---

### **Bệnh Tim theo Chủng Tộc**  

**Phân tích phân bố:**  
- Nhóm có nhiều người mắc bệnh tim nhất là "Người da trắng không gốc Tây Ban Nha" (`22.2K`).  
- Nhóm nhỏ hơn như "Người Hawaii bản địa hoặc các dân tộc đảo Thái Bình Dương khác, không gốc Tây Ban Nha" (`100` người) và "Người châu Á không gốc Tây Ban Nha" (`300` người).  
- Có sự mất cân bằng đáng kể về số ca bệnh tim giữa các nhóm chủng tộc, với số ca mắc bệnh ít hơn rõ rệt ở các nhóm thiểu số.  

---

### **Bệnh Tim theo Tình Trạng Sức Khỏe Tổng Quan**  

**Phân tích phân bố:**  
- Số ca mắc bệnh tim cao nhất thuộc nhóm có sức khỏe "Tốt" (`8.9K`), tiếp theo là nhóm "Khá" (`7.9K`).  
- Nhóm "Rất Tốt" và "Kém" có cùng số lượng ca mắc (`4.5K`).  
- Nhóm có sức khỏe "Xuất sắc" có số ca mắc bệnh tim thấp nhất (`1.0K`).  
- Có sự phân bố đáng chú ý của bệnh tim giữa các nhóm tình trạng sức khỏe, với tỷ lệ cao nhất ở những người tự đánh giá sức khỏe của họ là "Tốt" hoặc "Khá".  

---

### **Bệnh Tim theo Nhà Cung Cấp Dịch Vụ Y Tế**  

**Phân tích phân bố:**  
- Nhóm có số ca mắc bệnh tim cao nhất là những người có "Nhiều hơn một" nhà cung cấp dịch vụ y tế (`13.5K`).  
- Nhóm "Có một nhà cung cấp" cũng có số ca mắc bệnh tim đáng kể (`12.2K`).  
- Nhóm "Không có nhà cung cấp" có số ca mắc bệnh tim thấp nhất (`1.1K`).  
- Dữ liệu này cho thấy những người có nhiều hoặc ít nhất một nhà cung cấp dịch vụ y tế có khả năng mắc bệnh tim cao hơn so với những người không có nhà cung cấp dịch vụ y tế.  

---

### **Bệnh Tim theo Khả năng Khám Bác Sĩ**  

**Phân tích phân bố:**  
- Đa số người mắc bệnh tim thuộc nhóm có khả năng chi trả để khám bác sĩ (`24.5K`).  
- Một số ít người mắc bệnh tim không đủ khả năng chi trả (`2.3K`).  
- Điều này cho thấy dù có khả năng chi trả, vẫn có nhiều người mắc bệnh tim, chứng tỏ việc tiếp cận dịch vụ y tế không hoàn toàn giảm thiểu nguy cơ mắc bệnh.  
- Tuy nhiên, sự tồn tại của bệnh tim trong nhóm không đủ khả năng chi trả cũng nhấn mạnh vấn đề tiếp cận dịch vụ y tế phòng ngừa hoặc điều trị.  

---

### **Bệnh Tim theo Khám Sức Khỏe Định Kỳ**  

**Phân tích phân bố:**  
- Đa số người mắc bệnh tim đã kiểm tra sức khỏe trong vòng một năm qua (`24.9K`). Điều này cho thấy những người mắc bệnh tim thường có sự theo dõi y tế thường xuyên.  
- Có ít trường hợp mắc bệnh tim hơn trong nhóm kiểm tra sức khỏe trong vòng 2 năm (`1.1K`) và 5 năm (`0.5K`).  
- Rất ít người mắc bệnh tim chưa từng kiểm tra sức khỏe (`0.1K`) hoặc đã kiểm tra lần cuối hơn 5 năm trước (`0.3K`).  
- Dữ liệu này nhấn mạnh rằng ngay cả những người đi khám sức khỏe định kỳ vẫn có nguy cơ mắc bệnh tim, do đó cần theo dõi và phát hiện sớm để phòng ngừa.  


## **Bệnh tim và Nhồi máu cơ tim**

**Phân tích phân bố:**

* Một số lượng đáng kể người mắc bệnh tim cũng đã được chẩn đoán bị nhồi máu cơ tim (12.0K). Điều này cho thấy có mối tương quan mạnh mẽ giữa tiền sử nhồi máu cơ tim và sự hiện diện của bệnh tim.
* Tuy nhiên, cũng có một số lượng lớn người mắc bệnh tim nhưng không có tiền sử nhồi máu cơ tim (14.8K). Điều này cho thấy bệnh tim có thể phát triển mà không cần chẩn đoán nhồi máu cơ tim trước đó.
* Phân bố này chỉ ra rằng mặc dù tiền sử nhồi máu cơ tim là một chỉ báo quan trọng của bệnh tim, nhưng nhiều người mắc bệnh tim lại không có tiền sử này. Điều này nhấn mạnh sự cần thiết của việc đánh giá rủi ro tim mạch một cách toàn diện, thay vì chỉ dựa vào tiền sử nhồi máu cơ tim.

---

## **Bệnh tim và Đột quỵ**

**Phân tích phân bố:**

* Một số lượng đáng kể người mắc bệnh tim cũng từng bị đột quỵ (4.4K), cho thấy có sự liên quan đáng kể giữa đột quỵ và bệnh tim.
* Tuy nhiên, số người mắc bệnh tim nhưng không có tiền sử đột quỵ lại cao hơn (22.4K). Điều này cho thấy bệnh tim có thể xuất hiện mà không cần có tiền sử đột quỵ.
* Phân bố này nhấn mạnh rằng mặc dù đột quỵ là một yếu tố nguy cơ đáng kể đối với bệnh tim, nhưng phần lớn bệnh nhân mắc bệnh tim không có tiền sử đột quỵ. Điều này khẳng định tầm quan trọng của việc đánh giá rủi ro tim mạch một cách toàn diện, xem xét nhiều yếu tố nguy cơ khác ngoài tiền sử đột quỵ.

---

## **Bệnh tim và Bệnh thận**

**Phân tích phân bố:**

* Một số lượng đáng kể người mắc bệnh tim cũng bị chẩn đoán mắc bệnh thận (4.5K), cho thấy có mối liên quan giữa hai bệnh này.
* Tuy nhiên, phần lớn người mắc bệnh tim không bị chẩn đoán mắc bệnh thận (22.3K), điều này cho thấy bệnh tim thường xảy ra mà không liên quan đến bệnh thận.
* Phân bố này cho thấy mặc dù bệnh thận là một yếu tố nguy cơ quan trọng đối với bệnh tim, nhưng phần lớn bệnh nhân mắc bệnh tim không có tiền sử bệnh thận. Điều này nhấn mạnh sự cần thiết của việc đánh giá nhiều yếu tố nguy cơ khác đối với bệnh tim, không chỉ riêng bệnh thận.

---

## **Bệnh tim và Tiểu đường**

**Phân tích phân bố:**

* Số lượng người mắc bệnh tim nhưng không bị tiểu đường là cao nhất (16.5K). Điều này cho thấy bệnh tim phổ biến ngay cả ở những người không mắc bệnh tiểu đường.
* Một số lượng đáng kể người mắc bệnh tim có chẩn đoán tiểu đường (9.3K), cho thấy mối tương quan mạnh mẽ giữa hai bệnh này.
* Một số lượng nhỏ người mắc bệnh tim có tiền tiểu đường (0.9K) hoặc bị tiểu đường thai kỳ (0.1K), cho thấy các tình trạng này ít phổ biến hơn ở bệnh nhân mắc bệnh tim so với bệnh tiểu đường thực sự.
* Phân bố này nhấn mạnh tầm quan trọng của việc theo dõi và kiểm soát bệnh tiểu đường như một yếu tố nguy cơ quan trọng đối với bệnh tim. Tuy nhiên, việc bệnh tim xuất hiện ở cả những người không bị tiểu đường cũng cho thấy đây là một vấn đề đa yếu tố.

---

## **Bệnh tim và Chỉ số BMI**

**Phân tích phân bố:**

* Số lượng người mắc bệnh tim cao nhất nằm trong nhóm béo phì (10.6K), cho thấy mối liên quan mạnh mẽ giữa béo phì và bệnh tim.
* Nhóm thừa cân cũng có số lượng bệnh nhân mắc bệnh tim đáng kể (9.6K), nhấn mạnh thêm mối quan hệ giữa chỉ số BMI cao và nguy cơ mắc bệnh tim.
* Số người mắc bệnh tim trong nhóm có cân nặng bình thường (6.2K) và nhóm nhẹ cân (0.4K) ít hơn, điều này cho thấy duy trì cân nặng bình thường có thể liên quan đến việc giảm nguy cơ mắc bệnh tim.
* Phân bố này nhấn mạnh tầm quan trọng của việc kiểm soát cân nặng như một yếu tố quan trọng trong việc giảm nguy cơ mắc bệnh tim, với béo phì và thừa cân là những mối quan tâm chính.

---

## **Bệnh tim và Khó khăn khi đi bộ hoặc leo cầu thang**

**Phân tích phân bố:**

* Một số lượng đáng kể người mắc bệnh tim gặp khó khăn trong việc đi bộ hoặc leo cầu thang (10.8K), cho thấy có mối liên hệ chặt chẽ giữa các vấn đề về vận động và sự hiện diện của bệnh tim.
* Tuy nhiên, số lượng người mắc bệnh tim nhưng không gặp khó khăn trong việc đi bộ hoặc leo cầu thang lại cao hơn (16.0K), điều này cho thấy bệnh tim có thể xảy ra ngay cả ở những người không có vấn đề về vận động.
* Phân bố này cho thấy khó khăn trong việc đi bộ hoặc leo cầu thang là một yếu tố nguy cơ đáng kể đối với bệnh tim. Tuy nhiên, nó cũng nhấn mạnh rằng một số lượng lớn bệnh nhân mắc bệnh tim không có vấn đề về vận động, điều này khẳng định sự cần thiết của việc đánh giá rủi ro tim mạch một cách toàn diện dựa trên nhiều yếu tố sức khỏe khác nhau.
Dưới đây là bản dịch và tóm tắt nội dung của bạn mà không bao gồm hình ảnh từ GitHub:  

---

### **Bệnh Tim và Tình Trạng Sức Khỏe Thể Chất**  

**Phân Tích Phân Bố:**  
- Số lượng lớn nhất người mắc bệnh tim báo cáo có **0 ngày** cảm thấy sức khỏe thể chất không tốt (**11,6K**), cho thấy nhiều người mắc bệnh tim vẫn cảm nhận sức khỏe thể chất tốt.  
- **8,4K** người mắc bệnh tim có **14 ngày trở lên** cảm thấy sức khỏe thể chất không tốt, gợi ý mối liên quan giữa bệnh tim và thời gian kéo dài của sức khỏe kém.  
- **6,8K** người mắc bệnh tim có **1 đến 13 ngày** cảm thấy không khỏe.  
- Kết quả này nhấn mạnh tầm quan trọng của sức khỏe thể chất trong đánh giá nguy cơ bệnh tim.  

---

### **Bệnh Tim và Tình Trạng Sức Khỏe Tâm Thần**  

**Phân Tích Phân Bố:**  
- **16,7K** người mắc bệnh tim không có ngày nào cảm thấy sức khỏe tâm thần kém, cho thấy nhiều người vẫn đánh giá tinh thần của mình tốt.  
- **5,4K** người mắc bệnh tim trải qua **1 đến 13 ngày** sức khỏe tâm thần kém, thể hiện mối liên hệ giữa bệnh tim và các giai đoạn tâm lý không ổn định.  
- **4,7K** người mắc bệnh tim có **14 ngày trở lên** cảm thấy sức khỏe tâm thần không tốt, gợi ý yếu tố tâm lý cũng có thể góp phần vào bệnh tim.  
- Điều này nhấn mạnh sự cần thiết của việc đánh giá sức khỏe tinh thần trong phòng ngừa bệnh tim.  

---

### **Bệnh Tim và Bệnh Hen Suyễn**  

**Phân Tích Phân Bố:**  
- **21,5K** người mắc bệnh tim **chưa bao giờ bị hen suyễn**, cho thấy phần lớn không có tiền sử bệnh này.  
- **4,1K** người mắc bệnh tim **đang bị hen suyễn**, chỉ ra sự liên quan giữa bệnh hen suyễn và bệnh tim.  
- **1,2K** người từng bị hen suyễn nhưng đã khỏi có bệnh tim, cho thấy ảnh hưởng của hen suyễn trước đó là không đáng kể.  
- Kết quả này khẳng định cần theo dõi bệnh hen suyễn ở những người có nguy cơ mắc bệnh tim.  

---

### **Bệnh Tim và Hút Thuốc**  

**Phân Tích Phân Bố:**  
- **12,1K** người mắc bệnh tim **chưa bao giờ hút thuốc** (có thể phản ánh số lượng lớn nhóm này).  
- **11,1K** người mắc bệnh tim **từng hút thuốc**, cho thấy tác động lâu dài của thuốc lá dù đã cai.  
- **2,7K** người hút thuốc hàng ngày mắc bệnh tim, chứng tỏ hút thuốc hiện tại là nguy cơ lớn.  
- **0,9K** người chỉ hút thuốc thỉnh thoảng mắc bệnh tim, có thể do ít tiếp xúc với khói thuốc hơn.  
- Kết quả này nhấn mạnh tầm quan trọng của việc bỏ thuốc để giảm nguy cơ bệnh tim.  

---

### **Bệnh Tim và Uống Rượu Quá Độ**  

**Phân Tích Phân Bố:**  
- **24,8K** người mắc bệnh tim **không uống rượu quá độ**, cho thấy đây không phải yếu tố duy nhất gây bệnh tim.  
- **2,0K** người mắc bệnh tim **có uống rượu quá độ**, chứng minh mối liên hệ giữa rượu và bệnh tim.  
- Kết quả này nhấn mạnh rằng rượu chỉ là một phần trong bức tranh tổng thể về nguy cơ bệnh tim, và cần có cách tiếp cận toàn diện khi đánh giá nguy cơ bệnh lý.  

---

### **Bệnh Tim và Tập Thể Dục**  

**Phân Tích Phân Bố:**  
- **16,8K** người mắc bệnh tim **có tập thể dục** trong 30 ngày qua, chứng minh rằng tập thể dục không thể loại bỏ hoàn toàn nguy cơ bệnh tim.  
- **10,0K** người mắc bệnh tim **không tập thể dục**, thể hiện mối liên quan giữa lối sống ít vận động và bệnh tim.  
- Điều này nhấn mạnh rằng tập thể dục rất quan trọng nhưng cần kết hợp với chế độ ăn uống và kiểm tra sức khỏe định kỳ để giảm nguy cơ bệnh tim.  

---

### **Bệnh Tim và Độ Tuổi**  

**Phân Tích Phân Bố:**  
- Nhóm có số ca bệnh tim cao nhất là **70-74 tuổi (4,9K)**, tiếp theo là **80 tuổi trở lên (5,7K)** và **75-79 tuổi (4,4K)**.  
- Số ca bệnh tim tăng dần theo độ tuổi, cho thấy tuổi tác là yếu tố nguy cơ lớn.  
- Ít trường hợp bệnh tim hơn ở nhóm **18-24 tuổi**, nhấn mạnh rằng bệnh tim ít phổ biến ở người trẻ.  
- Kết quả này nhấn mạnh tầm quan trọng của việc kiểm tra sức khỏe định kỳ khi tuổi tác tăng, đặc biệt với người trên 65 tuổi.  


### **Bệnh tim và thời gian ngủ**  

**Phân tích phân bố**  

- Số lượng người mắc bệnh tim cao nhất thuộc nhóm ngủ bình thường (6 - 8 giờ) với `19.5K` người. Điều này cho thấy nhiều người mắc bệnh tim vẫn có giấc ngủ tiêu chuẩn.  
- Số người mắc bệnh tim trong nhóm ngủ rất ít (0 - 3 giờ) và nhóm ngủ quá dài (11 giờ trở lên) đều thấp, chỉ `0.6K` người. Điều này chứng tỏ thời gian ngủ cực đoan ít phổ biến ở nhóm mắc bệnh tim nhưng vẫn có tồn tại.  
- Nhóm ngủ ít (4 - 5 giờ) có `3.3K` người mắc bệnh tim, cho thấy mối liên hệ đáng kể giữa thiếu ngủ và bệnh tim.  
- Nhóm ngủ dài (9 - 10 giờ) có `2.8K` người mắc bệnh tim, cho thấy thời gian ngủ kéo dài cũng liên quan đến bệnh tim nhưng mức độ thấp hơn so với nhóm ngủ bình thường.  
- Sự phân bố này làm nổi bật mối quan hệ phức tạp giữa thời gian ngủ và bệnh tim. Mặc dù giấc ngủ bình thường phổ biến trong nhóm mắc bệnh tim, cả thiếu ngủ lẫn ngủ quá nhiều đều là các yếu tố cần xem xét khi đánh giá nguy cơ tim mạch. Điều này nhấn mạnh tầm quan trọng của việc duy trì thói quen ngủ lành mạnh để bảo vệ sức khỏe tim mạch.  

---

### **Bệnh tim và thói quen uống rượu**  

**Phân tích phân bố**  

- Số lượng người mắc bệnh tim cao nhất thuộc nhóm "Không uống rượu" với `15.9K` người. Điều này cho thấy việc không uống rượu phổ biến trong nhóm mắc bệnh tim, có thể do lý do sức khỏe hoặc tình trạng bệnh lý sẵn có.  
- Nhóm tiêu thụ rượu rất ít (0.01 - 1 ly) đứng thứ hai với `4.0K` người mắc bệnh tim, cho thấy một số ít người mắc bệnh tim vẫn có thói quen uống rượu tối thiểu.  
- Nhóm tiêu thụ ít rượu (1.01 - 5 ly) có `3.4K` người mắc bệnh tim, cho thấy uống rượu ở mức độ vừa phải cũng tồn tại trong nhóm này.  
- Nhóm uống rượu mức trung bình (5.01 - 10 ly) và cao (10.01 - 20 ly) có số lượng bệnh nhân tim mạch thấp hơn, lần lượt là `1.7K` và `1.0K` người.  
- Nhóm uống rượu rất nhiều (trên 20 ly) là nhóm có số lượng người mắc bệnh tim thấp nhất với `0.7K` người.  
- Sự phân bố này cho thấy nhiều người mắc bệnh tim không uống hoặc chỉ uống rất ít rượu, trong khi tiêu thụ rượu từ trung bình đến cao ít phổ biến hơn trong nhóm này. Điều này nhấn mạnh tầm quan trọng của việc cân nhắc thói quen uống rượu trong bối cảnh nguy cơ mắc bệnh tim, đồng thời cho thấy lợi ích tiềm năng của việc hạn chế hoặc không sử dụng rượu đối với sức khỏe tim mạch.  

---

## **Tương quan giữa bệnh tim và các đặc điểm khác**  

Phân tích này nhằm xác định mối quan hệ giữa bệnh tim và các đặc điểm khác trong tập dữ liệu thông qua ba bước chính:  

1. **Mã hóa biến phân loại:** Chuyển đổi tất cả các biến phân loại thành giá trị số bằng cách sử dụng phương pháp CatBoost Encoding. Việc này giúp đảm bảo các mô hình máy học có thể xử lý dữ liệu một cách hiệu quả.  
2. **Tính toán thông tin hỗ tương (Mutual Information):** Đánh giá mức độ phụ thuộc giữa từng đặc điểm với bệnh tim để xác định khả năng dự đoán của từng đặc điểm. Mutual Information có thể phát hiện cả mối quan hệ tuyến tính và phi tuyến tính.  
3. **Tính toán hệ số tương quan Pearson:** Tạo bản đồ nhiệt (heatmap) để trực quan hóa mối tương quan tuyến tính giữa bệnh tim và các đặc điểm khác. Điều này giúp hiểu rõ hơn về quan hệ tuyến tính trong tập dữ liệu.  

---

### **Mã hóa biến phân loại bằng CatBoost**  

Hầu hết các thuật toán học máy yêu cầu dữ liệu dưới dạng số. Vì vậy, trước khi huấn luyện mô hình hoặc tính toán tương quan, cần phải chuyển đổi dữ liệu phân loại thành dạng số. Một trong những phương pháp hiệu quả là **CatBoost Encoding**, đây là một kỹ thuật mã hóa mục tiêu (target-based encoding).  

Mã hóa mục tiêu thay thế một đặc điểm phân loại bằng giá trị trung bình của biến mục tiêu (bệnh tim) tương ứng với từng danh mục. Tuy nhiên, điều này có thể dẫn đến rò rỉ mục tiêu (target leakage) khi mô hình vô tình sử dụng thông tin của biến mục tiêu để dự đoán chính nó. Điều này làm cho mô hình bị overfitting và khó tổng quát hóa với dữ liệu chưa thấy trước.  

CatBoost Encoding khắc phục vấn đề này bằng cách sử dụng nguyên tắc sắp xếp dữ liệu tương tự như trong chuỗi thời gian. Thay vì sử dụng toàn bộ tập dữ liệu để tính toán, CatBoost chỉ sử dụng thông tin từ các quan sát trước đó, giúp giảm thiểu nguy cơ rò rỉ mục tiêu.  

---

### **Thông tin hỗ tương - Đánh giá khả năng dự đoán**  

Thông tin hỗ tương (Mutual Information - MI) là một thước đo sự phụ thuộc giữa hai biến. Nó cho biết một biến cung cấp bao nhiêu thông tin về biến còn lại. Không giống như hệ số tương quan Pearson chỉ đo lường mối quan hệ tuyến tính, MI có thể nắm bắt cả mối quan hệ tuyến tính và phi tuyến tính.  

#### **Ưu điểm của Mutual Information:**  

- **Phát hiện mối quan hệ phi tuyến tính:** Không giống như Pearson, MI có thể nắm bắt các mối quan hệ phức tạp giữa các đặc điểm và biến mục tiêu.  
- **Xác định tính độc lập:** Điểm MI bằng 0 cho thấy hai biến hoàn toàn độc lập, trong khi điểm cao hơn cho thấy có sự liên hệ giữa chúng.  
- **Đánh giá khả năng dự đoán:** Đặc điểm có điểm MI cao đồng nghĩa với việc nó chứa nhiều thông tin hơn về biến mục tiêu, giúp cải thiện hiệu suất của mô hình dự đoán.  

#### **Diễn giải điểm số Mutual Information:**  

Điểm MI càng cao, đặc điểm đó càng liên quan đến bệnh tim. Điều này giúp xác định những đặc điểm quan trọng cần tập trung khi xây dựng mô hình dự đoán nguy cơ bệnh tim.  
### **Các Đặc Điểm Có Điểm Thông Tin Tương Hỗ (Mutual Information) Cao Nhất**
* **could_not_afford_to_see_doctor (0.08)**:  
  Đây là đặc điểm có điểm thông tin tương hỗ cao nhất, cho thấy rằng việc không đủ khả năng tài chính để khám bác sĩ là yếu tố quan trọng nhất trong dự đoán bệnh tim. Điều này phản ánh mối liên hệ đáng kể giữa rào cản tài chính và nguy cơ mắc bệnh tim.

* **ever_diagnosed_with_heart_attack (0.07)**:  
  Được chẩn đoán từng bị đau tim là một đặc điểm có giá trị dự đoán cao đối với bệnh tim, bởi vì tiền sử đau tim là một chỉ báo mạnh về tình trạng sức khỏe tim mạch kéo dài.

* **ever_diagnosed_with_a_stroke (0.07)**:  
  Tiền sử bị đột quỵ cũng là một yếu tố dự báo mạnh cho bệnh tim, cho thấy rằng những người từng bị đột quỵ có khả năng cao mắc bệnh tim hơn.

* **difficulty_walking_or_climbing_stairs (0.06)**:  
  Khó khăn khi đi bộ hoặc leo cầu thang có liên quan chặt chẽ đến bệnh tim, phản ánh các hạn chế thể chất do các vấn đề về tim mạch gây ra.

* **ever_told_you_have_kidney_disease (0.06)**:  
  Bệnh thận có mối quan hệ đáng kể với bệnh tim, phù hợp với các nghiên cứu trước đây về sự tương quan giữa các vấn đề về tim mạch và thận.

* **sleep_category (0.06)**:  
  Giấc ngủ cũng có liên quan đáng kể đến bệnh tim, có thể cho thấy rằng giấc ngủ kém chất lượng là một yếu tố nguy cơ.

* **binge_drinking_status (0.06)**:  
  Việc uống rượu bia quá mức có liên hệ với bệnh tim, cho thấy rằng hành vi tiêu thụ rượu có thể góp phần vào nguy cơ mắc bệnh.

* **ever_told_you_had_diabetes (0.06)**:  
  Bệnh tiểu đường là một yếu tố nguy cơ quan trọng đối với bệnh tim và có điểm thông tin tương hỗ cao.

* **asthma_Status (0.05)**:  
  Bệnh hen suyễn cũng cung cấp thông tin đáng kể về nguy cơ mắc bệnh tim, có thể do các yếu tố nguy cơ chung hoặc tình trạng bệnh đi kèm.

### **Các Đặc Điểm Có Điểm Thông Tin Tương Hỗ Trung Bình**
* **race (0.05), exercise_status_in_past_30_Days (0.05), length_of_time_since_last_routine_checkup (0.05)**:  
  Những đặc điểm này có điểm thông tin tương hỗ trung bình, có nghĩa là chúng cung cấp thông tin hữu ích nhưng ít quan trọng hơn.

* **ever_told_you_had_a_depressive_disorder (0.04), smoking_status (0.04), general_health (0.04), physical_health_status (0.04)**:  
  Sức khỏe tâm thần, tình trạng hút thuốc và sức khỏe tổng quát đều có mối liên hệ đáng kể với bệnh tim, phản ánh bản chất đa yếu tố của các nguy cơ gây bệnh tim.

### **Các Đặc Điểm Có Điểm Thông Tin Tương Hỗ Thấp**
* **gender (0.03), health_care_provider (0.03), age_category (0.03), mental_health_status (0.03), drinks_category (0.03), BMI (0.02)**:  
  Những đặc điểm này có điểm số thấp hơn, cho thấy rằng chúng vẫn cung cấp thông tin nhưng đóng góp nhỏ hơn trong việc dự đoán bệnh tim.

### **Tóm Tắt**
Các đặc điểm có điểm thông tin tương hỗ cao nhất bao gồm rào cản tài chính trong việc khám bác sĩ, tiền sử bệnh tim mạch (như đau tim hoặc đột quỵ) và các vấn đề thể chất. Những yếu tố này có thể được ưu tiên trong việc phân tích sâu hơn hoặc xây dựng mô hình dự đoán bệnh tim.

---

## **Tương Quan Pearson**
Hệ số tương quan Pearson (r) là thước đo mối quan hệ tuyến tính giữa hai biến liên tục. Giá trị của r có thể nằm trong khoảng từ -1 đến 1, với:
* **+1**: Mối quan hệ tuyến tính dương hoàn hảo.
* **-1**: Mối quan hệ tuyến tính âm hoàn hảo.
* **0**: Không có mối quan hệ tuyến tính.

### **Phân Tích Tính Đa Cộng Tuyến**
Dưới đây là các mối quan hệ đáng chú ý giữa các đặc điểm trong tập dữ liệu:

#### **Mối Tương Quan Cao**
* **General Health và Physical Health Status (0.53)**:  
  Có sự tương quan dương mạnh giữa sức khỏe tổng quát và tình trạng sức khỏe thể chất, cho thấy rằng những người có sức khỏe tổng quát tốt thường có sức khỏe thể chất tốt hơn.

* **Difficulty Walking or Climbing Stairs và Physical Health Status (0.43)**:  
  Có sự tương quan mạnh giữa khó khăn vận động và sức khỏe thể chất, phản ánh rằng khó khăn trong các hoạt động thể chất là một yếu tố quan trọng trong đánh giá tình trạng sức khỏe.

* **General Health và Difficulty Walking or Climbing Stairs (0.29)**:  
  Tương quan trung bình, cho thấy rằng sức khỏe tổng quát có ảnh hưởng đáng kể đến khả năng vận động.

* **Ever Told You Had Diabetes và General Health (0.26)**:  
  Tương quan trung bình, cho thấy rằng bệnh tiểu đường là một yếu tố quan trọng ảnh hưởng đến sức khỏe tổng quát.

* **Ever Told You Had Diabetes và Physical Health Status (0.26)**:  
  Tương quan trung bình, cho thấy rằng bệnh tiểu đường có ảnh hưởng đáng kể đến tình trạng sức khỏe thể chất.

* **Length of Time Since Last Routine Checkup và General Health (0.26)**:  
  Tương quan trung bình, cho thấy rằng kiểm tra sức khỏe định kỳ có liên quan đến việc theo dõi sức khỏe tổng quát.

* **General Health và Exercise Status in Past 30 Days (0.29)**:  
  Tương quan trung bình, cho thấy rằng việc tập thể dục là một yếu tố quan trọng giúp duy trì sức khỏe tổng quát.

### **Tương Quan Trung Bình**  
- **Từng Được Chẩn Đoán Nhồi Máu Cơ Tim và Từng Được Chẩn Đoán Đột Quỵ (0.18):** Có tương quan trung bình, cho thấy hai bệnh này thường xảy ra cùng nhau.  
- **Từng Được Chẩn Đoán Tiểu Đường và Từng Được Chẩn Đoán Nhồi Máu Cơ Tim (0.20):** Tương quan trung bình cho thấy mối liên hệ phổ biến giữa bệnh tim và bệnh tiểu đường.  
- **Từng Được Chẩn Đoán Đột Quỵ và Từng Được Chẩn Đoán Nhồi Máu Cơ Tim (0.18):** Có tương quan trung bình, phản ánh yếu tố nguy cơ chung.  
- **Từng Được Chẩn Đoán Bệnh Thận và Từng Được Chẩn Đoán Tiểu Đường (0.19):** Tương quan trung bình cho thấy sự phổ biến của các bệnh đi kèm.  
- **Từng Được Chẩn Đoán Nhồi Máu Cơ Tim và Từng Được Chẩn Đoán Bệnh Thận (0.15):** Tương quan trung bình cho thấy mối liên hệ giữa bệnh thận và bệnh tim.  
- **Có Nhà Cung Cấp Dịch Vụ Y Tế và Sức Khỏe Tổng Quát (0.26):** Tương quan trung bình cho thấy mối liên hệ giữa việc có nhà cung cấp dịch vụ y tế và tình trạng sức khỏe tổng quát.  
- **Có Nhà Cung Cấp Dịch Vụ Y Tế và Không Đủ Khả Năng Tài Chính Để Gặp Bác Sĩ (0.16):** Tương quan trung bình, cho thấy các rào cản tài chính ảnh hưởng đến khả năng tiếp cận chăm sóc sức khỏe.  
- **Tình Trạng Hút Thuốc và Uống Rượu Binge (0.11):** Tương quan yếu, nhưng vẫn cho thấy mối liên hệ giữa các yếu tố lối sống này.  

---

### **Diễn Giải Biến Mục Tiêu**  
  
Biểu đồ heatmap hiển thị hệ số tương quan Pearson giữa các biến và bệnh tim. Các giá trị dao động từ `-1` đến `1`, trong đó:  
- Giá trị gần `1` biểu thị mối tương quan dương mạnh.  
- Giá trị gần `-1` biểu thị mối tương quan âm mạnh.  
- Giá trị xung quanh `0` cho thấy không có mối tương quan tuyến tính đáng kể.  

#### **Mối Tương Quan Tuyến Tính Dương Mạnh Nhất**  
- **Từng Được Chẩn Đoán Nhồi Máu Cơ Tim (0.43):**  
  → Đây là yếu tố có tương quan dương mạnh nhất với bệnh tim. Những người có tiền sử nhồi máu cơ tim có khả năng mắc bệnh tim cao hơn đáng kể.  

#### **Mối Tương Quan Tuyến Tính Dương Trung Bình**  
- **Sức Khỏe Tổng Quát (0.21):**  
  → Sức khỏe tổng quát kém có liên quan đến nguy cơ mắc bệnh tim cao hơn.  
- **Nhóm Tuổi (0.21):**  
  → Người lớn tuổi có nguy cơ mắc bệnh tim cao hơn.  
- **Khó Đi Lại Hoặc Leo Cầu Thang (0.17):**  
  → Những người gặp khó khăn trong vận động có khả năng mắc bệnh tim cao hơn.  
- **Từng Được Chẩn Đoán Tiểu Đường (0.16):**  
  → Có mối liên hệ trung bình giữa bệnh tiểu đường và bệnh tim.  
- **Từng Được Chẩn Đoán Đột Quỵ (0.15):**  
  → Những người có tiền sử đột quỵ có nguy cơ mắc bệnh tim cao hơn.  
- **Từng Được Chẩn Đoán Bệnh Thận (0.15):**  
  → Có mối liên hệ giữa bệnh thận và bệnh tim.  
- **Tình Trạng Sức Khỏe Thể Chất (0.14):**  
  → Sức khỏe thể chất kém có liên quan đến bệnh tim.  

#### **Mối Tương Quan Tuyến Tính Dương Yếu**  
- **Có Nhà Cung Cấp Dịch Vụ Y Tế (0.11):**  
  → Cho thấy một số mối liên hệ giữa việc có bác sĩ và bệnh tim, có thể do tỷ lệ chẩn đoán cao hơn.  
- **Tình Trạng Hút Thuốc (0.08):**  
  → Hút thuốc có tương quan yếu với bệnh tim.  
- **Thời Gian Kể Từ Lần Khám Sức Khỏe Gần Nhất (0.08):**  
  → Khoảng thời gian dài hơn giữa các lần khám sức khỏe có liên quan nhẹ đến bệnh tim.  
- **Tình Trạng Tập Thể Dục Trong 30 Ngày Qua (0.08):**  
  → Tần suất tập thể dục ít có thể liên quan đến bệnh tim.  
- **Tình Trạng Uống Rượu (0.06):**  
  → Có tương quan dương yếu giữa uống rượu và bệnh tim.  
- **Tình Trạng Giấc Ngủ (0.05):**  
  → Thiếu ngủ có thể liên quan đến bệnh tim.  
- **Chủng Tộc (0.05):**  
  → Có mối liên hệ rất nhẹ giữa chủng tộc và bệnh tim.  
- **Tình Trạng Uống Rượu Binge (0.05):**  
  → Uống rượu binge có tương quan yếu với bệnh tim.  
- **Tình Trạng Hen Suyễn (0.04):**  
  → Có tương quan rất yếu giữa hen suyễn và bệnh tim.  
- **Giới Tính (0.04):**  
  → Có tương quan yếu giữa giới tính và bệnh tim.  
- **Chỉ Số BMI (0.04):**  
  → Có tương quan yếu giữa BMI và bệnh tim.  
- **Tình Trạng Sức Khỏe Tinh Thần (0.04):**  
  → Có mối liên hệ yếu giữa sức khỏe tinh thần và bệnh tim.  
- **Từng Được Chẩn Đoán Rối Loạn Trầm Cảm (0.03):**  
  → Có mối liên hệ rất nhẹ giữa trầm cảm và bệnh tim.  
- **Không Đủ Khả Năng Tài Chính Để Gặp Bác Sĩ (0.00):**  
  → Không có mối tương quan tuyến tính đáng kể giữa khả năng tài chính và bệnh tim.  

---

### **So Sánh Tương Quan Pearson và Thông Tin Tương Hỗ (Mutual Information)**  

#### **So Sánh**  
- **Quan Hệ Tuyến Tính:** Tương quan Pearson hiệu quả trong việc xác định mối quan hệ tuyến tính. Biến **"Từng Được Chẩn Đoán Nhồi Máu Cơ Tim"** có tương quan cao nhất với bệnh tim, phản ánh mối liên hệ trực tiếp.  
- **Quan Hệ Phi Tuyến Tính:** Thông tin tương hỗ nắm bắt cả mối quan hệ tuyến tính và phi tuyến. **"Không Đủ Khả Năng Tài Chính Để Gặp Bác Sĩ"** có giá trị thông tin tương hỗ cao nhất, cho thấy yếu tố tài chính có ảnh hưởng lớn đến bệnh tim nhưng không có tương quan tuyến tính rõ ràng.  
- **Điểm Chung:** Một số yếu tố như **"Từng Được Chẩn Đoán Nhồi Máu Cơ Tim"** và **"Sức Khỏe Tổng Quát"** đều có giá trị cao ở cả hai phương pháp, chứng tỏ đây là những đặc điểm dự đoán quan trọng.  
- **Khác Biệt:** Thông tin tương hỗ phát hiện một số yếu tố quan trọng như **"Không Đủ Khả Năng Tài Chính Để Gặp Bác Sĩ"** và **"Tình Trạng Giấc Ngủ"**, vốn không nổi bật trong tương quan Pearson.  

#### **Kết Luận**  
- **Tương Quan Pearson** phù hợp để xác định các mối quan hệ tuyến tính mạnh.  
- **Thông Tin Tương Hỗ** cung cấp cái nhìn rộng hơn về cả mối quan hệ tuyến tính và phi tuyến tính.  
- Kết hợp cả hai phương pháp sẽ giúp tối ưu hóa việc chọn đặc trưng cho mô hình dự đoán bệnh tim.











































































