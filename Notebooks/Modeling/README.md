## **Mô Hình: Đánh Giá Rủi Ro Bệnh Tim Mạch Sử Dụng AI**

## **Giới Thiệu**

Trong notebook này, chúng ta sẽ tiến hành huấn luyện và đánh giá nhiều mô hình học máy để phân loại bệnh tim mạch:

* **Logistic Regression (Hồi Quy Logistic)**
* **Random Forest (Rừng Ngẫu Nhiên)**
* **XGBoost (XGBoost)**
* **LightGBM (LightGBM)**
* **Balanced Bagging (Bao Túi Cân Bằng)**
* **Easy Ensemble (Hợp Nhóm Dễ Dàng)**
* **Balanced Random Forest (Rừng Ngẫu Nhiên Cân Bằng)**
* **Balanced Bagging (LightGBM): Balanced Bagging as a Wrapper and LightGBM as a base estimator (Bao Túi Cân Bằng (LightGBM): Bao Túi Cân Bằng như một bộ bao bọc và LightGBM làm bộ ước lượng cơ sở)**
* **Easy Ensemble (LightGBM): Easy Ensemble as a Wrapper and LightGBM as a base estimator (Hợp Nhóm Dễ Dàng (LightGBM): Hợp Nhóm Dễ Dàng như một bộ bao bọc và LightGBM làm bộ ước lượng cơ sở)**

Mục tiêu của chúng tôi là dự đoán chính xác rủi ro bệnh tim mạch bằng cách sử dụng các mô hình này. Chúng tôi sẽ sử dụng tinh chỉnh siêu tham số với `Optuna` để tối ưu hiệu suất của mỗi mô hình. Ngoài ra, chúng tôi sẽ sử dụng các bộ phân loại `BalancedRandomForestClassifier`, `BalancedBaggingClassifier` và `EasyEnsembleClassifier` từ thư viện `imbalanced-learn` để giải quyết vấn đề mất cân bằng lớp. Các bộ phân loại này sử dụng phương pháp lấy mẫu bootstrap để cân bằng bộ dữ liệu, đảm bảo phân loại mạnh mẽ các lớp thiểu số. Bằng cách tập trung vào dữ liệu ít xuất hiện hơn, nó nâng cao hiệu suất mô hình, đặc biệt thích hợp cho các bộ dữ liệu mất cân bằng như dự đoán bệnh tim mạch.

Qua cách tiếp cận toàn diện này, chúng tôi mong muốn phát triển một mô hình đáng tin cậy và hiệu quả để đánh giá rủi ro bệnh tim mạch, góp phần vào kết quả sức khỏe tốt hơn.

## **Dữ Liệu**

Bộ dữ liệu sử dụng trong notebook Mô Hình này là kết quả của một quy trình xử lý dữ liệu toàn diện. Xử lý dữ liệu là một bước quan trọng trong quy trình làm việc của khoa học dữ liệu, bao gồm việc biến đổi và chuẩn bị dữ liệu thô thành định dạng có thể sử dụng hơn. Các nhiệm vụ chính được thực hiện trong quá trình xử lý dữ liệu bao gồm:

* Xử lý dữ liệu thiếu
* Ánh xạ dữ liệu
* Dọn dẹp dữ liệu
* Kỹ thuật đặc trưng
  
Các bước này đã đảm bảo rằng bộ dữ liệu đã được chuẩn bị tốt cho việc phân tích và mô hình hóa, giúp chúng tôi xây dựng các mô hình đáng tin cậy và mạnh mẽ cho việc dự đoán bệnh tim mạch.

## **Các Đặc Trưng Liên Quan Đến Bệnh Tim Mạch**

Sau vài ngày nghiên cứu và phân tích các đặc trưng trong bộ dữ liệu, chúng tôi đã xác định được các đặc trưng chính sau đây để đánh giá bệnh tim mạch:

* **Biến Đích (Biến Phụ Thuộc):**
    * Heart_disease: "Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease" (Đã từng được chẩn đoán bệnh đau ngực hoặc bệnh tim mạch vành)
* **Dân Số Học:**
    * Gender (Giới tính)
    * Race (Chủng tộc)
    * Age_category (Danh mục độ tuổi)
* **Lịch Sử Y Tế:**
    * General_Health (Tình trạng sức khỏe chung)
    * Have_Personal_Health_Care_Provider (Có bác sĩ chăm sóc sức khỏe cá nhân)
    * Could_Not_Afford_To_See_Doctor (Không đủ khả năng để gặp bác sĩ)
    * Length_of_time_since_last_routine_checkup (Thời gian kể từ lần kiểm tra sức khỏe định kỳ cuối)
    * Ever_Diagnosed_with_Heart_Attack (Đã từng được chẩn đoán nhồi máu cơ tim)
    * Ever_Diagnosed_with_a_Stroke (Đã từng được chẩn đoán đột quỵ)
    * Ever_told_you_had_a_depressive_disorder (Đã từng được cho biết có rối loạn trầm cảm)
    * Ever_told_you_have_kidney_disease (Đã từng được cho biết có bệnh thận)
    * Ever_told_you_had_diabetes (Đã từng được cho biết có bệnh tiểu đường)
    * BMI (Chỉ số khối cơ thể)
    * Difficulty_Walking_or_Climbing_Stairs (Khó khăn khi đi bộ hoặc leo cầu thang)
    * Physical_Health_Status (Tình trạng sức khỏe thể chất)
    * Mental_Health_Status (Tình trạng sức khỏe tâm thần)
    * Asthma_Status (Tình trạng hen suyễn)
* **Lối Sống:**
    * Smoking_status (Tình trạng hút thuốc)
    * Binge_Drinking_status (Tình trạng uống rượu quá độ)
    * Drinks_category (Danh mục uống rượu)
    * Exercise_in_Past_30_Days (Hoạt động thể chất trong 30 ngày qua)
    * Sleep_category (Danh mục giấc ngủ)

## **Chuyển Đổi Kiểu Dữ Liệu Đặc Trưng**

Trong pandas, kiểu dữ liệu object được sử dụng cho dữ liệu văn bản hoặc dữ liệu hỗn hợp. Khi một cột chứa dữ liệu phân loại, việc chuyển đổi rõ ràng nó sang kiểu dữ liệu category thường có lợi. Dưới đây là một số lý do tại sao:

**Lợi ích của việc chuyển sang kiểu phân loại:**
* **Tiết kiệm bộ nhớ:** Kiểu dữ liệu phân loại hiệu quả hơn về bộ nhớ. Thay vì lưu trữ mỗi chuỗi duy nhất riêng biệt, pandas lưu trữ các danh mục và sử dụng mã số nguyên để đại diện cho các giá trị.
* **Cải thiện hiệu suất:** Các thao tác trên dữ liệu phân loại có thể nhanh hơn vì pandas có thể sử dụng các mã số nguyên bên dưới.
* **Ý nghĩa rõ ràng:** Việc chuyển sang kiểu phân loại làm rõ tính phân loại của dữ liệu, cải thiện khả năng đọc mã và giảm thiểu rủi ro khi xử lý dữ liệu phân loại như dữ liệu liên tục.

## **Bệnh Tim Mạch: Biến Đích**

**Phân Tích Phân Phối**
* Có một sự mất cân bằng lớn giữa hai loại.
* Phần lớn cá nhân không bị bệnh tim mạch `418.3K`, trong khi số lượng nhỏ hơn bị bệnh tim mạch là `26.8K`.
* Sự mất cân bằng này có thể dễ dàng quan sát qua biểu đồ, với thanh màu xanh lá dài hơn nhiều so với thanh màu đỏ.

**Vấn Đề Mất Cân Bằng**
* **Định kiến Mô Hình:** Khi huấn luyện một mô hình phân loại trên bộ dữ liệu mất cân bằng này, mô hình có thể trở nên thiên lệch về việc dự đoán lớp chiếm ưu thế (Không bị bệnh tim mạch) thường xuyên hơn vì nó xuất hiện nhiều hơn trong dữ liệu huấn luyện.
* **Chỉ Số Hiệu Suất:** Các chỉ số hiệu suất phổ biến như độ chính xác có thể gây hiểu lầm trong bộ dữ liệu mất cân bằng. Ví dụ, một mô hình luôn dự đoán "Không bị bệnh tim mạch" sẽ có độ chính xác cao vì lớp chiếm ưu thế được đại diện tốt. Tuy nhiên, mô hình này sẽ không xác định đúng các cá nhân bị bệnh tim mạch, điều này rất quan trọng trong các ứng dụng y tế.
* **Nhớ và Chính Xác:** Các chỉ số như độ nhớ (nhạy cảm) và độ chính xác là những thông số thông tin hơn trong trường hợp này. Độ nhớ đo lường khả năng xác định các trường hợp đúng dương tính (bệnh tim mạch), trong khi độ chính xác đo lường độ chính xác của các dự đoán dương tính. Trong bộ dữ liệu mất cân bằng, một mô hình có thể có độ nhớ thấp đối với lớp thiểu số (bệnh tim mạch) ngay cả khi nó có độ chính xác cao tổng thể.

**Chiến Lược Giải Quyết Mất Cân Bằng**
Các bộ phân loại `BalancedRandomForestClassifier` hoặc `BalancedBaggingClassifier` hoặc `EasyEnsembleClassifier` từ thư viện `imbalanced-learn` xử lý hiệu quả vấn đề mất cân bằng lớp bằng cách sử dụng phương pháp lấy mẫu bootstrap để cân bằng bộ dữ liệu, đảm bảo phân loại mạnh mẽ các lớp thiểu số. Nó nâng cao hiệu suất mô hình bằng cách tập trung vào các dữ liệu ít xuất hiện hơn, làm cho nó lý tưởng cho các bộ dữ liệu mất cân bằng như dự đoán bệnh tim mạch.
## **Mã Hóa Danh Mục với Catboost**

Nhiều thuật toán học máy yêu cầu dữ liệu phải ở dạng số. Do đó, trước khi huấn luyện mô hình hoặc tính toán sự tương quan (Pearson) hoặc thông tin hỗn hợp (sức mạnh dự đoán), chúng ta cần chuyển đổi dữ liệu danh mục thành dạng số. Có nhiều phương pháp mã hóa danh mục khác nhau, và CatBoost là một trong số đó. CatBoost là một bộ mã hóa dựa trên mục tiêu. Đây là một bộ mã hóa có giám sát, mã hóa các cột danh mục theo giá trị mục tiêu, hỗ trợ cả mục tiêu nhị phân và liên tục.

Mã hóa mục tiêu (Target Encoding) là một kỹ thuật phổ biến được sử dụng để mã hóa danh mục. Nó thay thế một đặc trưng danh mục bằng giá trị trung bình của mục tiêu tương ứng với danh mục đó trong bộ dữ liệu huấn luyện kết hợp với xác suất mục tiêu trên toàn bộ bộ dữ liệu. Tuy nhiên, điều này tạo ra sự rò rỉ mục tiêu vì mục tiêu được sử dụng để dự đoán chính mục tiêu. Các mô hình như vậy có xu hướng bị overfitting và không tổng quát tốt khi gặp các tình huống chưa thấy.

Bộ mã hóa CatBoost tương tự như mã hóa mục tiêu, nhưng cũng sử dụng một nguyên tắc sắp xếp để khắc phục vấn đề rò rỉ mục tiêu. Nó sử dụng nguyên lý tương tự như kiểm tra dữ liệu chuỗi thời gian. Các giá trị thống kê mục tiêu phụ thuộc vào lịch sử đã quan sát, tức là xác suất mục tiêu cho đặc trưng hiện tại chỉ được tính từ các dòng (quan sát) trước đó.

## **Mô Hình Cơ Bản**

Ở đây, chúng ta sẽ huấn luyện các mô hình sau và so sánh hiệu suất của chúng ở cấp độ từng lớp:

* Hồi quy logistic (Logistic Regression)
* Rừng ngẫu nhiên (Random Forest)
* XGBoost
* LightGBM
* Balanced Bagging
* Easy Ensemble
* Balanced Random Forest
* Balanced Bagging (LightGBM): Balanced Bagging như một bộ bao bọc và LightGBM làm ước lượng cơ sở
* Easy Ensemble (LightGBM): Easy Ensemble như một bộ bao bọc và LightGBM làm ước lượng cơ sở

### **So Sánh Các Chỉ Số Cấp Lớp Cụ Thể**


* **Recall cao, Precision và F1 Score thấp:** Hầu hết các mô hình cho thấy recall kém cho lớp 1 (bệnh nhân mắc bệnh tim), ngoại trừ Balanced Bagging, Easy Ensemble, Balanced Random Forest, và khi các mô hình này kết hợp với LightGBM. Điều này cho thấy hầu hết các mô hình gặp khó khăn trong việc nhận diện các trường hợp dương tính (bệnh nhân mắc bệnh tim), dẫn đến một số lượng lớn các trường hợp âm tính giả (bệnh nhân bị nhận diện sai là không mắc bệnh tim).
* **Balanced Bagging và Easy Ensemble:**
  * Các mô hình Balanced Bagging và Easy Ensemble, cùng với Balanced Random Forest, được thiết kế để xử lý sự mất cân bằng lớp bằng cách cân bằng các lớp trong quá trình huấn luyện.
  * Hiệu suất:
    * Chúng đạt được recall cao hơn cho lớp 1, có nghĩa là chúng phát hiện ra phần lớn các trường hợp dương tính thực sự.
    * Tuy nhiên, sự đánh đổi thường là precision thấp hơn, dẫn đến điểm F1 thấp hơn.
* **Ý Nghĩa Trong Bối Cảnh Y Tế:** Trong bối cảnh y tế, recall cao là quan trọng vì cần xác định càng nhiều trường hợp dương tính thực sự càng tốt, mặc dù có thể có một số trường hợp âm tính giả. Việc bỏ sót một trường hợp dương tính (âm tính giả) có thể nghiêm trọng hơn so với việc có một âm tính giả.
* **Sử Dụng LightGBM Làm Estimator Cơ Sở:**
  * Hiệu suất với LightGBM:
    * Khi sử dụng LightGBM làm estimator cơ sở trong Balanced Bagging và Easy Ensemble, kết quả cho thấy recall cho lớp 1 đã được cải thiện.
    * Các mô hình này cũng có điểm ROC AUC hơi tốt hơn (0.885894 và 0.885778, tương ứng), cho thấy sự cân bằng tốt giữa độ nhạy và độ đặc hiệu.
    * LightGBM là một framework boosting mạnh mẽ, nổi tiếng với hiệu suất và sự hiệu quả, giúp đạt được các chỉ số hiệu suất tổng thể tốt hơn.
  * Cải Thiện:
    * Khi sử dụng Easy Ensemble như một bộ bao bọc và LightGBM làm estimator cơ sở, recall cho lớp 1 (bệnh nhân mắc bệnh tim) đã cải thiện đáng kể từ 24,4% (trong LightGBM độc lập) lên 80,7%.
    * ROC AUC cải thiện từ 88,4% lên 88,6% cho lớp 1, cho thấy một sự cân bằng tốt hơn giữa việc nhận diện đúng các dương tính thực sự và giảm thiểu các âm tính giả.
* **Ý Nghĩa Thực Tiễn:** Nhiệm Vụ Phân Loại Bệnh Tim:
  * Việc xác định bệnh nhân mắc bệnh tim (dương tính thực sự) là rất quan trọng.
  * Recall cao thường được ưu tiên hơn, mặc dù có thể có nhiều âm tính giả.
  * Recall cao đảm bảo rằng hầu hết bệnh nhân mắc bệnh tim được nhận diện, điều này rất quan trọng cho việc can thiệp và điều trị kịp thời.
  * Các âm tính giả, mặc dù không lý tưởng, có thể được quản lý thông qua xét nghiệm bổ sung và đánh giá y tế thêm.
* **Kết Luận:**
Balanced Bagging, Easy Ensemble, và Balanced Random Forest, đặc biệt khi kết hợp với LightGBM làm estimator cơ sở, cung cấp sự cân bằng tốt giữa việc nhận diện các dương tính thực sự và duy trì tỷ lệ âm tính giả hợp lý.
Đối với ứng dụng y tế như dự đoán bệnh tim, các phương pháp này đảm bảo rằng hầu hết các trường hợp bệnh tim sẽ được nhận diện, cho phép can thiệp y tế kịp thời, điều này rất quan trọng đối với chăm sóc bệnh nhân.

## **Tinh Chỉnh Siêu Tham Số Sử Dụng Optuna**

Các siêu tham số tốt nhất:
  {'n_estimators': 10,
'learning_rate': 0.1,
'boosting_type': 'gbdt',
'num_leaves': 104,
'max_depth': 10,
'min_child_samples': 24,
'subsample': 0.8437808863271848,
'colsample_bytree': 0.8,
'reg_alpha': 0,
'reg_lambda': 0.6}

## **Huấn Luyện Mô Hình Tốt Nhất - EasyEnsemble như một bộ bao bọc và LightGBM làm estimator cơ sở:**


* Lớp 0:
  * Mô hình có precision cao `(0.984788)` cho lớp 0, chỉ ra rằng khi mô hình dự đoán lớp 0, nó đúng `98.48%` thời gian.
  * Recall cho lớp 0 cũng khá cao `(0.785166)`, có nghĩa là nó nhận diện đúng `78.52%` tất cả các trường hợp lớp 0.
  * Điểm F1 `(0.873720)` cho thấy sự cân bằng tốt giữa precision và recall.
  * ROC AUC cho lớp 0 là `0.785166`, cho thấy khả năng phân biệt tốt.
* Lớp 1:
  * Precision cho lớp 1 thấp `(0.197094)`, có nghĩa là nhiều dự đoán lớp 1 thực ra là lớp 0.
  * Tuy nhiên, recall cho lớp 1 cao `(0.813019)`, có nghĩa là mô hình khá giỏi trong việc nhận diện các trường hợp lớp 1 thực sự.
  * Điểm F1 cho lớp 1 tương đối thấp `(0.317274)`, cho thấy sự đánh đổi giữa precision và recall.
  * ROC AUC cho lớp 1 giống như lớp 0 `(0.883942)`, cho thấy hiệu suất mô hình tổng thể tốt trong việc phân biệt các lớp.
    
**So Sánh với Các Mô Hình Riêng Lẻ và Kết Hợp:**
* LightGBM Alone:
  * LightGBM thường có hiệu suất mạnh mẽ nhờ khả năng boosting. Nó có thể đạt được độ chính xác cao và sự cân bằng tốt giữa precision/recall cho cả hai lớp.
  * Tuy nhiên, LightGBM một mình có thể gặp khó khăn với sự mất cân bằng lớp, thường dẫn đến recall thấp cho các lớp thiểu số (lớp 1: Recall `24.4%`).
* EasyEnsemble Alone:
  * EasyEnsemble không có LightGBM làm estimator cơ sở tập trung vào việc cân bằng dữ liệu bằng cách giảm mẫu và tạo nhiều mô hình.
  * Cách tiếp cận này cải thiện recall cho các lớp thiểu số nhưng có thể không đạt được precision cao cho lớp này.

