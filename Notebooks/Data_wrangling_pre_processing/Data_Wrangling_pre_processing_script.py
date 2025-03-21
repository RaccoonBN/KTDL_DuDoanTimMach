
# %% [markdown]
# # Phân Tích Dữ Liệu: Đánh Giá Nguy Cơ Bệnh Tim Dựa Trên AI  
# 
# * **Nhóm:** 3  

# %% [markdown]
# ## **Mục lục**<a is='Contents'></a>  
# * [Giới thiệu](#Introduction)  
# * [Bộ dữ liệu](#Dataset)  
# * [Cài đặt và các bước chuẩn bị](#Setup_and_preliminaries)  
#   * [Nhập thư viện](#Import_libraries)  
#   * [Các hàm cần thiết](#Necessary_Functions)  
# * [Trích xuất tên cột mô tả cho bộ dữ liệu](#Extracting_descriptive_column_names_for_the_dataset)  
# * [Nhập bộ dữ liệu](#Importing_dataset)  
# * [Xác thực bộ dữ liệu](#Validating_the_dataset)  
# * [Chỉnh sửa tên cột của bộ dữ liệu](#Correcting_dataset_column_names)  
# * [Các đặc điểm liên quan đến bệnh tim](#Heart_Disease_related_features)  
# * [Lựa chọn các đặc điểm liên quan đến bệnh tim](#Selection_Heart_disease_related_features)  
# * [Xử lý dữ liệu thiếu, chuyển đổi cột và kỹ thuật đặc trưng](#Imputing_missing_Data_and_transforming_columns)  
#   * [Bổ sung dữ liệu dựa trên phân phối](#Distribution_Based_Imputation)  
#   * [Cột 1: Bạn là nam hay nữ](#Column_1_Are_you_male_or_female)  
#   * [Cột 2: Đã từng được chẩn đoán mắc bệnh đau thắt ngực hoặc bệnh tim mạch vành](#Column_2_Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease)  
#   * [Cột 3: Nhóm chủng tộc được tính toán để sử dụng trong các bảng phổ biến trên Internet](#Column_3_Computed_race_groups_used_for_internet_prevalence_tables)  
#   * [Cột 4: Tuổi đã được tính toán, hợp nhất trên 80](#Column_4_Imputed_Age_value_collapsed_above_80)  
#   * [Cột 5: Tình trạng sức khỏe tổng quát](#Column_5_General_Health)  
#   * [Cột 6: Có nhà cung cấp dịch vụ chăm sóc sức khỏe cá nhân](#Column_6_Have_Personal_Health_Care_Provider)  
#   * [Cột 7: Không đủ khả năng tài chính để gặp bác sĩ](#Column_7_Could_Not_Afford_To_See_Doctor)  
#   * [Cột 8: Khoảng thời gian kể từ lần kiểm tra sức khỏe gần nhất](#Column_8_Length_of_time_since_last_routine_checkup)  
#   * [Cột 9: Đã từng được chẩn đoán mắc bệnh đau tim](#Column_9_Ever_Diagnosed_with_Heart_Attack)  
#   * [Cột 10: Đã từng được chẩn đoán bị đột quỵ](#Column_10_Ever_Diagnosed_with_a_Stroke)  
#   * [Cột 11: Đã từng được chẩn đoán mắc chứng rối loạn trầm cảm](#Column_11_Ever_told_you_had_a_depressive_disorder)  
#   * [Cột 12: Đã từng được chẩn đoán mắc bệnh thận](#Column_12_Ever_told_you_have_kidney_disease)  
#   * [Cột 13: Đã từng được chẩn đoán mắc bệnh tiểu đường](#Column_13_Ever_told_you_had_diabetes)  
#   * [Cột 14: Phân loại chỉ số khối cơ thể đã tính toán](#Column_14_Computed_body_mass_index_categories)  
#   * [Cột 15: Gặp khó khăn khi đi bộ hoặc leo cầu thang](#Column_15_Difficulty_Walking_or_Climbing_Stairs)  
#   * [Cột 16: Tình trạng sức khỏe thể chất đã tính toán](#Column_16_Computed_Physical_Health_Status)  
#   * [Cột 17: Tình trạng sức khỏe tinh thần đã tính toán](#Column_17_Computed_Mental_Health_Status)  
#   * [Cột 18: Tình trạng hen suyễn đã tính toán](#Column_18_Computed_Asthma_Status)  
#   * [Cột 19: Tập thể dục trong 30 ngày qua](#Column_19_Exercise_in_Past_30_Days)  
#   * [Cột 20: Tình trạng hút thuốc đã tính toán](#Column_20_Computed_Smoking_Status)  
#   * [Cột 21: Biến số tính toán về uống rượu say](#Column_21_Binge_Drinking_Calculated_Variable)  
#   * [Cột 22: Bạn ngủ bao nhiêu giờ mỗi ngày](#Column_22_How_Much_Time_Do_You_Sleep)  
#   * [Cột 23: Số lượng đồ uống có cồn trung bình mỗi tuần đã tính toán](#Column_23_Computed_number_of_drinks_of_alcohol_beverages_per_week)  
# * [Loại bỏ các cột không cần thiết](#Dropping_unnecessary_columns)  
# * [Xem lại cấu trúc cuối cùng của tập dữ liệu đã làm sạch](#Review_final_structure_of_the_cleaned_dataframe)  
# * [Lưu tập dữ liệu đã làm sạch](#Saving_the_cleaned_dataframe)

# %% [markdown]
# ## **Giới thiệu**<a id='Introduction'></a>
# [Contents](#Contents)
# 
# Trong notebook này, nhóm đã thực hiện một loạt các bước xử lý dữ liệu để chuẩn bị tập dữ liệu cho quá trình phân tích. Xử lý dữ liệu là một bước quan trọng trong quy trình khoa học dữ liệu, bao gồm việc chuyển đổi và ánh xạ dữ liệu thô thành một định dạng dễ sử dụng hơn.
# 
# * **Dealing with Missing Data:** Xác định và điền giá trị bị thiếu trong các cột quan trọng, chẳng hạn như cột "gender", để đảm bảo tính đầy đủ của tập dữ liệu.
# * **Data Mapping:** Chuyển đổi các biến phân loại thành các biểu diễn có ý nghĩa hơn, giúp dữ liệu dễ phân tích và diễn giải hơn.
# * **Data Cleaning:**  Loại bỏ hoặc chỉnh sửa các mục không nhất quán và sai lệch để nâng cao chất lượng dữ liệu.
# * **Feature Engineering:** Tạo ra các đặc trưng mới có thể cải thiện khả năng dự đoán của mô hình.

# %% [markdown]
# ## **Dataset**<a id='Dataset'></a>
# [Contents](#Contents)
# 
# * **Behavioral Risk Factor Surveillance System (BRFSS)** là hệ thống khảo sát qua điện thoại hàng đầu của quốc gia, thu thập dữ liệu từ cư dân Hoa Kỳ về các hành vi rủi ro liên quan đến sức khỏe, các bệnh mãn tính và việc sử dụng các dịch vụ phòng ngừa. Được thành lập vào năm 1984 với 15 tiểu bang, BRFSS hiện thu thập dữ liệu trên toàn bộ 50 tiểu bang, cũng như tại Đặc khu Columbia và ba vùng lãnh thổ của Hoa Kỳ. **CDC BRFSS** thực hiện hơn 400.000 cuộc phỏng vấn với người trưởng thành mỗi năm, khiến nó trở thành hệ thống khảo sát sức khỏe được thực hiện liên tục lớn nhất thế giới.  
# * Tập dữ liệu được lấy từ **Kaggle** [(Behavioral Risk Factor Surveillance System (BRFSS) 2022)](https://www.kaggle.com/datasets/ariaxiong/behavioral-risk-factor-surveillance-system-2022/data) và ban đầu được tải xuống từ [trang web CDC BRFSS 2022.](https://www.cdc.gov/brfss/annual_data/annual_2022.html)  

# %% [markdown]
# ## **Cài đặt và các bước chuẩn bị**<a id='Setup_and_preliminaries'></a>  
# [Quay lại Mục lục](#Contents)

# %% [markdown]
# ### Import libraries<a id='Import_libraries'></a>
# [Contents](#Contents)

# %%
# Hãy import các gói cần thiết:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
from scipy.stats import gamma, linregress
from bs4 import BeautifulSoup
import re
from fancyimpute import KNN
import dask.dataframe as dd

# Chạy đoạn mã dưới đây để tùy chỉnh hiển thị trong notebook:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Định dạng số dấu phẩy động thành 2 chữ số thập phân: 
pd.set_option('float_format', '{:.2f}'.format)

# %% [markdown]
# ### **Necessary  functions**<a id='Necessary_Functions'></a>
# [Contents](#Contents)

# %%
def summarize_df(df):
    """
    Tạo một DataFrame tóm tắt cho DataFrame đầu vào.
    
    Tham số:
    df (pd.DataFrame): DataFrame cần được tóm tắt.
    
    Trả về:
    Một DataFrame chứa các cột sau:
              - 'unique_count': Số lượng giá trị duy nhất trong mỗi cột.
              - 'data_types': Kiểu dữ liệu của từng cột.
              - 'missing_counts': Số lượng giá trị bị thiếu (NaN) trong mỗi cột.
              - 'missing_percentage': Phần trăm giá trị bị thiếu trong mỗi cột.
    """
    # Số lượng giá trị duy nhất cho mỗi cột:
    unique_counts = df.nunique()    
    # Kiểu dữ liệu của từng cột:
    data_types = df.dtypes    
    # Số lượng giá trị bị thiếu (NaN) trong mỗi cột:
    missing_counts = df.isnull().sum()    
    # Phần trăm giá trị bị thiếu trong mỗi cột:
    missing_percentage = 100 * df.isnull().mean()    
    # Kết hợp các chỉ số trên thành một DataFrame:
    summary_df = pd.concat([unique_counts, data_types, missing_counts, missing_percentage], axis=1)    
    # Đổi tên các cột để dễ đọc hơn:
    summary_df.columns = ['unique_count', 'data_types', 'missing_counts', 'missing_percentage']   
    # Trả về DataFrame tóm tắt:
    return summary_df

#-----------------------------------------------------------------------------------------------------------------#

# Hàm làm sạch và định dạng nhãn
def clean_label(label):
    # Thay thế bất kỳ ký tự nào không phải chữ cái hoặc số bằng khoảng trắng
    label = re.sub(r'[^a-zA-Z0-9\s]', '', label)
    # Thay thế khoảng trắng bằng dấu gạch dưới
    label = re.sub(r'\s+', '_', label)
    return label

#-----------------------------------------------------------------------------------------------------------------#

# Hàm để điền giá trị bị thiếu dựa trên phân phối dữ liệu
def impute_missing(row):
    if pd.isna(row['Are_you_male_or_female_3']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Are_you_male_or_female_3']

#-----------------------------------------------------------------------------------------------------------------#

# Hàm tính số lần xuất hiện và phần trăm của từng giá trị trong cột
def value_counts_with_percentage(df, column_name):
    # Tính số lần xuất hiện của từng giá trị
    counts = df[column_name].value_counts(dropna=False)
    
    # Tính phần trăm của từng giá trị
    percentages = df[column_name].value_counts(dropna=False, normalize=True) * 100
    
    # Kết hợp số lần xuất hiện và phần trăm vào một DataFrame
    result = pd.DataFrame({
        'Count': counts,
        'Percentage': percentages
    })
    
    return result


# %% [markdown]
# ## **Trích xuất tên cột mô tả cho tập dữ liệu**  
# 
# Tập dữ liệu **Behavioral Risk Factor Surveillance System (BRFSS)** có sẵn trên **Kaggle**, chứa nhiều thông tin được thu thập thông qua khảo sát. Tuy nhiên, tên các cột trong tập dữ liệu được biểu diễn bằng các nhãn ngắn hoặc mã (ví dụ: `_STATE`, `FMONTH`, `IDATE`), điều này có thể gây khó hiểu nếu không có ngữ cảnh bổ sung.  
# 
# Để đảm bảo chúng ta hiểu đầy đủ ý nghĩa của từng cột trong tập dữ liệu, điều quan trọng là phải thay thế các mã ngắn này bằng tên mô tả tương ứng. Những tên mô tả này cung cấp cái nhìn rõ ràng về loại dữ liệu mà mỗi cột chứa, giúp tập dữ liệu dễ hiểu và dễ phân tích hơn.  
# 
# ### **Tổng quan về quy trình:**  
# * **Xác định nguồn của tên mô tả:** Các tên mô tả tương ứng với các nhãn ngắn này thường được ghi lại trong [codebook in HTML](https://github.com/akthammomani/AI_powered_health_risk_assessment_app/tree/main/data_directory) hoặc **metadata** do cơ quan thu thập dữ liệu cung cấp. Trong trường hợp này, các tên mô tả được tìm thấy trong tài liệu **HTML** do **BRFSS** cung cấp.  
# * **Phân tích tài liệu HTML:** Sử dụng các kỹ thuật **web scraping**, chẳng hạn như **BeautifulSoup** trong **Python**, chúng ta có thể phân tích tài liệu **HTML** để trích xuất thông tin liên quan. Cụ thể, chúng ta sẽ tìm kiếm các bảng hoặc phần trong **HTML** liệt kê các nhãn ngắn cùng với tên mô tả tương ứng.  
# * **Khớp và thay thế:** Chúng ta tạo một **mapping** giữa nhãn ngắn và tên mô tả. Sau đó, **mapping** này được áp dụng vào tập dữ liệu để thay thế các nhãn ngắn bằng tên mô tả có ý nghĩa hơn.  
# * **Lưu tập dữ liệu đã cải tiến:** Tập dữ liệu với tên cột mô tả được lưu lại để phục vụ phân tích sau này, đảm bảo rằng tất cả người dùng có thể dễ dàng hiểu nội dung của các cột.  

# %%
# Đường dẫn đến tệp HTML:
file_path = 'USCODE22_LLCP_102523.HTML'

# Đọc tệp HTML:
with open(file_path, 'r', encoding='windows-1252') as file:
    html_content = file.read()

# Phân tích nội dung HTML bằng BeautifulSoup:
soup = BeautifulSoup(html_content, 'html.parser')

# Tìm tất cả các bảng chứa thông tin cần thiết:
tables = soup.find_all('table', class_='table')

# Khởi tạo danh sách để lưu trữ dữ liệu trích xuất:
labels = []
sas_variable_names = []

# Lặp qua từng bảng để trích xuất 'Label' và 'SAS Variable Name':
for table in tables:
    # Tìm tất cả các phần tử 'td' trong bảng:
    cells = table.find_all('td', class_='l m linecontent')
    
    # Lặp qua từng ô để tìm 'Label' và 'SAS Variable Name':
    for cell in cells:
        text = cell.get_text(separator="\n")
        label = None
        sas_variable_name = None
        for line in text.split('\n'):
            if line.strip().startswith('Label:'):
                label = line.split('Label:')[1].strip()
            elif line.strip().startswith('SAS\xa0Variable\xa0Name:'):
                sas_variable_name = line.split('SAS\xa0Variable\xa0Name:')[1].strip()
        if label and sas_variable_name:
            labels.append(label)
            sas_variable_names.append(sas_variable_name)
        else:
            print("Không tìm thấy Label hoặc SAS Variable Name trong văn bản:")
            print(text)

# Tạo DataFrame:
data = {'SAS Variable Name': sas_variable_names, 'Label': labels}
cols_df = pd.DataFrame(data)

# Lưu DataFrame vào tệp CSV:
output_file_path = 'extracted_data.csv'
cols_df.to_csv(output_file_path, index=False)

print(f"Dữ liệu đã được trích xuất và lưu thành công vào {output_file_path}")

cols_df.head()


# %%
# Chạy đoạn mã dưới đây để kiểm tra lại từng đặc trưng, bao gồm số lượng & phần trăm dữ liệu bị thiếu, số giá trị duy nhất, kiểu dữ liệu:

summarize_df(cols_df)

# %% [markdown]
# No Missing Data - looks like we have 324 columns 

# %% [markdown]
# ## **Nhập tập dữ liệu**<a id='Importing_dataset'></a>  
# [Contents](#Contents)

# %%
# Đầu tiên, hãy tải tập dữ liệu chính BRFSS 2022:
df = pd.read_csv('brfss2022.csv')

# %% [markdown]
# ## **Xác thực tập dữ liệu**<a id='Validating_the_dataset'></a>  
# [Contents](#Contents)

# %%
# Bây giờ, hãy xem 5 hàng đầu tiên của df:
df.head()

# %%
# Bây giờ, hãy xem kích thước của df:

shape = df.shape
print("Number of rows:", shape[0], "\nNumber of columns:", shape[1])

# %% [markdown]
# ## **Chỉnh sửa tên cột trong tập dữ liệu**<a id='Correcting_dataset_column_names'></a>  
# [Contents](#Contents)  
# 
# Để thay thế **SAS Variable Names** trong tập dữ liệu của bạn bằng các **labels** tương ứng (trong đó khoảng trắng trong labels được thay thế bằng dấu gạch dưới **"_"**), bạn có thể thực hiện các bước sau:  
# 
# * Tạo một **mapping** từ **SAS Variable Names** sang **labels** đã được chỉnh sửa.  
# * Sử dụng **mapping** này để đổi tên các cột trong tập dữ liệu.

# %%
# Hàm để làm sạch và định dạng nhãn
def clean_label(label):
    # Thay thế bất kỳ ký tự nào không phải chữ cái hoặc số bằng rỗng
    label = re.sub(r'[^a-zA-Z0-9\s]', '', label)
    # Thay thế khoảng trắng bằng dấu gạch dưới
    label = re.sub(r'\s+', '_', label)
    return label

# Tạo một dictionary để ánh xạ từ SAS Variable Names sang Labels đã làm sạch
mapping = {row['SAS Variable Name']: clean_label(row['Label']) for _, row in cols_df.iterrows()}

# In dictionary ánh xạ để kiểm tra các thay đổi
# print("Ánh xạ đổi tên cột:")
# for k, v in mapping.items():
#     print(f"{k}: {v}")

# Đổi tên các cột trong DataFrame thực tế
df.rename(columns=mapping, inplace=True)
df.head()


# %% [markdown]
# 
# 
# ## **Các đặc trưng liên quan đến bệnh tim**<a id='Heart_Disease_related_features'></a>  
# [Contents](#Contents)  
# 
# Sau nhiều ngày nghiên cứu và phân tích các đặc trưng của tập dữ liệu, chúng tôi đã xác định các đặc trưng quan trọng sau để đánh giá bệnh tim:  
# 
# * **Biến mục tiêu (Biến phụ thuộc):**  
#     * **Heart_disease**: Đã từng được chẩn đoán mắc chứng đau thắt ngực hoặc bệnh tim mạch vành  
# 
# * **Nhân khẩu học:**  
#     * **Gender**: Giới tính (Nam hay Nữ)  
#     * **Race**: Nhóm chủng tộc được tính toán dùng cho bảng tỷ lệ phổ biến trên internet  
#     * **Age**: Tuổi được ước tính, nhóm tuổi trên 80 được gộp chung  
# 
# * **Tiền sử y tế:**  
#     * General_Health: Tình trạng sức khỏe tổng quát  
#     * Have_Personal_Health_Care_Provider: Có bác sĩ chăm sóc sức khỏe cá nhân  
#     * Could_Not_Afford_To_See_Doctor: Không có khả năng tài chính để đi khám bác sĩ  
#     * Length_of_time_since_last_routine_checkup: Khoảng thời gian từ lần kiểm tra sức khỏe định kỳ gần nhất  
#     * Ever_Diagnosed_with_Heart_Attack: Đã từng được chẩn đoán bị đau tim  
#     * Ever_Diagnosed_with_a_Stroke: Đã từng được chẩn đoán bị đột quỵ  
#     * Ever_told_you_had_a_depressive_disorder: Đã từng được chẩn đoán rối loạn trầm cảm  
#     * Ever_told_you_have_kidney_disease: Đã từng được chẩn đoán mắc bệnh thận  
#     * Ever_told_you_had_diabetes: Đã từng được chẩn đoán mắc bệnh tiểu đường  
#     * Reported_Weight_in_Pounds: Cân nặng (đơn vị: pound)  
#     * Reported_Height_in_Feet_and_Inches: Chiều cao (đơn vị: feet và inch)  
#     * Computed_body_mass_index_categories: Phân loại chỉ số khối cơ thể (BMI)  
#     * Difficulty_Walking_or_Climbing_Stairs: Gặp khó khăn khi đi bộ hoặc leo cầu thang  
#     * Computed_Physical_Health_Status: Tình trạng sức khỏe thể chất được tính toán  
#     * Computed_Mental_Health_Status: Tình trạng sức khỏe tâm thần được tính toán  
#     * Computed_Asthma_Status: Tình trạng hen suyễn được tính toán  
# 
# * **Lối sống:**  
#     * Leisure_Time_Physical_Activity_Calculated_Variable: Mức độ hoạt động thể chất trong thời gian rảnh  
#     * Smoked_at_Least_100_Cigarettes: Đã từng hút ít nhất 100 điếu thuốc trong đời  
#     * Computed_Smoking_Status: Tình trạng hút thuốc được tính toán  
#     * Binge_Drinking_Calculated_Variable: Tình trạng uống rượu bia quá mức được tính toán  
#     * Computed_number_of_drinks_of_alcohol_beverages_per_week: Số lượng đồ uống có cồn tiêu thụ trung bình mỗi tuần  
#     * Exercise_in_Past_30_Days: Đã tập thể dục trong 30 ngày qua  
#     * How_Much_Time_Do_You_Sleep: Thời gian ngủ trung bình mỗi ngày  
# 

# %% [markdown]
# ## **Lựa chọn các đặc trưng liên quan đến bệnh tim**<a id='Selection_Heart_disease_related_features'></a>  
# [Contents](#Contents)

# %%
# Ở đây, chúng ta sẽ chọn các đặc trưng chính liên quan trực tiếp đến bệnh tim:
df = df[["Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease", # Biến mục tiêu
         "Are_you_male_or_female", # Nhân khẩu học
         "Computed_race_groups_used_for_internet_prevalence_tables", # Nhân khẩu học
         "Imputed_Age_value_collapsed_above_80", # Nhân khẩu học
         "General_Health", # Tiền sử y tế
         "Have_Personal_Health_Care_Provider", # Tiền sử y tế
         "Could_Not_Afford_To_See_Doctor", # Tiền sử y tế
         "Length_of_time_since_last_routine_checkup", # Tiền sử y tế
         "Ever_Diagnosed_with_Heart_Attack", # Tiền sử y tế
         "Ever_Diagnosed_with_a_Stroke", # Tiền sử y tế
         "Ever_told_you_had_a_depressive_disorder", # Tiền sử y tế
         "Ever_told_you_have_kidney_disease", # Tiền sử y tế
         "Ever_told_you_had_diabetes", # Tiền sử y tế
         "Reported_Weight_in_Pounds", # Tiền sử y tế
         "Reported_Height_in_Feet_and_Inches", # Tiền sử y tế
         "Computed_body_mass_index_categories", # Tiền sử y tế
         "Difficulty_Walking_or_Climbing_Stairs", # Tiền sử y tế
         "Computed_Physical_Health_Status", # Tiền sử y tế
         "Computed_Mental_Health_Status", # Tiền sử y tế
         "Computed_Asthma_Status", # Tiền sử y tế
         "Leisure_Time_Physical_Activity_Calculated_Variable", # Lối sống
         "Smoked_at_Least_100_Cigarettes", # Lối sống
         "Computed_Smoking_Status", # Lối sống
         "Binge_Drinking_Calculated_Variable", # Lối sống
         "Computed_number_of_drinks_of_alcohol_beverages_per_week", # Lối sống
         "Exercise_in_Past_30_Days", # Lối sống
         "How_Much_Time_Do_You_Sleep" # Lối sống
        ]]
df.head()


# %%
# Bây giờ, hãy xem kích thước của df sau khi chọn đặc trưng:
shape = df.shape
print("Số hàng:", shape[0], "\nSố cột:", shape[1])


# %% [markdown]
# ## **Điền dữ liệu bị thiếu, chuyển đổi cột và kỹ thuật tạo đặc trưng**<a id='Imputing_missing_Data_and_transforming_columns'></a>  
# [Contents](#Contents)  
# 
# Trong bước này, chúng ta xử lý dữ liệu bị thiếu, ánh xạ các giá trị phân loại và đổi tên cột để cải thiện chất lượng và độ rõ ràng của dữ liệu. Các hành động chính được thực hiện như sau:  
# 
# * **Thay thế các giá trị cụ thể bằng NaN:** Xác định và thay thế các giá trị sai hoặc giá trị giữ chỗ bằng NaN để chuẩn hóa cách biểu diễn dữ liệu bị thiếu.  
# * **Tính toán phân bố giá trị:** Xác định phân bố của các giá trị hiện có để hiểu trạng thái cơ bản của dữ liệu.  
# * **Điền dữ liệu bị thiếu:** Sử dụng một hàm để điền giá trị bị thiếu dựa trên phân bố đã tính toán, đảm bảo dữ liệu vẫn phản ánh đúng đặc điểm ban đầu.  
# * **Ánh xạ các giá trị phân loại:** Áp dụng từ điển ánh xạ để chuyển đổi các mã số thành nhãn phân loại có ý nghĩa.  
# * **Đổi tên cột:** Cập nhật tên cột để phản ánh chính xác nội dung của chúng và giúp dữ liệu dễ đọc hơn.  
# * **Kỹ thuật tạo đặc trưng:** Tạo các đặc trưng mới có thể cải thiện khả năng dự đoán của mô hình.  
# 
# Những bước này rất quan trọng để xây dựng một mô hình dự đoán bệnh tim chính xác và đáng tin cậy.

# %% [markdown]
# ### **Điền dữ liệu bị thiếu dựa trên phân bố**<a id='Distribution_Based_Imputation'></a>  
# [Contents](#Contents)  
# 
# Để xử lý dữ liệu bị thiếu trong dự án này, chúng ta sẽ sử dụng phương pháp **Điền dữ liệu bị thiếu dựa trên phân bố**:  
# 
# * **Giới thiệu**  
#     * **Dựa trên phân bố:** Quá trình điền dữ liệu dựa trên phân bố hiện có của các danh mục trong tập dữ liệu.  
#     * **Điền dữ liệu bị thiếu:** Hành động thay thế các giá trị bị thiếu bằng dữ liệu phù hợp.  
# 
# * **Tại sao phương pháp này hiệu quả?**  
#     * **Bảo toàn phân bố gốc:** Bằng cách sử dụng tỷ lệ quan sát được để hướng dẫn việc điền dữ liệu, phương pháp này giữ nguyên phân bố ban đầu của các danh mục.  
#     * **Điền dữ liệu ngẫu nhiên:** Việc chọn giá trị ngẫu nhiên dựa trên phân bố hiện có giúp tránh sai lệch hệ thống có thể xảy ra khi sử dụng phương pháp điền dữ liệu cố định.  
#     * **Khả năng mở rộng:** Phương pháp này có thể dễ dàng mở rộng cho các tập dữ liệu lớn hơn và áp dụng cho các biến phân loại khác có giá trị bị thiếu.  
# 
# * **Ưu điểm**  
#     * **Giảm sai lệch:** Đảm bảo rằng các giá trị được điền không làm lệch tập dữ liệu theo hướng có lợi cho một danh mục cụ thể.  
#     * **Đơn giản:** Dễ hiểu và triển khai.  
#     * **Linh hoạt:** Có thể áp dụng cho bất kỳ biến phân loại nào có giá trị bị thiếu.  
# 
# Phương pháp này đặc biệt hữu ích trong các tình huống cần bảo toàn phân bố tự nhiên của dữ liệu để phục vụ cho việc phân tích hoặc xây dựng mô hình sau này.

# %%
#Hãy chạy đoạn mã dưới đây để kiểm tra lại từng đặc trưng, bao gồm:
summarize_df(df)

# %% [markdown]
# ### **Cột 1: Are_you_male_or_female**  
# [Contents](#Contents)  
# 
# Chúng ta có 4 phiên bản của cùng một cột, vì vậy bây giờ hãy giữ lại cột có ít dữ liệu bị thiếu nhất (`21.58%`).

# %%
# Lấy danh sách tên các cột trong DataFrame:
print(df.columns)

# %%
# Chọn các đặc trưng chính liên quan đến bệnh tim:
df.columns = ['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease', # Đây là biến mục tiêu!!!
       'Are_you_male_or_female_1', 'Are_you_male_or_female_2',
       'Are_you_male_or_female_3', 'Are_you_male_or_female_4',
       'Computed_race_groups_used_for_internet_prevalence_tables',
       'Imputed_Age_value_collapsed_above_80', 'General_Health',
       'Have_Personal_Health_Care_Provider', 'Could_Not_Afford_To_See_Doctor',
       'Length_of_time_since_last_routine_checkup',
       'Ever_Diagnosed_with_Heart_Attack', 'Ever_Diagnosed_with_a_Stroke',
       'Ever_told_you_had_a_depressive_disorder',
       'Ever_told_you_have_kidney_disease', 'Ever_told_you_had_diabetes',
       'Reported_Weight_in_Pounds', 'Reported_Height_in_Feet_and_Inches',
       'Computed_body_mass_index_categories',
       'Difficulty_Walking_or_Climbing_Stairs',
       'Computed_Physical_Health_Status', 'Computed_Mental_Health_Status',
       'Computed_Asthma_Status',
       'Leisure_Time_Physical_Activity_Calculated_Variable',
       'Smoked_at_Least_100_Cigarettes', 'Computed_Smoking_Status',
       'Binge_Drinking_Calculated_Variable',
       'Computed_number_of_drinks_of_alcohol_beverages_per_week',
       'Exercise_in_Past_30_Days', 'How_Much_Time_Do_You_Sleep']

# Chạy lệnh dưới đây để kiểm tra lại từng đặc trưng, bao gồm số lượng dữ liệu thiếu & tỷ lệ phần trăm, số lượng giá trị duy nhất, kiểu dữ liệu:
summarize_df(df)


# %% [markdown]
# Alright, as we can see above, now, let's drop 'Are_you_male_or_female_1', 'Are_you_male_or_female_2' and 'Are_you_male_or_female_4'

# %%
# Loại bỏ các cột không cần thiết:
columns_to_drop = ['Are_you_male_or_female_1', 'Are_you_male_or_female_2', 'Are_you_male_or_female_4']
df = df.drop(columns=columns_to_drop)

# Chạy lệnh dưới đây để kiểm tra lại từng đặc trưng, bao gồm số lượng dữ liệu thiếu & tỷ lệ phần trăm, số lượng giá trị duy nhất, kiểu dữ liệu:
summarize_df(df)


# %%
# Xem số lượng cột:
df.Are_you_male_or_female_3.value_counts(dropna=False)

# %% [markdown]
# **Are_you_male_or_female_3:**
# * 2: Femal
# * 1: Male
# * 3: Nonbinary
# * 7: Don’t know/Not Sure
# * 9: Refused
# 
# So based on above, let's change 7 and 9 to nan

# %%
# Thay thế giá trị 7 và 9 bằng NaN
df['Are_you_male_or_female_3'].replace([7, 9], np.nan, inplace=True)
df.Are_you_male_or_female_3.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có
value_counts = df['Are_you_male_or_female_3'].value_counts(normalize=True, dropna=True)
print("Original distribution:\n", value_counts)

# %%
# Hàm để điền giá trị thiếu dựa trên phân bố hiện có
def impute_missing_gender(row):
    if pd.isna(row['Are_you_male_or_female_3']):  # Nếu giá trị bị thiếu (NaN)
        return np.random.choice(value_counts.index, p=value_counts.values)  # Chọn ngẫu nhiên dựa trên phân bố hiện có
    else:
        return row['Are_you_male_or_female_3']  # Nếu không thiếu, giữ nguyên giá trị

# Áp dụng hàm điền giá trị thiếu cho cột 'Are_you_male_or_female_3'
df['Are_you_male_or_female_3'] = df.apply(impute_missing_gender, axis=1)


# %%
# Xác minh việc điền giá trị thiếu
imputed_value_counts = df['Are_you_male_or_female_3'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)


# %% [markdown]
# Được rồi, như chúng ta có thể thấy ở trên, không có dữ liệu bị thiếu trong cột này và tỷ lệ được giữ nguyên (Phép nội suy ngẫu nhiên đã hoạt động như mong đợi).

# %%
# Tạo một từ điển ánh xạ:
gender_mapping = {2: 'female', 1: 'male', 3: 'nonbinary'}

# Áp dụng ánh xạ cho cột giới tính:
df['Are_you_male_or_female_3'] = df['Are_you_male_or_female_3'].map(gender_mapping)

# Đổi tên cột:
df.rename(columns={'Are_you_male_or_female_3': 'gender'}, inplace=True)

df.head()  # Hiển thị 5 dòng đầu tiên của DataFrame


# %%
# Chạy đoạn mã dưới đây để kiểm tra lại từng đặc trưng, bao gồm số lượng dữ liệu thiếu & phần trăm, số lượng giá trị duy nhất, kiểu dữ liệu:
summarize_df(df)

# %% [markdown]
# ### **Column 2: Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease**<a id='Column_2_Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'></a>
# [Contents](#Contents)

# %%
# Xem số lượng cột:

df.Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease.value_counts(dropna=False)

# %% [markdown]
# **Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease:**
# * 2: No
# * 1: Yes
# * 7: Don’t know/Not Sure
# * 9: Refused
# 
# Được rồi, tiếp theo, hãy thay đổi 7 và 9 thành NaN:

# %%
# Thay thế giá trị 7 và 9 bằng NaN

df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].replace([7, 9], np.nan, inplace=True)
df.Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease.value_counts(dropna=False)

# %% [markdown]
# Được rồi, một lần nữa, hãy sử dụng **Distribution-Based Imputation** cho dữ liệu bị thiếu ở trên:

# %%
# Tính toán phân phối của các giá trị hiện có

value_counts = df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].value_counts(normalize=True, dropna=True)
print("Original distribution:\n", value_counts)

# %%
# Hàm để điền giá trị thiếu dựa trên phân phối hiện có
def impute_missing(row):
    if pd.isna(row['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease']

# %%
# Áp dụng hàm điền giá trị thiếu
df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu
imputed_value_counts = df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu

imputed_value_counts = df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].value_counts(dropna=False, normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ:
heart_disease_mapping = {2: 'no', 1: 'yes'}

# Áp dụng ánh xạ vào cột "Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease":
df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'] = df['Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease'].map(heart_disease_mapping)

# Đổi tên cột:
df.rename(columns={'Ever_Diagnosed_with_Angina_or_Coronary_Heart_Disease': 'heart_disease'}, inplace=True)


# %%
# Chạy đoạn mã dưới đây để kiểm tra lại từng đặc trưng, bao gồm số lượng giá trị bị thiếu và phần trăm, số lượng giá trị duy nhất, kiểu dữ liệu:
summarize_df(df)


# %% [markdown]
# ### **Column 3: Computed_race_groups_used_for_internet_prevalence_tables**<a id='Column_3_Computed_race_groups_used_for_internet_prevalence_tables'></a>
# [Contents](#Contents)

# %%
# Xem số lg cộtcột
df.Computed_race_groups_used_for_internet_prevalence_tables.value_counts(dropna=False)

# %% [markdown]
# Alright, so good news is there's no missing data in this column

# %% [markdown]
# **Computed_race_groups_used_for_internet_prevalence_tables:**
# * 1: white_only_non_hispanic
# * 2: black_only_non_hispanic
# * 3: american_indian_or_alaskan_native_only_non_hispanic
# * 4: asian_only_non_hispanic
# * 5: native_hawaiian_or_other_pacific_islander_only_non_hispanic
# * 6: multiracial_non_hispanic
# * 7: hispanic
# 

# %%
# Tạo một từ điển ánh xạ:
race_mapping = {1: 'white_only_non_hispanic',
2: 'black_only_non_hispanic',
3: 'american_indian_or_alaskan_native_only_non_hispanic',
4: 'asian_only_non_hispanic',
5: 'native_hawaiian_or_other_pacific_islander_only_non_hispanic',
6: 'multiracial_non_hispanic',
7: 'hispanic'}

# Áp dụng ánh xạ cho cột chủng tộc:
df['Computed_race_groups_used_for_internet_prevalence_tables'] = df['Computed_race_groups_used_for_internet_prevalence_tables'].map(race_mapping)

# Đổi tên cột:
df.rename(columns={'Computed_race_groups_used_for_internet_prevalence_tables': 'race'}, inplace=True)


# %%
# xem số lg cộtcột
df.race.value_counts(dropna=False)

# %% [markdown]
# ### **column 4: Imputed_Age_value_collapsed_above_80**<a id='Column_4_Imputed_Age_value_collapsed_above_80'></a>
# [Contents](#Contents)

# %%
#view column counts:
df.Imputed_Age_value_collapsed_above_80.value_counts(dropna=False)

# %%
# Định nghĩa các khoảng giá trị (bins) và nhãn tương ứng (labels)
bins = [17, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 99]
labels = [
    'Age_18_to_24', 'Age_25_to_29', 'Age_30_to_34', 'Age_35_to_39',
    'Age_40_to_44', 'Age_45_to_49', 'Age_50_to_54', 'Age_55_to_59',
    'Age_60_to_64', 'Age_65_to_69', 'Age_70_to_74', 'Age_75_to_79',
    'Age_80_or_older'
         ]

# %%
# Phân loại các giá trị tuổi vào các khoảng:
df['age_category'] = pd.cut(df['Imputed_Age_value_collapsed_above_80'], bins=bins, labels=labels, right=True)
df.age_category.value_counts(dropna=False)

# %% [markdown]
# ### **Cột 5: Sức khỏe tổng quát (General_Health)**<a id='Column_5_General_Health'></a>  
# [Danh mục nội dung](#Contents)

# %%
#view column counts:
df.General_Health.value_counts(dropna=False)

# %% [markdown]
# **General_Health:**
# * 1: excellent
# * 2: very_good
# * 3: good
# * 4: fair
# * 5: poor
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:
# 

# %%
# Thay thế giá trị 7 và 9 bằng NaN
df['General_Health'].replace([7, 9], np.nan, inplace=True)
df.General_Health.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có
value_counts = df['General_Health'].value_counts(normalize=True, dropna=True)
print("Original General_Health:\n", value_counts)

# %%
# Hàm để điền giá trị khuyết dựa trên phân bố dữ liệu
def impute_missing(row):
    if pd.isna(row['General_Health']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['General_Health']

# %%
# Áp dụng hàm điền giá trị khuyết
df['General_Health'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị khuyết
imputed_value_counts = df['General_Health'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ cho tình trạng sức khỏe
health_mapping = {
    1: 'excellent',
    2: 'very_good',
    3: 'good',
    4: 'fair',
    5: 'poor'
}

# Áp dụng ánh xạ cho cột General_Health
df['General_Health'] = df['General_Health'].map(health_mapping)

# Đổi tên cột General_Health thành general_health để dễ đọc
df.rename(columns={'General_Health': 'general_health'}, inplace=True)


# %%
#view column counts:
df.general_health.value_counts(dropna=False)

# %% [markdown]
# ### **Cột 6: Có nhà cung cấp dịch vụ chăm sóc sức khỏe cá nhân (Have_Personal_Health_Care_Provider)**<a id='Column_6_Have_Personal_Health_Care_Provider'></a>  
# [Danh mục nội dung](#Contents)

# %%
#view column counts:
df.Have_Personal_Health_Care_Provider.value_counts(dropna=False)

# %% [markdown]
# **Have_Personal_Health_Care_Provider:**
# * 1: yes_only_one
# * 2: more_than_one
# * 3: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# %%
# Thay thế giá trị 7 và 9 bằng NaN

df['Have_Personal_Health_Care_Provider'].replace([7, 9], np.nan, inplace=True)
df.Have_Personal_Health_Care_Provider.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có

value_counts = df['Have_Personal_Health_Care_Provider'].value_counts(normalize=True, dropna=True)
print("Original Have_Personal_Health_Care_Provider:\n", value_counts)

# %%
# Hàm để điền giá trị thiếu dựa trên phân bố dữ liệu

def impute_missing(row):
    if pd.isna(row['Have_Personal_Health_Care_Provider']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Have_Personal_Health_Care_Provider']

# %%
# Apply the imputation function
df['Have_Personal_Health_Care_Provider'] = df.apply(impute_missing, axis=1)

# %%
# Áp dụng hàm điền giá trị thiếu

imputed_value_counts = df['Have_Personal_Health_Care_Provider'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu
imputed_value_counts = df['Have_Personal_Health_Care_Provider'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ:
porvider_mapping = {1: 'yes_only_one',
                  2: 'more_than_one',
                  3: 'no'
                 }

# Áp dụng ánh xạ cho cột nhà cung cấp dịch vụ y tế:
df['Have_Personal_Health_Care_Provider'] = df['Have_Personal_Health_Care_Provider'].map(porvider_mapping)

# Đổi tên cột:
df.rename(columns={'Have_Personal_Health_Care_Provider': 'health_care_provider'}, inplace=True)


# %% [markdown]
# ### **Cột 7: Không có khả năng chi trả để đi khám bác sĩ**<a id='Column_7_Could_Not_Afford_To_See_Doctor'></a>  
# [Danh mục nội dung](#Contents)

# %%
#view column counts:
df.Could_Not_Afford_To_See_Doctor.value_counts(dropna=False)

# %% [markdown]
# **Could_Not_Afford_To_See_Doctor:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# %%
# Replace 7 and 9 with NaN
df['Could_Not_Afford_To_See_Doctor'].replace([7, 9], np.nan, inplace=True)
df.Could_Not_Afford_To_See_Doctor.value_counts(dropna=False)

# %%
# Calculate the distribution of existing values
value_counts = df['Could_Not_Afford_To_See_Doctor'].value_counts(normalize=True, dropna=True)
print("Original Could_Not_Afford_To_See_Doctor:\n", value_counts)

# %%
# Function to impute missing values based on distribution
def impute_missing(row):
    if pd.isna(row['Could_Not_Afford_To_See_Doctor']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Could_Not_Afford_To_See_Doctor']

# %%
# Apply the imputation function
df['Could_Not_Afford_To_See_Doctor'] = df.apply(impute_missing, axis=1)

# %%
# Verify the imputation
imputed_value_counts = df['Could_Not_Afford_To_See_Doctor'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Verify the imputation
imputed_value_counts = df['Could_Not_Afford_To_See_Doctor'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Create a mapping dictionary:
doctor_mapping = {1: 'yes',
                  2: 'no'
                 }

# Apply the mapping to the doctor column:
df['Could_Not_Afford_To_See_Doctor'] = df['Could_Not_Afford_To_See_Doctor'].map(doctor_mapping)

# Rename the column:
df.rename(columns={'Could_Not_Afford_To_See_Doctor': 'could_not_afford_to_see_doctor'}, inplace=True)

# %% [markdown]
# ### **Column 8: Length_of_time_since_last_routine_checkup**<a id='Column_8_Length_of_time_since_last_routine_checkup'></a>
# [Contents](#Contents)

# %%
#view column counts:
df.Length_of_time_since_last_routine_checkup.value_counts(dropna=False)

# %% [markdown]
# **Could_Not_Afford_To_See_Doctor:**
# * 1: 'past_year',
# * 2: 'past_2_years',
# * 3: 'past_5_years',
# * 4: '5+_years_ago',
# * 7: 'dont_know',
# * 8: 'never',
# * 9: 'refused',
# so for 7, 9 let's convert to nan:

# %%
#Replace 7 and 9 with NaN:
df['Length_of_time_since_last_routine_checkup'].replace([7, 9], np.nan, inplace=True)
df.Length_of_time_since_last_routine_checkup.value_counts(dropna=False)

# %%
# Calculate the distribution of existing values:
value_counts = df['Length_of_time_since_last_routine_checkup'].value_counts(normalize=True, dropna=True)
print("Original Length_of_time_since_last_routine_checkup:\n", value_counts)

# %%
# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Length_of_time_since_last_routine_checkup']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Length_of_time_since_last_routine_checkup']

# %%
# Apply the imputation function:
df['Length_of_time_since_last_routine_checkup'] = df.apply(impute_missing, axis=1)

# %%
# Verify the imputation:
imputed_value_counts = df['Length_of_time_since_last_routine_checkup'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Verify the imputation:
imputed_value_counts = df['Length_of_time_since_last_routine_checkup'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Create a mapping dictionary:
checkup_mapping = {1: 'past_year',
                   2: 'past_2_years',
                   3: 'past_5_years',
                   4: '5+_years_ago',
                   8: 'never',
                 }

# Apply the mapping to the checkup_mapping column:
df['Length_of_time_since_last_routine_checkup'] = df['Length_of_time_since_last_routine_checkup'].map(checkup_mapping)

# Rename the column:
df.rename(columns={'Length_of_time_since_last_routine_checkup': 'length_of_time_since_last_routine_checkup'}, inplace=True)

# %%
#view column counts:
df['length_of_time_since_last_routine_checkup'].value_counts(dropna=False,normalize=True)

# %% [markdown]
# ### **Column 9: Ever_Diagnosed_with_Heart_Attack**<a id='Column_9_Ever_Diagnosed_with_Heart_Attack'></a>
# [Contents](#Contents)

# %%
#view column counts:
df['Ever_Diagnosed_with_Heart_Attack'].value_counts(dropna=False)

# %% [markdown]
# **Ever_Diagnosed_with_Heart_Attack:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# %%
#Replace 7 and 9 with NaN:
df['Ever_Diagnosed_with_Heart_Attack'].replace([7, 9], np.nan, inplace=True)
df.Ever_Diagnosed_with_Heart_Attack.value_counts(dropna=False)

# %%
# Calculate the distribution of existing values:
value_counts = df['Ever_Diagnosed_with_Heart_Attack'].value_counts(normalize=True, dropna=True)
print("Original Length_of_time_since_last_routine_checkup:\n", value_counts)

# %%
# Function to impute missing values based on distribution:
def impute_missing(row):
    if pd.isna(row['Ever_Diagnosed_with_Heart_Attack']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_Diagnosed_with_Heart_Attack']

# %%
# Apply the imputation function:
df['Ever_Diagnosed_with_Heart_Attack'] = df.apply(impute_missing, axis=1)

# %%
# Verify the imputation:
imputed_value_counts = df['Ever_Diagnosed_with_Heart_Attack'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Verify the imputation:
imputed_value_counts = df['Ever_Diagnosed_with_Heart_Attack'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Create a mapping dictionary:
heart_attack_mapping = {1: 'yes',
                   2: 'no',

                 }

# Apply the mapping to the heart_attack_mapping column:
df['Ever_Diagnosed_with_Heart_Attack'] = df['Ever_Diagnosed_with_Heart_Attack'].map(heart_attack_mapping)

# Rename the column:
df.rename(columns={'Ever_Diagnosed_with_Heart_Attack': 'ever_diagnosed_with_heart_attack'}, inplace=True)

# %%
#view column counts:
df['ever_diagnosed_with_heart_attack'].value_counts(dropna=False,normalize=True) # 

# %% [markdown]
# ### **Column 10: Ever_Diagnosed_with_a_Stroke**<a id='Column_10_Ever_Diagnosed_with_a_Stroke'></a>
# [Contents](#Contents)

# %%
#view column counts:
df['Ever_Diagnosed_with_a_Stroke'].value_counts(dropna=False)

# %% [markdown]
# **Ever_Diagnosed_with_Heart_Attack:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# %%
#Replace 7 and 9 with NaN:
df['Ever_Diagnosed_with_a_Stroke'].replace([7, 9], np.nan, inplace=True)
df.Ever_Diagnosed_with_a_Stroke.value_counts(dropna=False)

# %%
#Tính toán phân bố của các giá trị hiện có:
value_counts = df['Ever_Diagnosed_with_a_Stroke'].value_counts(normalize=True, dropna=True)
print("Original Ever_Diagnosed_with_a_Stroke:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Ever_Diagnosed_with_a_Stroke']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_Diagnosed_with_a_Stroke']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Ever_Diagnosed_with_a_Stroke'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Ever_Diagnosed_with_a_Stroke'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Ever_Diagnosed_with_a_Stroke'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ:
stroke_mapping = {1: 'yes',
                   2: 'no',

                 }

# Áp dụng ánh xạ cho cột stroke:
df['Ever_Diagnosed_with_a_Stroke'] = df['Ever_Diagnosed_with_a_Stroke'].map(stroke_mapping)

# Rename the column:
df.rename(columns={'Ever_Diagnosed_with_a_Stroke': 'ever_diagnosed_with_a_stroke'}, inplace=True)

# %%
# Xem số lượng cột:
df['ever_diagnosed_with_a_stroke'].value_counts(dropna=False,normalize=True) # 

# %% [markdown]
# ### **Column 11: Ever_told_you_had_a_depressive_disorder**<a id='Column_11_Ever_told_you_had_a_depressive_disorder'></a>
# [Contents](#Contents)

# %%
# Xem số lượng giá trị trong từng cột:
value_counts_with_percentage(df, 'Ever_told_you_had_a_depressive_disorder')

# %% [markdown]
# **Ever_told_you_had_a_depressive_disorder:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# %%
# Thay thế 7 và 9 bằng NaN:
df['Ever_told_you_had_a_depressive_disorder'].replace([7, 9], np.nan, inplace=True)
df.Ever_told_you_had_a_depressive_disorder.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Ever_told_you_had_a_depressive_disorder'].value_counts(normalize=True, dropna=True)
print("Original Ever_told_you_had_a_depressive_disorder:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Ever_told_you_had_a_depressive_disorder']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_told_you_had_a_depressive_disorder']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Ever_told_you_had_a_depressive_disorder'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Ever_told_you_had_a_depressive_disorder'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Ever_told_you_had_a_depressive_disorder'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ
depressive_disorder_mapping = {1: 'yes', 2: 'no'}

# Áp dụng ánh xạ cho cột 'Ever_told_you_had_a_depressive_disorder'
df['Ever_told_you_had_a_depressive_disorder'] = df['Ever_told_you_had_a_depressive_disorder'].map(depressive_disorder_mapping)

# Đổi tên cột thành 'ever_told_you_had_a_depressive_disorder'
df.rename(columns={'Ever_told_you_had_a_depressive_disorder': 'ever_told_you_had_a_depressive_disorder'}, inplace=True)


# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'ever_told_you_had_a_depressive_disorder')

# %% [markdown]
# ### **Column 12: Ever_told_you_have_kidney_disease**<a id='Column_12_Ever_told_you_have_kidney_disease'></a>
# [Contents](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Ever_told_you_have_kidney_disease')

# %% [markdown]
# **Ever_told_you_had_a_depressive_disorder:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# %%
# Thay thế 7 và 9 bằng NaN:
df['Ever_told_you_have_kidney_disease'].replace([7, 9], np.nan, inplace=True)
df.Ever_told_you_have_kidney_disease.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Ever_told_you_have_kidney_disease'].value_counts(normalize=True, dropna=True)
print("Original Ever_told_you_have_kidney_disease:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Ever_told_you_have_kidney_disease']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_told_you_have_kidney_disease']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Ever_told_you_have_kidney_disease'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Ever_told_you_have_kidney_disease'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Ever_told_you_have_kidney_disease'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ
depressive_disorder_mapping = {1: 'yes', 2: 'no'}

# Áp dụng ánh xạ cho cột 'Ever_told_you_had_a_depressive_disorder'
df['Ever_told_you_had_a_depressive_disorder'] = df['Ever_told_you_had_a_depressive_disorder'].map(depressive_disorder_mapping)

# Đổi tên cột thành 'ever_told_you_had_a_depressive_disorder'
df.rename(columns={'Ever_told_you_had_a_depressive_disorder': 'ever_told_you_had_a_depressive_disorder'}, inplace=True)


# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'ever_told_you_have_kidney_disease')

# %% [markdown]
# ### **Column 13: Ever_told_you_had_diabetes**<a id='Column_13_Ever_told_you_had_diabetes'></a>
# [Contents](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Ever_told_you_had_diabetes')

# %% [markdown]
# **Ever_told_you_had_diabetes:**
# * 1: 'yes',
# * 2: 'yes_during_pregnancy',
# * 3: 'no',
# * 4: 'no_prediabetes',
# * 7: 'dont_know',
# * 9: 'refused',
# 
# so for 7, 9 let's convert to nan:

# %%
# Thay thế 7 và 9 bằng NaN:
df['Ever_told_you_had_diabetes'].replace([7, 9], np.nan, inplace=True)
df.Ever_told_you_had_diabetes.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Ever_told_you_had_diabetes'].value_counts(normalize=True, dropna=True)
print("Original Ever_told_you_have_kidney_disease:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Ever_told_you_had_diabetes']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Ever_told_you_had_diabetes']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Ever_told_you_had_diabetes'] = df.apply(impute_missing, axis=1)

# %%
# Áp dụng hàm điền giá trị thiếu:
imputed_value_counts = df['Ever_told_you_had_diabetes'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Ever_told_you_had_diabetes'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ  
diabetes_mapping = {1: 'yes',  
                  2: 'yes_during_pregnancy',  
                  3: 'no',  
                  4: 'no_prediabetes',  

                 }  

# Áp dụng ánh xạ cho cột diabetes  
df['Ever_told_you_had_diabetes'] = df['Ever_told_you_had_diabetes'].map(diabetes_mapping)  

# Đổi tên cột  
df.rename(columns={'Ever_told_you_had_diabetes': 'ever_told_you_had_diabetes'}, inplace=True)  


# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'ever_told_you_had_diabetes')

# %% [markdown]
# ### **Cột 14: Danh mục chỉ số khối cơ thể đã tính toán**<a id='Column_14_Computed_body_mass_index_categories'></a>  
# [ Nội dung ](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Computed_body_mass_index_categories')

# %% [markdown]
# **Computed_body_mass_index_categories:**
# * 1: 'underweight_bmi_less_than_18_5',
# * 2: 'normal_weight_bmi_18_5_to_24_9',
# * 3: 'overweight_bmi_25_to_29_9',
# * 4: 'obese_bmi_30_or_more',
# 

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Computed_body_mass_index_categories'].value_counts(normalize=True, dropna=True)
print("Original Computed_body_mass_index_categories:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Computed_body_mass_index_categories']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_body_mass_index_categories']

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
df['Computed_body_mass_index_categories'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:

imputed_value_counts = df['Computed_body_mass_index_categories'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Computed_body_mass_index_categories'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ  
bmi_mapping = {1: 'underweight_bmi_less_than_18_5',  
                    2: 'normal_weight_bmi_18_5_to_24_9',  
                    3: 'overweight_bmi_25_to_29_9',  
                    4: 'obese_bmi_30_or_more',  
                 }  

# Áp dụng ánh xạ cho cột BMI  
df['Computed_body_mass_index_categories'] = df['Computed_body_mass_index_categories'].map(bmi_mapping)  

# Đổi tên cột  
df.rename(columns={'Computed_body_mass_index_categories': 'BMI'}, inplace=True)  


# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'BMI')

# %% [markdown]
# ### **Column 15: Difficulty_Walking_or_Climbing_Stairs**<a id='Column_15_Difficulty_Walking_or_Climbing_Stairs'></a>
# [Contents](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Difficulty_Walking_or_Climbing_Stairs')

# %% [markdown]
# **Difficulty_Walking_or_Climbing_Stairs:**
# * 1: yes
# * 2: no
# * 7: dont_know
# * 9: refused
# 
# so for 7, 9 let's convert to nan:

# %%
# Thay thế 7 và 9 bằng NaN:
df['Difficulty_Walking_or_Climbing_Stairs'].replace([7, 9], np.nan, inplace=True)
df.Difficulty_Walking_or_Climbing_Stairs.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Difficulty_Walking_or_Climbing_Stairs'].value_counts(normalize=True, dropna=True)
print("Original Difficulty_Walking_or_Climbing_Stairs:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Difficulty_Walking_or_Climbing_Stairs']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Difficulty_Walking_or_Climbing_Stairs']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Difficulty_Walking_or_Climbing_Stairs'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Difficulty_Walking_or_Climbing_Stairs'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Difficulty_Walking_or_Climbing_Stairs'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ  
climbing_mapping = {1: 'yes',  
                   2: 'no',  
                 }  

# Áp dụng ánh xạ cho cột Difficulty_Walking_or_Climbing_Stairs  
df['Difficulty_Walking_or_Climbing_Stairs'] = df['Difficulty_Walking_or_Climbing_Stairs'].map(climbing_mapping)  

# Đổi tên cột  
df.rename(columns={'Difficulty_Walking_or_Climbing_Stairs': 'difficulty_walking_or_climbing_stairs'}, inplace=True)  


# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'difficulty_walking_or_climbing_stairs')

# %% [markdown]
# ### **Cột 16: Trạng thái sức khỏe thể chất đã tính toán**<a id='Column_16_Computed_Physical_Health_Status'></a>  
# [ Nội dung ](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Computed_Physical_Health_Status')

# %% [markdown]
# **Computed_Physical_Health_Status:**
# * 1: 'zero_days_not_good',
# * 2: '1_to_13_days_not_good',
# * 3: '14_plus_days_not_good',
# * 9: 'dont_know'
# 
# so for 9 let's convert to nan:

# %%
#thay đổi 7 thành 99
df['Computed_Physical_Health_Status'].replace([9], np.nan, inplace=True)
df.Computed_Physical_Health_Status.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Computed_Physical_Health_Status'].value_counts(normalize=True, dropna=True)
print("Original Computed_Physical_Health_Status:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Computed_Physical_Health_Status']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_Physical_Health_Status']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Computed_Physical_Health_Status'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Computed_Physical_Health_Status'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Computed_Physical_Health_Status'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ  
health_status_mapping = {1: 'zero_days_not_good',  
                    2: '1_to_13_days_not_good',  
                    3: '14_plus_days_not_good',  
                 }  

# Áp dụng ánh xạ cho cột Computed_Physical_Health_Status  
df['Computed_Physical_Health_Status'] = df['Computed_Physical_Health_Status'].map(health_status_mapping)  

# Đổi tên cột  
df.rename(columns={'Computed_Physical_Health_Status': 'physical_health_status'}, inplace=True)  


# %%
# Xem số lượng và phần trăm giá trị trong từng cột::
value_counts_with_percentage(df, 'physical_health_status')

# %% [markdown]
# ### **Cột 17: Trạng thái sức khỏe tinh thần đã tính toán**<a id='Column_17_Computed_Mental_Health_Status'></a>  
# [ Nội dung ](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Computed_Mental_Health_Status')

# %% [markdown]
# **Computed_Physical_Health_Status:**
# * 1: 'zero_days_not_good',
# * 2: '1_to_13_days_not_good',
# * 3: '14_plus_days_not_good',
# * 9: 'dont_know'
# 
# so for 9 let's convert to nan:

# %%
# Thay thế 7 và 9 bằng NaN:
df['Computed_Mental_Health_Status'].replace([9], np.nan, inplace=True)
df.Computed_Mental_Health_Status.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Computed_Mental_Health_Status'].value_counts(normalize=True, dropna=True)
print("Original Computed_Mental_Health_Status:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Computed_Mental_Health_Status']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_Mental_Health_Status']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Computed_Mental_Health_Status'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Computed_Mental_Health_Status'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Verify the imputation:
imputed_value_counts = df['Computed_Mental_Health_Status'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ  
m_health_status_mapping = {1: 'zero_days_not_good',  # 0 ngày cảm thấy không tốt  
                    2: '1_to_13_days_not_good',  # 1 đến 13 ngày cảm thấy không tốt  
                    3: '14_plus_days_not_good',  # 14 ngày trở lên cảm thấy không tốt  
                 }  

# Áp dụng ánh xạ cho cột Computed_Mental_Health_Status  
df['Computed_Mental_Health_Status'] = df['Computed_Mental_Health_Status'].map(m_health_status_mapping)  

# Đổi tên cột  
df.rename(columns={'Computed_Mental_Health_Status': 'mental_health_status'}, inplace=True)  


# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'mental_health_status')

# %% [markdown]
# ### **Cột 18: Trạng thái hen suyễn đã tính toán**<a id='Column_18_Computed_Asthma_Status'></a>  
# [ Nội dung ](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Computed_Asthma_Status')

# %% [markdown]
# **Computed_Asthma_Status:**
# * 1: 'current_asthma',
# * 2: 'former_asthma',
# * 3: 'never_asthma',
# * 9: 'dont_know_refused_missing'
# 
# so for 9 let's convert to nan:

# %%
#thay đổi 7 thành 9 
df['Computed_Asthma_Status'].replace([9], np.nan, inplace=True)
df.Computed_Asthma_Status.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Computed_Asthma_Status'].value_counts(normalize=True, dropna=True)
print("Original Computed_Asthma_Status:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Computed_Asthma_Status']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_Asthma_Status']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Computed_Asthma_Status'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Computed_Asthma_Status'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Computed_Asthma_Status'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ  
Asthma_Status_mapping = {1: 'current_asthma',
                           2: 'former_asthma',
                           3: 'never_asthma',

                 }
# Áp dụng ánh xạ cho cột Computed_Mental_Health_Status 
df['Computed_Asthma_Status'] = df['Computed_Asthma_Status'].map(Asthma_Status_mapping)

# Đổi tên cột  
df.rename(columns={'Computed_Asthma_Status': 'asthma_Status'}, inplace=True)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'asthma_Status')

# %% [markdown]
# ### **Cột 19: Tập thể dục trong 30 ngày qua**<a id='Column_19_Exercise_in_Past_30_Days'></a>  
# [ Nội dung ](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Exercise_in_Past_30_Days')

# %% [markdown]
# **Exercise_in_Past_30_Days:**
# * 1: 'yes',
# * 2: 'no',
# * 7: 'dont_know'
# * 9: 'refused_missing'
# 
# so for 7, 9 let's convert to nan:

# %%
# Thay thế 7 và 9 bằng NaN:
df['Exercise_in_Past_30_Days'].replace([7, 9], np.nan, inplace=True)
df.Exercise_in_Past_30_Days.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Exercise_in_Past_30_Days'].value_counts(normalize=True, dropna=True)
print("Original Exercise_in_Past_30_Days:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Exercise_in_Past_30_Days']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Exercise_in_Past_30_Days']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Exercise_in_Past_30_Days'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Exercise_in_Past_30_Days'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Exercise_in_Past_30_Days'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ  
exercise_Status_mapping = {1: 'yes',
                           2: 'no',

                 }

# Áp dụng ánh xạ cho cột Computed_Mental_Health_Status  
df['Exercise_in_Past_30_Days'] = df['Exercise_in_Past_30_Days'].map(exercise_Status_mapping)


# Đổi tên cột  
df.rename(columns={'Exercise_in_Past_30_Days': 'exercise_status_in_past_30_Days'}, inplace=True)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'exercise_status_in_past_30_Days')

# %% [markdown]
# ### **Cột 20: Trạng thái hút thuốc đã tính toán**<a id='Column_20_Computed_Smoking_Status'></a>  
# [ Nội dung ](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Computed_Smoking_Status')

# %% [markdown]
# **Computed_Smoking_Status:**
# * 1: 'current_smoker_every_day',
# * 2: 'current_smoker_some_days',
# * 3: 'former_smoker',
# * 4: 'never_smoked',
# * 9: 'dont_know_refused_missing'
# 
# so for 9 let's convert to nan:

# %%
# Thay thế 7 và 9 bằng NaN:
df['Computed_Smoking_Status'].replace([9], np.nan, inplace=True)
df.Computed_Smoking_Status.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Computed_Smoking_Status'].value_counts(normalize=True, dropna=True)
print("Original Computed_Smoking_Status:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Computed_Smoking_Status']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Computed_Smoking_Status']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Computed_Smoking_Status'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Computed_Smoking_Status'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Computed_Smoking_Status'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ 
smoking_Status_mapping = {1: 'current_smoker_every_day',
                           2: 'current_smoker_some_days',
                           3: 'former_smoker',
                           4: 'never_smoked'
                          }


# Áp dụng ánh xạ cho cột Computed_Mental_Health_Status 
df['Computed_Smoking_Status'] = df['Computed_Smoking_Status'].map(smoking_Status_mapping)

# Đổi tên cột 
df.rename(columns={'Computed_Smoking_Status': 'smoking_status'}, inplace=True)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'smoking_status')

# %% [markdown]
# ### **Cột 21: Biến số tính toán về uống rượu quá mức**<a id='Column_21_Binge_Drinking_Calculated_Variable'></a>  
# [ Nội dung ](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'Binge_Drinking_Calculated_Variable')

# %% [markdown]
# **Binge_Drinking_Calculated_Variable:**
# * 1: 'no',
# * 2: 'yes',
# * 9: 'dont_know_refused_missing'
# 
# so for 9 let's convert to nan:

# %%
# Thay thế 7 và 9 bằng NaN:
df['Binge_Drinking_Calculated_Variable'].replace([9], np.nan, inplace=True)
df.Binge_Drinking_Calculated_Variable.value_counts(dropna=False)

# %%
# Tính toán phân bố của các giá trị hiện có:
value_counts = df['Binge_Drinking_Calculated_Variable'].value_counts(normalize=True, dropna=True)
print("Original Binge_Drinking_Calculated_Variable:\n", value_counts)

# %%
# Hàm điền giá trị thiếu dựa trên phân bố:
def impute_missing(row):
    if pd.isna(row['Binge_Drinking_Calculated_Variable']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['Binge_Drinking_Calculated_Variable']

# %%
# Áp dụng hàm điền giá trị thiếu:
df['Binge_Drinking_Calculated_Variable'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Binge_Drinking_Calculated_Variable'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc điền giá trị thiếu:
imputed_value_counts = df['Binge_Drinking_Calculated_Variable'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Tạo một từ điển ánh xạ 
binge_drinking_status = {1: 'no',
                           2: 'yes'
                          }

# Áp dụng ánh xạ cho cột Computed_Mental_Health_Status 
df['Binge_Drinking_Calculated_Variable'] = df['Binge_Drinking_Calculated_Variable'].map(binge_drinking_status)

# Đổi tên cột  
df.rename(columns={'Binge_Drinking_Calculated_Variable': 'binge_drinking_status'}, inplace=True)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'binge_drinking_status')

# %% [markdown]
# ### **Column 22: How_Much_Time_Do_You_Sleep**<a id='Column_22_How_Much_Time_Do_You_Sleep'></a>
# [Contents](#Contents)

# %%
# Xem số lượng và phần trăm giá trị trong từng cột:
value_counts_with_percentage(df, 'How_Much_Time_Do_You_Sleep')

# %%
def categorize_sleep_hours(df, column_name):  
    # Định nghĩa từ điển ánh xạ cho các giá trị đã biết  
    sleep_mapping = {  
        77: 'dont_know',  
        99: 'refused_to_answer',  
        np.nan: 'missing'  
    }  

    # Phân loại số giờ ngủ  
    for hour in range(0, 4):  
        sleep_mapping[hour] = 'very_short_sleep_0_to_3_hours'  
    for hour in range(4, 6):  
        sleep_mapping[hour] = 'short_sleep_4_to_5_hours'  
    for hour in range(6, 9):  
        sleep_mapping[hour] = 'normal_sleep_6_to_8_hours'  
    for hour in range(9, 11):  
        sleep_mapping[hour] = 'long_sleep_9_to_10_hours'  
    for hour in range(11, 25):  
        sleep_mapping[hour] = 'very_long_sleep_11_or_more_hours'  

    # Áp dụng ánh xạ giá trị vào các danh mục tương ứng  
    df['sleep_category'] = df[column_name].map(sleep_mapping)  

    return df  


# %%
#Dùng hàm để phân loại giờ ngủ.
df = categorize_sleep_hours(df, 'How_Much_Time_Do_You_Sleep')

# %%
#Xem số lượng cột và tỷ lệ phần trăm.
value_counts_with_percentage(df, 'sleep_category')

# %%
# Thay thế giá trị 7 và 9 bằng NaN:
#df['sleep_category'].replace([7, 9], np.nan, inplace=True)

df['sleep_category'].replace(['missing', 'dont_know','refused_to_answer'], np.nan, inplace=True)
df.sleep_category.value_counts(dropna=False)

# %%
# Tính phân phối của các giá trị hiện có:
value_counts = df['sleep_category'].value_counts(normalize=True, dropna=True)
print("Original sleep_category:\n", value_counts)

# %%
# Hàm để ước lượng các giá trị thiếu dựa trên phân phối:
def impute_missing(row):
    if pd.isna(row['sleep_category']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['sleep_category']

# %%
# Áp dụng hàm ước lượng giá trị thiếu:
df['sleep_category'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc ước lượng giá trị thiếu:
imputed_value_counts = df['sleep_category'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc ước lượng giá trị thiếu:
imputed_value_counts = df['sleep_category'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xem số lượng cột và tỷ lệ phần trăm:
value_counts_with_percentage(df, 'sleep_category')

# %% [markdown]
# ### **Cột 23: Computed_number_of_drinks_of_alcohol_beverages_per_week**<a id='Column_23_Computed_number_of_drinks_of_alcohol_beverages_per_week'></a>
# [Contents](#Contents)

# %%
# Xem số lượng cột và tỷ lệ phần trăm:
value_counts_with_percentage(df, 'Computed_number_of_drinks_of_alcohol_beverages_per_week')

# %%
# Chia cho 100 để có được số lượng đồ uống mỗi tuần
df['drinks_per_week'] = df['Computed_number_of_drinks_of_alcohol_beverages_per_week'] / 100

# %%
# Định nghĩa hàm để phân loại mức độ tiêu thụ đồ uống
def categorize_drinks(drinks_per_week):
    #if drinks_per_week == 0:
        #return 'did_not_drink'
    if drinks_per_week == 99900 / 100:
        return 'do_not_know'
    elif 0.01 <= drinks_per_week <= 1:
        return 'very_low_consumption_0.01_to_1_drinks'
    elif 1.01 <= drinks_per_week <= 5:
        return 'low_consumption_1.01_to_5_drinks'
    elif 5.01 <= drinks_per_week <= 10:
        return 'moderate_consumption_5.01_to_10_drinks'
    elif 10.01 <= drinks_per_week <= 20:
        return 'high_consumption_10.01_to_20_drinks'
    elif drinks_per_week > 20:
        return 'very_high_consumption_more_than_20_drinks'
    else:
        return 'did_not_drink'

# %%
# Áp dụng hàm phân loại mức độ tiêu thụ đồ uống
df['drinks_category'] = df['drinks_per_week'].apply(categorize_drinks)


# %%
# Xem số lượng cột và tỷ lệ phần trăm:
value_counts_with_percentage(df, 'drinks_category')

# %%
# Thay thế giá trị 7 và 9 bằng NaN:
df['drinks_category'].replace(['do_not_know'], np.nan, inplace=True)
df.drinks_category.value_counts(dropna=False)

# %%
# Tính phân phối của các giá trị hiện có:# Tính phân phối của các giá trị hiện có:
value_counts = df['drinks_category'].value_counts(normalize=True, dropna=True)
print("Original drinks_category:\n", value_counts)

# %%
# Hàm để ước lượng giá trị thiếu dựa trên phân phối:
def impute_missing(row):
    if pd.isna(row['drinks_category']):
        return np.random.choice(value_counts.index, p=value_counts.values)
    else:
        return row['drinks_category']

# %%
# Áp dụng hàm ước lượng giá trị thiếu:
df['drinks_category'] = df.apply(impute_missing, axis=1)

# %%
# Xác minh việc ước lượng giá trị thiếu:
imputed_value_counts = df['drinks_category'].value_counts(dropna=False) # normalize=True
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Xác minh việc ước lượng giá trị thiếu:
imputed_value_counts = df['drinks_category'].value_counts(dropna=False,normalize=True) # 
print("Distribution after imputation:\n", imputed_value_counts)

# %%
# Kiểm tra cuối cùng sau khi ước lượng giá trị thiếu:
value_counts_with_percentage(df, 'drinks_category')

# %% [markdown]
# ## **Xóa các cột không cần thiết**<a id='Dropping_unnecessary_columns'></a>  
# [Contents](#Contents)

# %%
# Ở đây, chúng ta sẽ xóa các cột không cần thiết:
columns_to_drop = ['Imputed_Age_value_collapsed_above_80', 'Reported_Weight_in_Pounds', 
                   'Reported_Height_in_Feet_and_Inches', 'Leisure_Time_Physical_Activity_Calculated_Variable',
                  'Smoked_at_Least_100_Cigarettes', 'Computed_number_of_drinks_of_alcohol_beverages_per_week',
                  'How_Much_Time_Do_You_Sleep', 'drinks_per_week']
df = df.drop(columns=columns_to_drop)

# %% [markdown]
# ## **Review the final structre of the cleaned dataframe**<a id='Review_final_structure_of_the_cleaned_dataframe'></a>
# [Contents](#Contents)

# %%
# Bây giờ, hãy xem hình dạng của df:
shape = df.shape
print("Number of rows:", shape[0], "\nNumber of columns:", shape[1])

# %%
summarize_df(df)

# %% [markdown]
#Tuyệt vời, không có dữ liệu bị thiếu. **Như chúng ta có thể thấy ở trên, chúng ta đã làm sạch dữ liệu, loại bỏ dữ liệu bị thiếu và vẫn giữ nguyên kích thước của tập dữ liệu "số hàng".**
# %% [markdown]
# ## **Lưu DataFrame đã làm sạch**<a id='Saving_the_cleaned_dataframe'></a>  
# [Contents](#Contents)

# %%
output_file_path = "./brfss2022_data_wrangling_output.csv"

df.to_csv(output_file_path, index=False)


