import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
import io
from lightgbm import LGBMClassifier
import category_encoders as ce
from imblearn.ensemble import EasyEnsembleClassifier
import shap
import plotly.express as px

with open('best_model.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

with open('cbe_encoder.pkl', 'rb') as encoder_file:
    encoder = pkl.load(encoder_file)

data = pd.read_csv('brfss2022_data_wrangling_output.zip', compression='zip')
data['heart_disease'] = data['heart_disease'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')

icon = Image.open("heart_disease.jpg")
st.set_page_config(layout='wide', page_title='DỰ ĐOÁN NGUY CƠ MẮC BỆNH TIM BẰNG AI', page_icon=icon)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style_v1.css")

row0_0, row0_1, row0_2, row0_3 = st.columns((0.08, 6, 3, 0.17))
with row0_1:
    st.title("Ứng dụng Đánh giá Nguy cơ Bệnh Tim Mạch với Trí Tuệ Nhân Tạo")
st.write('---')

# Ánh xạ nhóm tuổi
age_mapping = {
    "Age_18_to_24": "18 đến 24 tuổi",
    "Age_25_to_29": "25 đến 29 tuổi",
    "Age_30_to_34": "30 đến 34 tuổi",
    "Age_35_to_39": "35 đến 39 tuổi",
    "Age_40_to_44": "40 đến 44 tuổi",
    "Age_45_to_49": "45 đến 49 tuổi",
    "Age_50_to_54": "50 đến 54 tuổi",
    "Age_55_to_59": "55 đến 59 tuổi",
    "Age_60_to_64": "60 đến 64 tuổi",
    "Age_65_to_69": "65 đến 69 tuổi",
    "Age_70_to_74": "70 đến 74 tuổi",
    "Age_75_to_79": "75 đến 79 tuổi",
    "Age_80_or_older": "80 tuổi trở lên"
}
# Ánh xạ giới tính
gender_mapping = {
    "female": "Nữ",
    "male": "Nam",
    "nonbinary": "Không xác định"
}

# Ánh xạ dân tộc
race_mapping = {
    "white_only_non_hispanic": "Châu Âu (Không phải người Hispano)",
    "black_only_non_hispanic": "Châu Phi (Không phải người Hispano)",
    "asian_only_non_hispanic": "Châu Á (Không phải người Hispano)",
    "american_indian_or_alaskan_native_only_non_hispanic": "Người Mỹ bản xứ hoặc người Alaskan (Không phải người Hispano)",
    "multiracial_non_hispanic": "Đa chủng tộc (Không phải người Hispano)",
    "hispanic": "Người Hispano",
    "native_hawaiian_or_other_pacific_islander_only_non_hispanic": "Người Hawaii bản xứ hoặc người đảo Thái Bình Dương (Không phải người Hispano)"
}

# Ánh xạ các giá trị Medical History
general_health_mapping = {
    "excellent": "Tuyệt vời",
    "very_good": "Rất tốt",
    "good": "Tốt",
    "fair": "Hài lòng",
    "poor": "Kém"
}

heart_attack_mapping = {
    "yes": "Có",
    "no": "Không"
}

kidney_disease_mapping = {
    "yes": "Có",
    "no": "Không"
}

asthma_mapping = {
    "never_asthma": "Chưa từng mắc",
    "current_asthma": "Đang mắc",
    "former_asthma": "Đã từng mắc"
}

could_not_afford_to_see_doctor_mapping = {
    "yes": "Có",
    "no": "Không"
}

health_care_provider_mapping = {
    "yes_only_one": "Có một bác sĩ chính",
    "more_than_one": "Có nhiều bác sĩ",
    "no": "Không có"
}

stroke_mapping = {
    "yes": "Có",
    "no": "Không"
}

diabetes_mapping = {
    "yes": "Có",
    "no": "Không",
    "no_prediabetes": "Không, tiền tiểu đường",
    "yes_during_pregnancy": "Có trong thai kỳ"
}

bmi_mapping = {
    "underweight_bmi_less_than_18_5": "Thiếu cân (BMI < 18.5)",
    "normal_weight_bmi_18_5_to_24_9": "Cân nặng bình thường (BMI từ 18.5 đến 24.9)",
    "overweight_bmi_25_to_29_9": "Thừa cân (BMI từ 25 đến 29.9)",
    "obese_bmi_30_or_more": "Béo phì (BMI >= 30)"
}

length_of_time_since_last_routine_checkup_mapping = {
    "past_year": "Trong năm qua",
    "past_2_years": "Trong 2 năm qua",
    "past_5_years": "Trong 5 năm qua",
    "5+_years_ago": "Hơn 5 năm",
    "never": "Chưa bao giờ"
}

depressive_disorder_mapping = {
    "yes": "Có",
    "no": "Không"
}

physical_health_mapping = {
    "zero_days_not_good": "Không có ngày nào không tốt",
    "1_to_13_days_not_good": "Từ 1 đến 13 ngày không tốt",
    "14_plus_days_not_good": "Hơn 14 ngày không tốt"
}

mental_health_mapping = {
    "zero_days_not_good": "Không có ngày nào không tốt",
    "1_to_13_days_not_good": "Từ 1 đến 13 ngày không tốt",
    "14_plus_days_not_good": "Hơn 14 ngày không tốt"
}

walking_mapping = {
    "yes": "Có",
    "no": "Không"
}

smoking_status_mapping = {
    "never_smoked": "Chưa bao giờ hút thuốc",
    "former_smoker": "Đã từng hút thuốc",
    "current_smoker_some_days": "Hút thuốc vài ngày",
    "current_smoker_every_day": "Hút thuốc mỗi ngày"
}

sleep_category_mapping = {
    "very_short_sleep_0_to_3_hours": "Ngủ rất ít (0-3 giờ)",
    "short_sleep_4_to_5_hours": "Ngủ ít (4-5 giờ)",
    "normal_sleep_6_to_8_hours": "Ngủ đủ (6-8 giờ)",
    "long_sleep_9_to_10_hours": "Ngủ dài (9-10 giờ)",
    "very_long_sleep_11_or_more_hours": "Ngủ rất dài (11 giờ trở lên)"
}

drinks_category_mapping = {
    "did_not_drink": "Không uống rượu",
    "very_low_consumption_0.01_to_1_drinks": "Uống rất ít (0.01 đến 1 ly)",
    "low_consumption_1.01_to_5_drinks": "Uống ít (1.01 đến 5 ly)",
    "moderate_consumption_5.01_to_10_drinks": "Uống vừa phải (5.01 đến 10 ly)",
    "high_consumption_10.01_to_20_drinks": "Uống nhiều (10.01 đến 20 ly)",
    "very_high_consumption_more_than_20_drinks": "Uống rất nhiều (hơn 20 ly)"
}

binge_drinking_status_mapping = {
    "yes": "Có",
    "no": "Không"
}

exercise_status_mapping = {
    "yes": "Có",
    "no": "Không"
}

# Giới thiệu và cách hoạt động 
st.markdown("""
<div class="flex-container">
    <div class="flex-item introduction">
        <h2>Giới Thiệu</h2>
        <p>Ứng dụng Đánh giá Nguy cơ Bệnh Tim Mạch sử dụng Trí Tuệ Nhân Tạo cung cấp cho người dùng các điểm số nguy cơ cá nhân hóa và các khuyến nghị hành động để giúp giảm thiểu nguy cơ mắc bệnh tim mạch. Bằng cách sử dụng các mô hình AI tiên tiến, ứng dụng này mang đến các đánh giá dễ hiểu và các biện pháp phòng ngừa để bảo vệ sức khỏe tim mạch của bạn một cách đơn giản và dễ dàng.</p>
    </div>
    <div class="flex-item how-it-works">
        <h2>Cách Hoạt Động</h2>
        <ul>
            <li><strong>Thông tin người dùng:</strong> Nhập thông tin sức khỏe của bạn, chẳng hạn như tuổi tác, chỉ số BMI, mức độ hoạt động thể chất, tình trạng hút thuốc và tiền sử bệnh (ví dụ: nhồi máu cơ tim, đột quỵ, tiểu đường).</li>
            <li><strong>Phân tích Dữ Liệu:</strong> Ứng dụng sẽ phân tích thông tin bạn cung cấp bằng các mô hình AI chuyên biệt để dự đoán nguy cơ bệnh tim mạch.</li>
            <li><strong>Đánh Giá Nguy Cơ:</strong> Nhận được điểm số nguy cơ cá nhân cho thấy khả năng mắc bệnh tim mạch của bạn.</li>
            <li><strong>Khuyến Nghị:</strong> Nhận các lời khuyên hành động để giảm thiểu nguy cơ, bao gồm các đề xuất thay đổi lối sống.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

st.write('---')
# Giao diện nhập liệu
row1_0, row1_1, row1_2, row1_3, row1_5 = st.columns((0.08, 3, 3, 3, 0.17))
with row1_1:
    st.write("#### Thông tin nhân khẩu học")

row2_0, row2_1, row2_2, row2_3, row2_5 = st.columns((0.08, 3, 3, 3, 0.17))

# Câu hỏi về giới tính
gender = row2_1.selectbox("Giới tính của bạn là gì?", list(gender_mapping.values()), index=1)
selected_gender = list(gender_mapping.keys())[list(gender_mapping.values()).index(gender)]

# Câu hỏi về dân tộc
race = row2_2.selectbox("Dân tộc của bạn là gì?", list(race_mapping.values()), index=0)
selected_race = list(race_mapping.keys())[list(race_mapping.values()).index(race)]

# Câu hỏi về độ tuổi với ánh xạ tiếng Việt
age_category = row2_3.selectbox("Nhóm tuổi của bạn là?", list(age_mapping.values()), index=4)
selected_age = list(age_mapping.keys())[list(age_mapping.values()).index(age_category)]


row3_0, row3_1, row3_2, row3_3, row3_5 = st.columns((0.08, 3, 3, 3, 0.17))
with row3_1:
    st.write("#### Tiền sử y tế")

row4_0, row4_1, row4_2, row4_3, row4_5 = st.columns((0.08, 3, 3, 3, 0.17))

# Câu hỏi về tình trạng sức khỏe tổng quát
general_health = row4_1.selectbox("Bạn đánh giá sức khỏe tổng quát của mình như thế nào?", list(general_health_mapping.values()), index=0)
selected_general_health = list(general_health_mapping.keys())[list(general_health_mapping.values()).index(general_health)]

# Câu hỏi về nhồi máu cơ tim
heart_attack = row4_1.selectbox("Bạn đã bao giờ được chẩn đoán mắc nhồi máu cơ tim?", list(heart_attack_mapping.values()), index=1)
selected_heart_attack = list(heart_attack_mapping.keys())[list(heart_attack_mapping.values()).index(heart_attack)]

# Câu hỏi về bệnh thận
kidney_disease = row4_1.selectbox("Bác sĩ đã bao giờ nói với bạn rằng bạn mắc bệnh thận?", list(kidney_disease_mapping.values()), index=1)
selected_kidney_disease = list(kidney_disease_mapping.keys())[list(kidney_disease_mapping.values()).index(kidney_disease)]

# Câu hỏi về hen suyễn
asthma = row4_1.selectbox("Bạn đã bao giờ được chẩn đoán mắc hen suyễn?", list(asthma_mapping.values()), index=0)
selected_asthma = list(asthma_mapping.keys())[list(asthma_mapping.values()).index(asthma)]

# Câu hỏi về khả năng tài chính để khám bệnh
could_not_afford_to_see_doctor = row4_1.selectbox("Bạn đã bao giờ không thể đi khám bác sĩ khi cần do chi phí?", list(could_not_afford_to_see_doctor_mapping.values()), index=1)
selected_could_not_afford_to_see_doctor = list(could_not_afford_to_see_doctor_mapping.keys())[list(could_not_afford_to_see_doctor_mapping.values()).index(could_not_afford_to_see_doctor)]

# Câu hỏi về bác sĩ chăm sóc chính
health_care_provider = row4_2.selectbox("Bạn có bác sĩ chăm sóc chính không?", list(health_care_provider_mapping.values()), index=0)
selected_health_care_provider = list(health_care_provider_mapping.keys())[list(health_care_provider_mapping.values()).index(health_care_provider)]

# Câu hỏi về đột quỵ
stroke = row4_2.selectbox("Bạn đã bao giờ được chẩn đoán mắc đột quỵ?", list(stroke_mapping.values()), index=1)
selected_stroke = list(stroke_mapping.keys())[list(stroke_mapping.values()).index(stroke)]

# Câu hỏi về bệnh tiểu đường
diabetes = row4_2.selectbox("Bạn đã bao giờ được chẩn đoán mắc bệnh tiểu đường?", list(diabetes_mapping.values()), index=1)
selected_diabetes = list(diabetes_mapping.keys())[list(diabetes_mapping.values()).index(diabetes)]

# Câu hỏi về chỉ số BMI
bmi = row4_2.selectbox("Chỉ số BMI của bạn là bao nhiêu?", list(bmi_mapping.values()), index=1)
selected_bmi = list(bmi_mapping.keys())[list(bmi_mapping.values()).index(bmi)]

# Câu hỏi về kiểm tra sức khỏe định kỳ
length_of_time_since_last_routine_checkup = row4_2.selectbox("Kể từ lần kiểm tra sức khỏe định kỳ gần nhất, đã bao lâu?", list(length_of_time_since_last_routine_checkup_mapping.values()), index=0)
selected_length_of_time_since_last_routine_checkup = list(length_of_time_since_last_routine_checkup_mapping.keys())[list(length_of_time_since_last_routine_checkup_mapping.values()).index(length_of_time_since_last_routine_checkup)]

# Câu hỏi về rối loạn trầm cảm
depressive_disorder = row4_3.selectbox("Bác sĩ đã bao giờ nói với bạn rằng bạn mắc rối loạn trầm cảm?", list(depressive_disorder_mapping.values()), index=1)
selected_depressive_disorder = list(depressive_disorder_mapping.keys())[list(depressive_disorder_mapping.values()).index(depressive_disorder)]

# Câu hỏi về sức khỏe thể chất
physical_health = row4_3.selectbox("Trong 30 ngày qua, bạn có bao nhiêu ngày không khỏe về thể chất?", list(physical_health_mapping.values()), index=0)
selected_physical_health = list(physical_health_mapping.keys())[list(physical_health_mapping.values()).index(physical_health)]

# Câu hỏi về sức khỏe tinh thần
mental_health = row4_3.selectbox("Trong 30 ngày qua, bạn có bao nhiêu ngày không khỏe về tinh thần?", list(mental_health_mapping.values()), index=0)
selected_mental_health = list(mental_health_mapping.keys())[list(mental_health_mapping.values()).index(mental_health)]

# Câu hỏi về khả năng đi lại
walking = row4_3.selectbox("Bạn có gặp khó khăn khi đi bộ hoặc leo cầu thang không?", list(walking_mapping.values()), index=1)
selected_walking = list(walking_mapping.keys())[list(walking_mapping.values()).index(walking)]


row5_0, row5_1, row5_2, row5_3, row5_5 = st.columns((0.08, 3, 3, 3, 0.17))
with row5_1:
    st.write("#### Lối sống")

row6_0, row6_1, row6_2, row6_3, row6_5 = st.columns((0.08, 3, 3, 3, 0.17))
smoking_status = row6_1.selectbox("Tình trạng hút thuốc của bạn?", list(smoking_status_mapping.values()), index=0)
selected_smoking = list(smoking_status_mapping.keys())[list(smoking_status_mapping.values()).index(smoking_status)]

sleep_category = row6_1.selectbox("Bạn ngủ bao nhiêu giờ mỗi đêm?", list(sleep_category_mapping.values()), index=2)
selected_sleep = list(sleep_category_mapping.keys())[list(sleep_category_mapping.values()).index(sleep_category)]

drinks_category = row6_2.selectbox("Bạn uống bao nhiêu ly rượu trong một tuần?", list(drinks_category_mapping.values()), index=0)
selected_drinks = list(drinks_category_mapping.keys())[list(drinks_category_mapping.values()).index(drinks_category)]

binge_drinking_status = row6_2.selectbox("Trong 30 ngày qua, bạn có từng uống rượu theo kiểu uống say không?", list(binge_drinking_status_mapping.values()), index=1)
selected_binge_drinking = list(binge_drinking_status_mapping.keys())[list(binge_drinking_status_mapping.values()).index(binge_drinking_status)]

exercise_status = row6_3.selectbox("Trong 30 ngày qua, bạn có tập thể dục không?", list(exercise_status_mapping.values()), index=0)
selected_exercise = list(exercise_status_mapping.keys())[list(exercise_status_mapping.values()).index(exercise_status)]


# Collect input data
input_data = {
    'gender': gender,
    'race': race,
    'general_health': general_health,
    'health_care_provider': health_care_provider,
    'could_not_afford_to_see_doctor': could_not_afford_to_see_doctor,
    'length_of_time_since_last_routine_checkup': length_of_time_since_last_routine_checkup,
    'ever_diagnosed_with_heart_attack': heart_attack,
    'ever_diagnosed_with_a_stroke': stroke,
    'ever_told_you_had_a_depressive_disorder': depressive_disorder,
    'ever_told_you_have_kidney_disease': kidney_disease,
    'ever_told_you_had_diabetes': diabetes,
    'BMI': bmi,
    'difficulty_walking_or_climbing_stairs': walking,
    'physical_health_status': physical_health,
    'mental_health_status': mental_health,
    'asthma_Status': asthma,
    'smoking_status': smoking_status,
    'binge_drinking_status': binge_drinking_status,
    'exercise_status_in_past_30_Days': exercise_status,
    'age_category': age_category,
    'sleep_category': sleep_category,
    'drinks_category': drinks_category
}

def predict_heart_disease_risk(input_data, model, encoder):
    input_df = pd.DataFrame([input_data])
    input_encoded = encoder.transform(input_df, y=None, override_return_df=False)
    prediction = model.predict_proba(input_encoded)[:, 1][0] * 100
    return prediction

st.write('---')
row8_0, row8_1, row8_2, row8_5 = st.columns((0.08, 7, 5, 0.27))

with row8_1:
    st.write("#### Đánh giá nguy cơ bệnh tim bằng AI")

btn1 = row8_1.button('Nhận Đánh giá Nguy cơ Bệnh Tim của Bạn')

if btn1:
    try:
        risk = predict_heart_disease_risk(input_data, model, encoder)
        with row8_1:
            st.write(f"Dự đoán nguy cơ mắc bệnh tim: {risk:.2f}%")
            input_df = pd.DataFrame([input_data])
            input_encoded = encoder.transform(input_df, y=None, override_return_df=False)
            print(input_encoded)
            lgbm_model = model.estimators_[0].steps[-1][1]
            explainer = shap.TreeExplainer(lgbm_model)
            shap_values = explainer.shap_values(input_encoded)
            print(input_encoded)
            print(input_encoded.shape)

            # Sửa cách xử lý shap_values
            if isinstance(shap_values, list):
                feature_importances = np.abs(shap_values[0]).sum(axis=0)
            else:
                feature_importances = np.abs(shap_values).sum(axis=0)
                
            feature_importances /= feature_importances.sum()
            feature_importances *= 100
            feature_importance_df = pd.DataFrame({
                'Feature': input_encoded.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            recommendations = []
            if risk > 70:
                recommendations.append("Nguy cơ mắc bệnh tim của bạn rất cao. Dưới đây là một số lời khuyên để giảm nguy cơ của bạn:")
            elif risk > 40:
                recommendations.append("Nguy cơ mắc bệnh tim của bạn cao. Dưới đây là một số lời khuyên để giảm nguy cơ của bạn:")
            elif risk > 25:
                recommendations.append("Nguy cơ mắc bệnh tim của bạn ở mức độ trung bình. Dưới đây là một số lời khuyên để giảm nguy cơ của bạn:")
            else:
                recommendations.append("Nguy cơ mắc bệnh tim của bạn thấp. Hãy tiếp tục duy trì lối sống lành mạnh.")
            if risk > 25:
                cumulative_importance = 0
                important_features = set()
                for index, row in feature_importance_df.iterrows():
                    cumulative_importance += row['Importance']
                    important_features.add(row['Feature'])
                    if cumulative_importance >= 50:
                        break
                
                # Đảm bảo các đặc tính quan trọng chỉ được thêm một lần duy nhất
                additional_features = [
                    ('ever_told_you_had_diabetes', diabetes == "yes"),
                    ('ever_diagnosed_with_heart_attack', heart_attack == "yes"),
                    ('ever_told_you_had_a_depressive_disorder', depressive_disorder == "yes"),
                    ('ever_diagnosed_with_a_stroke', stroke == "yes"),
                    ('age_category', age_category in ["Age_55_to_59", "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79", "Age_80_or_older"]),
                    ('length_of_time_since_last_routine_checkup', length_of_time_since_last_routine_checkup in ["Age_55_to_59", "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79", "Age_80_or_older"]),
                    ('general_health', general_health in ["fair", "poor"]),
                    ('BMI', bmi in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]),
                    ('smoking_status', smoking_status != "never_smoked"),
                    ('exercise_status_in_past_30_Days', exercise_status == "no"),
                    ('binge_drinking_status', binge_drinking_status == "yes"),
                    ('drinks_category', drinks_category in ["high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"]),
                    ('sleep_category', sleep_category in ["short_sleep_4_to_5_hours", "very_short_sleep_0_to_3_hours"]),
                    ('physical_health_status', physical_health in ["1_to_13_days_not_good", "14_plus_days_not_good"]),
                    ('mental_health_status', mental_health in ["1_to_13_days_not_good", "14_plus_days_not_good"]),
                    ('asthma_Status', asthma in ["current_asthma", "former_asthma"]),
                    ('difficulty_walking_or_climbing_stairs', walking == "yes"),
                    ('length_of_time_since_last_routine_checkup', length_of_time_since_last_routine_checkup != "past_year"),
                    ('could_not_afford_to_see_doctor', could_not_afford_to_see_doctor == "yes"),
                    ('health_care_provider', health_care_provider == "no"),
                    ('ever_told_you_have_kidney_disease', kidney_disease == "yes")
                ]


                for feature, condition in additional_features:
                    if condition:
                        important_features.add(feature)

                feature_name_mapping = {
                    'ever_diagnosed_with_heart_attack': 'Nhồi máu cơ tim',
                    'general_health': 'Sức khỏe chung',
                    'ever_diagnosed_with_a_stroke': 'Đột quỵ',
                    'ever_told_you_have_kidney_disease': 'Bệnh thận',
                    'ever_told_you_had_diabetes': 'Bệnh tiểu đường',
                    'physical_health_status': 'Sức khỏe thể chất',
                    'ever_told_you_had_a_depressive_disorder': 'Rối loạn trầm cảm',
                    'sleep_category': 'Giấc ngủ',
                    'age_category': 'Độ tuổi',
                    'length_of_time_since_last_routine_checkup': 'Thời gian kiểm tra sức khỏe gần nhất',
                    'BMI': 'Chỉ số BMI',
                    'smoking_status': 'Hút thuốc',
                    'exercise_status_in_past_30_Days': 'Tập thể dục',
                    'binge_drinking_status': 'Uống rượu thái quá',
                    'drinks_category': 'Rượu bia',
                    'could_not_afford_to_see_doctor': 'Không đủ tiền để đi khám bác sĩ',
                    'health_care_provider': 'Cung cấp dịch vụ y tế',
                    'asthma_Status': 'Hen suyễn',
                    'difficulty_walking_or_climbing_stairs': 'Khó khăn khi đi bộ hoặc leo cầu thang',
                    'mental_health_status': 'Sức khỏe tâm thần'
                }

                # Đảm bảo rằng các yếu tố có khuyến nghị được đưa vào danh sách yếu tố cuối cùng
                final_features = []  # Danh sách các yếu tố quan trọng
                feature_to_recommendation = {}  # Từ điển lưu trữ khuyến nghị cho từng yếu tố

                # Duyệt qua các yếu tố quan trọng và thêm khuyến nghị cho từng yếu tố nếu cần
                for feature in important_features:
                    importance = feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0]
                    
                    # Kiểm tra và tạo khuyến nghị cho các yếu tố cụ thể
                    if feature == 'ever_diagnosed_with_heart_attack' and heart_attack == "yes":
                        recommendation = f"- Tiền sử nhồi máu cơ tim đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy thường xuyên thăm khám bác sĩ tim mạch và tuân thủ các chỉ định về thuốc. Theo dõi các triệu chứng mới hoặc tiến triển xấu và tìm sự hỗ trợ y tế ngay khi cần."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)
                    
                    if feature == 'ever_diagnosed_with_a_stroke' and stroke == "yes":
                        recommendation = f"- Tiền sử đột quỵ đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy tuân theo chỉ dẫn của bác sĩ thần kinh và dùng thuốc theo chỉ định. Thực hiện các bài tập vật lý trị liệu để phục hồi sức mạnh và khả năng vận động."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'age_category' and age_category in ["Age_55_to_59", "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79", "Age_80_or_older"]:
                        recommendation = f"- Nhóm tuổi đóng góp {importance:.2f}% vào nguy cơ của bạn. Dù không thể thay đổi độ tuổi, nhưng việc duy trì lối sống lành mạnh có thể giảm thiểu các nguy cơ liên quan đến tuổi tác. Hãy đảm bảo kiểm tra sức khỏe định kỳ, ăn uống cân đối, vận động và tránh hút thuốc."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'general_health' and general_health in ["fair", "poor"]:
                        recommendation = f"- Tình trạng sức khỏe chung đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy tập trung vào việc cải thiện sức khỏe tổng thể qua chế độ ăn uống cân bằng và kiểm tra sức khỏe định kỳ."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'ever_told_you_have_kidney_disease' and kidney_disease == "yes":
                        recommendation = f"- Bệnh thận đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy theo dõi chức năng thận thường xuyên và tuân thủ chỉ dẫn của bác sĩ để quản lý tình trạng này. Uống đủ nước và duy trì chế độ ăn uống thân thiện với thận."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'ever_told_you_had_diabetes' and diabetes == "yes":
                        recommendation = f"- Tiểu đường đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy kiểm soát tiểu đường thông qua chế độ ăn uống, luyện tập và dùng thuốc theo chỉ định của bác sĩ."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'smoking_status' and smoking_status != "never_smoked":
                        recommendation = f"- Tình trạng hút thuốc đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy bỏ thuốc để giảm nguy cơ mắc bệnh tim mạch."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'exercise_status_in_past_30_Days' and exercise_status == "no":
                        recommendation = f"- Thiếu vận động đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy tham gia vào các hoạt động thể chất thường xuyên để cải thiện sức khỏe tim mạch."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'binge_drinking_status' and binge_drinking_status == "yes":
                        recommendation = f"- Uống rượu quá mức đóng góp {importance:.2f}% vào nguy cơ của bạn. Giảm hoặc loại bỏ việc tiêu thụ rượu có thể giảm đáng kể nguy cơ bệnh tim mạch. Hãy xem xét tìm kiếm sự hỗ trợ để kiểm soát hoặc ngừng uống rượu."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'drinks_category' and drinks_category in ["high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"]:
                        recommendation = f"- Tiêu thụ rượu đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy hạn chế việc uống rượu để giảm nguy cơ."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'sleep_category' and sleep_category in ["short_sleep_4_to_5_hours", "very_short_sleep_0_to_3_hours"]:
                        recommendation = f"- Chất lượng giấc ngủ đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy cố gắng ngủ đủ 7-9 giờ mỗi đêm. Giấc ngủ đủ và chất lượng rất quan trọng đối với sức khỏe tim mạch."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'physical_health_status' and physical_health in ["1_to_13_days_not_good", "14_plus_days_not_good"]:
                        recommendation = f"- Tình trạng sức khỏe thể chất đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy tham gia vào các hoạt động thể chất thường xuyên và tham khảo ý kiến bác sĩ nếu gặp vấn đề sức khỏe thể chất kéo dài."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'mental_health_status' and mental_health in ["1_to_13_days_not_good", "14_plus_days_not_good"]:
                        recommendation = f"- Tình trạng sức khỏe tâm thần đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy tìm kiếm sự hỗ trợ từ chuyên gia sức khỏe tâm thần và thực hành các hoạt động giảm stress."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'asthma_Status' and asthma in ["current_asthma", "former_asthma"]:
                        recommendation = f"- Hen suyễn đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy tuân thủ kế hoạch điều trị hen suyễn, tránh các tác nhân gây hen và sử dụng thuốc theo chỉ định."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'ever_told_you_had_a_depressive_disorder' and depressive_disorder == "yes":
                        recommendation = f"- Rối loạn trầm cảm đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy tìm kiếm sự hỗ trợ từ chuyên gia sức khỏe tâm thần, thực hành các hoạt động giảm căng thẳng và duy trì một lối sống lành mạnh."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'difficulty_walking_or_climbing_stairs' and walking == "yes":
                        recommendation = f"- Khó khăn trong việc đi bộ hoặc leo cầu thang đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy tham khảo ý kiến bác sĩ để có các phương pháp điều trị và bài tập giúp cải thiện khả năng vận động và sức mạnh."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'length_of_time_since_last_routine_checkup' and length_of_time_since_last_routine_checkup != "past_year":
                        recommendation = f"- Thời gian từ lần kiểm tra sức khỏe gần nhất đóng góp {importance:.2f}% vào nguy cơ của bạn. Các kiểm tra sức khỏe định kỳ rất quan trọng để phát hiện sớm và quản lý các vấn đề sức khỏe. Hãy lên lịch hẹn thăm khám bác sĩ định kỳ để theo dõi và duy trì sức khỏe tim mạch."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'could_not_afford_to_see_doctor' and could_not_afford_to_see_doctor == "yes":
                        recommendation = f"- Khó khăn tài chính trong việc thăm khám bác sĩ đóng góp {importance:.2f}% vào nguy cơ của bạn. Hãy khám phá các dịch vụ y tế cộng đồng, các phòng khám theo mô hình thu phí thấp hoặc các lựa chọn bảo hiểm sức khỏe để đảm bảo bạn có quyền truy cập vào dịch vụ y tế cần thiết."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'health_care_provider' and health_care_provider == "no":
                        recommendation = f"- Không có bác sĩ chăm sóc sức khỏe chính đóng góp {importance:.2f}% vào nguy cơ của bạn. Việc xây dựng mối quan hệ với bác sĩ chăm sóc sức khỏe chính giúp quản lý và ngăn ngừa các vấn đề sức khỏe. Hãy tìm kiếm bác sĩ chăm sóc sức khỏe chính để đảm bảo việc kiểm tra và tư vấn y tế định kỳ."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                    if feature == 'BMI' and bmi in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]:
                        recommendation = f"- Chỉ số BMI đóng góp {importance:.2f}% vào nguy cơ của bạn. Duy trì một cân nặng khỏe mạnh thông qua chế độ ăn uống cân đối và luyện tập thể thao thường xuyên có thể giúp giảm nguy cơ bệnh tim mạch. Hãy tham khảo bác sĩ để nhận lời khuyên cụ thể."
                        feature_to_recommendation[feature] = recommendation
                        final_features.append(feature)

                # Tính toán tỷ lệ đóng góp còn lại cho "Yếu tố khác"
                total_importance = sum([feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0] for feature in final_features])
                other_factors_importance = 100 - total_importance

                # Chuẩn bị dữ liệu cho biểu đồ hình tròn
                pie_data = {
                    'Feature': [feature_name_mapping[feature] for feature in final_features] + ['Yếu tố khác'],
                    'Importance': [feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].values[0] for feature in final_features] + [other_factors_importance]
                }
                pie_df = pd.DataFrame(pie_data)


               # Tạo biểu đồ hình tròn
            fig = px.pie(pie_df, names='Feature', values='Importance')  #, title='Đóng góp vào Nguy cơ Bệnh Tim Mạch'

            # Hiển thị biểu đồ hình tròn
            with row8_2:
                st.write("""
                        #### Đóng góp vào Nguy cơ Bệnh Tim Mạch
                        """)
                st.plotly_chart(fig)

            # Hiển thị các khuyến nghị theo thứ tự giảm dần
            sorted_recommendations = sorted([(feature, feature_to_recommendation[feature]) for feature in final_features], key=lambda x: feature_importance_df.loc[feature_importance_df['Feature'] == x[0], 'Importance'].values[0], reverse=True)
            for feature, recommendation in sorted_recommendations:
                st.write(recommendation)
            else:
                st.write("Nguy cơ bệnh tim mạch của bạn là thấp. Tiếp tục duy trì lối sống lành mạnh và giữ gìn sức khỏe tốt.")

    except Exception as e:
        row8_1.error(e)

    # Use Local CSS File
    local_css("style.css")