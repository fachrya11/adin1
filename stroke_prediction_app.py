import pandas as pd
import streamlit as st
import pickle


# Fungsi untuk memproses input pengguna
def preprocess_input(input_data, label_encoders, categorical_columns, data_columns):
    input_df = pd.DataFrame([input_data])
    for column in categorical_columns:
        input_df[column] = label_encoders[column].transform(input_df[column])
    if 'bmi' in input_df.columns and input_df['bmi'].isnull().any():
        input_df['bmi'].fillna(data_columns['bmi'].median(), inplace=True)
    return input_df[data_columns.columns]

# Load model, label encoders, categorical columns, dan unique values
with open('stroke_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('categorical_columns.pkl', 'rb') as f:
    categorical_columns = pickle.load(f)

with open('unique_values.pkl', 'rb') as f:
    unique_values = pickle.load(f)

# Load dataset untuk mendapatkan informasi kolom
file_path = 'full_data.csv'
data = pd.read_csv(file_path)
data_columns = data.drop(columns=['stroke'])

# Aplikasi Streamlit
st.title("Aplikasi Prediksi Penyakit Stroke")

st.write("Masukkan data pasien untuk memprediksi apakah pasien berisiko terkena stroke atau tidak.")

# Input pengguna
user_input = {
    'age': st.slider('Usia', 0, 100, 25),
    'gender': st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan']),
    'hypertension': st.selectbox('Tingkat Hipertensi', ['Belum Kritis', 'Hipertensi Level 1', 'Hipertensi Level 2', 'Hipertensi Kritis'], index=0),
    'heart_disease': st.selectbox('Penyakit Jantung', ['Tidak', 'Ya']),
    'ever_married': st.selectbox('Pernah Menikah', ['Ya', 'Tidak']),
    'work_type': st.selectbox('Jenis Pekerjaan', ['Swasta', 'Wiraswasta', 'Pekerja Pemerintahan/Kantoran', 'Anak-anak', 'Belum Pernah Bekerja']),
    'Residence_type': st.selectbox('Tempat Tinggal', ['Perkotaan', 'Pedesaan']),
    'avg_glucose_level': st.number_input('Rata-rata Level Glukosa', min_value=0.0, max_value=300.0, value=100.0),
    'bmi': st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0),
    'smoking_status': st.selectbox('Status Merokok', ['Sebelumnya Merokok', 'Tidak Pernah Merokok', 'Merokok', 'Tidak Diketahui'])
}

# Map input ke nilai yang digunakan saat pelatihan
hypertension_map = {
    'Belum Kritis': 0,
    'Hipertensi Level 1': 1,
    'Hipertensi Level 2': 2,
    'Hipertensi Kritis': 3
}
user_input['hypertension'] = hypertension_map[user_input['hypertension']]

gender_map = {
    'Laki-laki': 'Male',
    'Perempuan': 'Female'
}
user_input['gender'] = gender_map[user_input['gender']]

heart_disease_map = {
    'Tidak': 0,
    'Ya': 1
}
user_input['heart_disease'] = heart_disease_map[user_input['heart_disease']]

ever_married_map = {
    'Ya': 'Yes',
    'Tidak': 'No'
}
user_input['ever_married'] = ever_married_map[user_input['ever_married']]

work_type_map = {
    'Swasta': 'Private',
    'Wiraswasta': 'Self-employed',
    'Pekerja Pemerintahan/Kantoran': 'Govt_job',
    'Anak-anak': 'children',
    'Belum Pernah Bekerja': 'Never_worked'
}
user_input['work_type'] = work_type_map[user_input['work_type']]

residence_type_map = {
    'Perkotaan': 'Urban',
    'Pedesaan': 'Rural'
}
user_input['Residence_type'] = residence_type_map[user_input['Residence_type']]

smoking_status_map = {
    'Sebelumnya Merokok': 'formerly smoked',
    'Tidak Pernah Merokok': 'never smoked',
    'Merokok': 'smokes',
    'Tidak Diketahui': 'Unknown'
}
user_input['smoking_status'] = smoking_status_map[user_input['smoking_status']]

# Validasi input pengguna
if st.button('Prediksi'):
    processed_input = preprocess_input(user_input, label_encoders, categorical_columns, data_columns)
    prediction = model.predict(processed_input)
    if prediction == 1:
        st.write("Pasien berisiko terkena stroke.")
    else:
        st.write("Pasien tidak berisiko terkena stroke.")
