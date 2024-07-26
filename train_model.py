import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Load dataset
file_path = 'full_data.csv'
data = pd.read_csv(file_path)

# Mengonversi kolom kategori menjadi numerik
label_encoders = {}
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Mengisi nilai yang hilang dengan median
data['bmi'].fillna(data['bmi'].median(), inplace=True)

# Memisahkan fitur dan label
X = data.drop(columns=['stroke'])
y = data['stroke']

# Membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Mengumpulkan nilai unik untuk setiap kolom kategori
unique_values = {}
for column in categorical_columns:
    unique_values[column] = data[column].unique()

# Menyimpan model, label encoders, categorical columns, dan unique values
with open('stroke_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('categorical_columns.pkl', 'wb') as f:
    pickle.dump(categorical_columns, f)

with open('unique_values.pkl', 'wb') as f:
    pickle.dump(unique_values, f)


print("Model, label encoders, categorical columns, dan unique values berhasil disimpan.")


