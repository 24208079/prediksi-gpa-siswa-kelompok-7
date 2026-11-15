
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI DASHBOARD
# ==========================================

st.set_page_config(
    page_title="Prediksi GPA Siswa",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# JUDUL DAN DESKRIPSI
# ==========================================

st.title("🎓 Dashboard Prediksi GPA Siswa")

st.write("""
Dashboard ini digunakan untuk memprediksi nilai Grade Point Average (GPA)
siswa berdasarkan beberapa faktor, seperti usia, pendidikan orang tua,
waktu belajar, jumlah ketidakhadiran, dukungan orang tua, dan aktivitas siswa.

Model prediksi yang digunakan adalah Linear Regression hasil
Hyperparameter Tuning.
""")

st.divider()

# ==========================================
# MEMUAT MODEL DAN DATASET
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("model_final.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_proyek.csv")

paket_model = load_model()
model = paket_model["model"]
scaler = paket_model["scaler"]
feature_columns = paket_model["feature_columns"]

df = load_dataset()

# ==========================================
# INFORMASI DATASET
# ==========================================

st.subheader("📊 Informasi Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", len(df))

with col2:
    st.metric("Jumlah Fitur Model", len(feature_columns))

with col3:
    st.metric("Model Prediksi", "Linear Regression")

# ==========================================
# VISUALISASI 1 - DISTRIBUSI GPA
# ==========================================

st.subheader("Visualisasi 1: Distribusi GPA Siswa")

fig1, ax1 = plt.subplots(figsize=(8, 4))

ax1.hist(df["GPA"], bins=20, edgecolor="black")

ax1.set_title("Distribusi Nilai GPA Siswa")
ax1.set_xlabel("GPA")
ax1.set_ylabel("Jumlah Siswa")

st.pyplot(fig1)
plt.close(fig1)

# ==========================================
# VISUALISASI 2 - STUDY TIME DAN GPA
# ==========================================

st.subheader("Visualisasi 2: Hubungan Waktu Belajar dan GPA")

fig2, ax2 = plt.subplots(figsize=(8, 4))

ax2.scatter(
    df["StudyTimeWeekly"],
    df["GPA"],
    alpha=0.6
)

ax2.set_title("Hubungan Study Time Weekly dengan GPA")
ax2.set_xlabel("Waktu Belajar per Minggu")
ax2.set_ylabel("GPA")

st.pyplot(fig2)
plt.close(fig2)

st.divider()

# ==========================================
# INPUT DATA MELALUI SIDEBAR
# ==========================================

st.sidebar.header("📝 Input Data Siswa")

st.sidebar.write(
    "Masukkan informasi siswa untuk melakukan prediksi GPA."
)

age = st.sidebar.number_input(
    "Usia",
    min_value=10,
    max_value=30,
    value=17
)

parental_education = st.sidebar.selectbox(
    "Pendidikan Orang Tua",
    [0, 1, 2, 3, 4]
)

study_time = st.sidebar.number_input(
    "Waktu Belajar per Minggu",
    min_value=0.0,
    max_value=50.0,
    value=10.0
)

absences = st.sidebar.number_input(
    "Jumlah Ketidakhadiran",
    min_value=0,
    max_value=50,
    value=5
)

parental_support = st.sidebar.selectbox(
    "Dukungan Orang Tua",
    [0, 1, 2, 3, 4]
)

gender = st.sidebar.selectbox(
    "Gender",
    [0, 1]
)

ethnicity = st.sidebar.selectbox(
    "Ethnicity",
    [0, 1, 2, 3, 4]
)

tutoring = st.sidebar.selectbox(
    "Mengikuti Tutoring",
    [0, 1]
)

extracurricular = st.sidebar.selectbox(
    "Mengikuti Extracurricular",
    [0, 1]
)

sports = st.sidebar.selectbox(
    "Mengikuti Sports",
    [0, 1]
)

music = st.sidebar.selectbox(
    "Mengikuti Music",
    [0, 1]
)

volunteering = st.sidebar.selectbox(
    "Mengikuti Volunteering",
    [0, 1]
)

# ==========================================
# MENYIAPKAN DATA INPUT
# ==========================================

input_data = pd.DataFrame([{
    "Age": age,
    "ParentalEducation": parental_education,
    "StudyTimeWeekly": study_time,
    "Absences": absences,
    "ParentalSupport": parental_support,
    "Gender_1": gender,

    "Ethnicity_1": int(ethnicity == 1),
    "Ethnicity_2": int(ethnicity == 2),
    "Ethnicity_3": int(ethnicity == 3),

    "Tutoring_1": tutoring,
    "Extracurricular_1": extracurricular,
    "Sports_1": sports,
    "Music_1": music,
    "Volunteering_1": volunteering
}])

# Menyesuaikan urutan kolom dengan model
input_data = input_data[feature_columns]

# ==========================================
# TOMBOL PREDIKSI
# ==========================================

st.sidebar.divider()

if st.sidebar.button("🔍 Prediksi GPA", use_container_width=True):

    # Scaling data input menggunakan scaler dari Tahap 3
    input_scaled = scaler.transform(input_data)

    # Melakukan prediksi
    hasil_prediksi = model.predict(input_scaled)[0]

    st.subheader("🎯 Hasil Prediksi GPA")

    st.success(
        f"Prediksi GPA siswa adalah: {hasil_prediksi:.2f}"
    )

    st.info(
        "Hasil ini merupakan estimasi berdasarkan pola yang dipelajari "
        "model Linear Regression dari data pelatihan."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Proyek Data Science | Prediksi GPA Siswa | "
    "CRISP-DM Tahap 6 - Deployment"
)
