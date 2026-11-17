import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression as Regresi

# Tampilan Judul Web
st.title("Aplikasi Regresi Linear: Prediksi Harga Rental PS")
st.write("Menggunakan library `scikit-learn`")

# 1. Data Historis
x = np.array([[2], [4], [6], [4], [6], [2], [4], [3]])
y = np.array([10000, 20000, 30000, 20000, 30000, 10000, 20000, 15000])

st.subheader("1. Data Transaksi Historis")
# Menampilkan data dalam bentuk kolom teks sejajar
col1, col2 = st.columns(2)
col1.write("**Durasi (Jam):**")
col1.write(x.flatten())
col2.write("**Billing (Rupiah):**")
col2.write(y)

# 2. Proses Pelatihan Model
model = Regresi()
model.fit(x, y)

st.subheader("2. Hasil Pelatihan Model")
st.write(f"**Intercept (Konstanta):** {model.intercept_:.2f}")
st.write(f"**Slope (Kemiringan):** {model.coef_[0]:.2f}")

# 3. Prediksi Interaktif
st.subheader("3. Prediksi Harga Baru")
# Mengganti angka paten '8' dengan input interaktif (default 8)
input_durasi = st.number_input("Masukkan durasi rental (jam):", min_value=1, value=8)

durasi_baru = np.array([[input_durasi]])
harga_prediksi = model.predict(durasi_baru)
st.success(f"Prediksi harga untuk {input_durasi} jam: **Rp {harga_prediksi[0]:,.0f}**")

# 4. Evaluasi Galat
st.subheader("4. Evaluasi Model")
y_prediksi_semua = model.predict(x)
galat = y - y_prediksi_semua
jumlah_galat = np.sum(galat)
st.info(f"Jumlah total galat (residual): **{jumlah_galat:.2f}**")