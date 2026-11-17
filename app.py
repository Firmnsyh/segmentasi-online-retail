import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    # PERBAIKAN DI SINI: Hapus "deployment_model/" karena file ada di root repo
    pkg = joblib.load("clustering_deployment.joblib") 
    return pkg["model"], pkg["scaler"], pkg["feature_columns"]

model, scaler, feature_columns = load_model()

# --- UI STREAMLIT ---
st.title("Segmentasi Pelanggan Online Retail")
st.write("Prediksi cluster pelanggan menggunakan K-Means berdasarkan RFM.")

col1, col2, col3 = st.columns(3)
with col1:
    recency = st.number_input("Recency (Hari)", min_value=0, value=30)
with col2:
    frequency = st.number_input("Frequency (Transaksi)", min_value=1, value=2)
with col3:
    monetary = st.number_input("Monetary (£)", min_value=0.0, value=500.0, step=10.0)

if st.button("🚀 Prediksi Cluster"):
    input_data = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })
    
    # Transformasi Log (Sama seperti training)
    input_data["Frequency_log"] = np.log1p(input_data["Frequency"])
    input_data["Monetary_log"] = np.log1p(input_data["Monetary"])
    
    # Scaling & Predict
    input_scaled = scaler.transform(input_data[feature_columns])
    prediction = int(model.predict(input_scaled)[0])
    
    # Mapping Nama Cluster
    if prediction == 0:
        nama, warna, saran = ("Pelanggan Reguler (Core)", "blue", "Pertahankan hubungan, berikan rekomendasi produk silang (cross-selling).")
    elif prediction == 1:
        nama, warna, saran = ("Pelanggan Tidak Aktif (At-Risk)", "orange", "Kirimkan kampanye reaktivasi ('We miss you') atau diskon khusus.")
    else:
        nama, warna, saran = ("Pelanggan VIP (High Value)", "green", "Berikan layanan eksklusif, reward loyalitas, dan prioritas dukungan.")

    st.success(f"Hasil: Pelanggan ini masuk ke **Cluster {prediction}**")
    st.markdown(f"### Segmentasi: :{warna}[{nama}]")
    st.info(f"💡 **Rekomendasi Bisnis:** {saran}")
