import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    pkg = joblib.load("deployment_model/clustering_deployment.joblib")
    return pkg["model"], pkg["scaler"], pkg["feature_columns"]

model, scaler, feature_columns = load_model()

st.title("Segmentasi Pelanggan Online Retail")
st.write("Prediksi cluster pelanggan menggunakan K-Means berdasarkan RFM.")

recency   = st.number_input("Recency (hari sejak transaksi terakhir)", min_value=0,   value=15)
frequency = st.number_input("Frequency (jumlah transaksi)",            min_value=1,   value=8)
monetary  = st.number_input("Monetary (total belanja, £)",             min_value=0.0, value=2500.00, step=10.0)

if st.button("Prediksi Cluster"):
    d = pd.DataFrame({"Recency":[recency], "Frequency":[frequency], "Monetary":[monetary]})
    d["Frequency_log"] = np.log1p(d["Frequency"])      # sama persis pipeline training (Sel 8)
    d["Monetary_log"]  = np.log1p(d["Monetary"])
    cluster = int(model.predict(scaler.transform(d[feature_columns]))[0])

    meta = {
        0: ("Pelanggan Reguler (Core)",        "blue",   "Pertahankan hubungan, berikan rekomendasi produk silang (cross-selling)."),
        1: ("Pelanggan Tidak Aktif (At-Risk)", "orange", "Kirimkan kampanye reaktivasi ('We miss you') atau diskon khusus."),
        2: ("Pelanggan VIP (High Value)",      "green",  "Berikan layanan eksklusif, reward loyalitas, dan prioritas dukungan."),
    }
    nama, warna, saran = meta[cluster]
    st.success(f"Pelanggan ini masuk ke **Cluster {cluster}**")
    st.markdown(f"### Segmentasi: :{warna}[{nama}]")
    st.info(f"💡 **Rekomendasi Bisnis:** {saran}")

    with st.expander("Lihat Detail Perhitungan"):
        st.markdown(
            f"| Fitur | Nilai |\n|---|---|\n"
            f"| Recency | {recency} |\n| Frequency | {frequency} |\n| Monetary | {monetary:,.2f} |\n"
            f"| Frequency_log | {np.log1p(frequency):.4f} |\n| Monetary_log | {np.log1p(monetary):.4f} |"
        )
        st.write("Nilai setelah scaling (urutan fitur model): "
                 + ", ".join(f"{v:.4f}" for v in scaler.transform(d[feature_columns])[0]))
