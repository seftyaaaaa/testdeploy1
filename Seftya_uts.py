import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import cv2

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Dashboard Klasifikasi dan Deteksi Objek",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model_uts/Seftya Pratista_Laporan 4.pt.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model_uts/Seftya Pratista_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# DAFTAR LABEL KELAS
# ==========================
class_names = ["Ballet Flat", "Boat", "Brogue", "Clog", "Sneaker"]

# ==========================
# SIDEBAR NAVIGASI
# ==========================
st.sidebar.image("model_uts/LOGO UUSK.jpg", use_container_width=True)
menu = st.sidebar.radio(
    "🧭 Pilih Mode Analisis:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]
)
st.sidebar.info("📤 Unggah gambar sesuai mode yang dipilih untuk mulai analisis AI.")

# ==========================
# HEADER UTAMA
# ==========================
st.title("Dashboard Klasifikasi dan Deteksi Objek")
st.markdown(""" Hai!! Selamat datang di dunia **AI Vision!** Di sini kamu bisa lihat langsung bagaimana kecerdasan buatan bekerja mendeteksi dan mengklasifikasikan objek dari gambar. 
Unggah fotomu, biarkan AI bekerja, dan saksikan bagaimana teknologi mengenali dunia di sekitarmu! 🔍🤩 """)

uploaded_file = st.file_uploader("📂 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES INPUT GAMBAR
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="🖼️ Gambar Diupload", use_container_width=True)

    # ==========================
    # MODE DETEKSI OBJEK (YOLO)
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        with st.spinner("🚀 AI sedang mendeteksi objek... harap tunggu sebentar!"):
            results = yolo_model(img)
            result_img = results[0].plot()
            boxes = results[0].boxes

        with col2:
            if boxes is not None and len(boxes) > 0:
                st.image(result_img, caption="📦 Hasil Deteksi Objek", use_container_width=True)
                st.success(f"✅ {len(boxes)} objek berhasil terdeteksi!")

                # Buat tabel hasil deteksi
                data = []
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    data.append({
                        "Label": results[0].names[cls],
                        "Confidence": f"{conf*100:.2f}%",
                        "x1": int(xyxy[0]),
                        "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]),
                        "y2": int(xyxy[3])
                    })
                
                df = pd.DataFrame(data)
                st.subheader("📋 Rincian Hasil Deteksi")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.markdown("> 💡 Semakin tinggi nilai *confidence*, semakin yakin model terhadap deteksi tersebut.")
            else:
                st.warning("⚠️ Tidak ada objek yang terdeteksi dalam gambar ini.")
                st.markdown("> 🚫 Coba unggah gambar dengan objek yang lebih jelas atau berbeda sudut pandang.")

    # ==========================
    # MODE KLASIFIKASI GAMBAR
    # ==========================
    elif menu == "Klasifikasi Gambar":
        with st.spinner("🧠 Sedang melakukan klasifikasi gambar..."):
            target_size = classifier.input_shape[1:3]
            img_resized = img.resize(target_size)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        with col2:
            st.subheader("📊 Hasil Klasifikasi")
            st.metric(label="Kategori Prediksi", value=class_names[class_index])
            st.progress(float(confidence))
            st.write(f"**Tingkat Keyakinan Model:** {confidence*100:.2f}%")

            st.write("🔢 Probabilitas per kelas:")
            prob_dict = {class_names[i]: f"{prediction[0][i]*100:.2f}%" for i in range(len(class_names))}
            st.json(prob_dict)

else:
    st.info("👆 Silakan unggah gambar terlebih dahulu untuk mulai deteksi atau klasifikasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<center>created by <b>Seftya Pratista | 2208108010054</b><br>"
    "Proyek UAS Praktikum Pemrograman Big Data | Universitas Syiah Kuala</center>",
    unsafe_allow_html=True
)
