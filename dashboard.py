import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import time

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.set_page_config(page_title="AI Vision App", page_icon="🧠", layout="centered")
st.title("🧠 AI Vision App: Object Detection & Image Classification")

menu = st.sidebar.radio("Pilih Mode:", ["🔍 Deteksi Objek (YOLO)", "📸 Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📂 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    with st.spinner("🔄 Memproses gambar..."):
        time.sleep(1)

        if menu == "🔍 Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
            st.success("✅ Deteksi selesai!")

        elif menu == "📸 Klasifikasi Gambar":
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))

            st.write("### 🎯 Hasil Prediksi:")
            st.metric(label="Kelas", value=f"{class_index}")
            st.metric(label="Probabilitas", value=f"{confidence*100:.2f}%")
            st.success("✅ Klasifikasi selesai!")
else:
    st.info("👆 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")
