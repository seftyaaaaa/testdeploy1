import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import io

# ==========================
# CONFIG STREAMLIT
# ==========================
st.set_page_config(
    page_title="AI Vision App",
    page_icon="🧠",
    layout="wide",
)

# ==========================
# LOAD MODELS (with cache)
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek YOLOv8
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI HEADER
# ==========================
st.title("🧠 AI Vision App")
st.markdown("### ✨ Deteksi Objek & Klasifikasi Gambar berbasis Deep Learning")

menu = st.sidebar.radio("📌 Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# Ganti dengan label kelas model klasifikasimu
class_labels = ['Kucing', 'Anjing', 'Burung', 'Ikan']

# ==========================
# MAIN APP
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    col1, col2 = st.columns(2)

    with st.spinner("🔍 Sedang memproses gambar..."):
        if menu == "Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()

            with col1:
                st.image(result_img, caption="✅ Hasil Deteksi", use_container_width=True)

            with col2:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    data = []
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        data.append({
                            "Kelas": yolo_model.names[cls],
                            "Confidence (%)": round(conf * 100, 2),
                            "Koordinat (x1, y1, x2, y2)": [round(x, 2) for x in xyxy]
                        })
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Tidak ada objek terdeteksi.")

            buf = io.BytesIO()
            Image.fromarray(result_img).save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="💾 Unduh Hasil Deteksi",
                data=byte_im,
                file_name="hasil_deteksi.png",
                mime="image/png"
            )

        elif menu == "Klasifikasi Gambar":
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.pred
