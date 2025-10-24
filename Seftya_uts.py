import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import time

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
    yolo_model = YOLO("model_uts/Seftya Pratista_Laporan 4.pt.pt")
    classifier = tf.keras.models.load_model("model_uts/Seftya Pratista_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# LABEL KELAS KLASIFIKASI
# ==========================
class_names = ["Ballet Flat", "Boat", "Brogue", "Clog", "Sneaker"]

# ==========================
# SIDEBAR NAVIGASI
# ==========================
st.sidebar.image("model_uts/LOGO UUSK.jpg", use_container_width=True)
st.sidebar.title("🔧 Pengaturan Dashboard")
menu = st.sidebar.radio(
    "Pilih Mode:",
    ["▶️Deteksi Objek (YOLO)", "▶️Klasifikasi Gambar"]
)
st.sidebar.info("Unggah gambar untuk mulai analisis AI!")

# ==========================
# JUDUL HALAMAN
# ==========================
st.title("Dashboard Klasifikasi dan Deteksi Objek")
st.markdown("""
Hai!! Selamat datang di dunia **AI Vision!**  
Di sini kamu bisa lihat langsung bagaimana kecerdasan buatan bekerja mendeteksi dan mengklasifikasikan objek dari gambar.  
Unggah fotomu, biarkan AI bekerja, dan saksikan bagaimana teknologi mengenali dunia di sekitarmu! 🔍🤩
""")

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

    with col2:
        # ==========================
        # MODE: DETEKSI OBJEK
        # ==========================
        if menu == "🎯 Deteksi Objek (YOLO)":
            st.subheader("🎯 Hasil Deteksi Objek (YOLOv8)")

            with st.spinner("🔍 AI sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
                time.sleep(1)

            st.success(f"✅ Deteksi selesai! Ditemukan {len(detections)} objek.")

            # Tampilkan gambar hasil deteksi dengan efek transisi
            placeholder = st.empty()
            placeholder.image(result_img, caption="📦 Hasil Deteksi YOLO", use_container_width=True)

            # Jika ada objek terdeteksi
            if len(detections) > 0:
                st.markdown("### 📋 Rincian Hasil Deteksi")
                det_table = []
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    label = yolo_model.names[int(cls)]
                    det_table.append({
                        "Label": label,
                        "Confidence": f"{conf*100:.2f}%",
                        "Koordinat (x1, y1, x2, y2)": f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}"
                    })
                st.table(det_table)
            else:
                st.warning("⚠️ Tidak ada objek terdeteksi pada gambar ini.")

        # ==========================
        # MODE: KLASIFIKASI GAMBAR
        # ==========================
        elif menu == "🧩 Klasifikasi Gambar":
            st.subheader("🧩 Hasil Klasifikasi Gambar")

            with st.spinner("🤖 AI sedang mengklasifikasikan gambar..."):
                target_size = classifier.input_shape[1:3]
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                time.sleep(1)

            st.metric(label="Kategori Prediksi", value=class_names[class_index])
            st.progress(float(confidence))
            st.write(f"**Tingkat Keyakinan Model:** {confidence*100:.2f}%")

            st.write("🔢 Probabilitas per kelas:")
            prob_dict = {class_names[i]: f"{prediction[0][i]*100:.2f}%" for i in range(len(class_names))}
            st.json(prob_dict)

else:
    st.info("👆 Silakan unggah gambar terlebih dahulu untuk memulai analisis AI.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<center><b>Seftya Pratista | 2208108010054</b><br>Proyek UAS Praktikum Pemrograman Big Data | Universitas Syiah Kuala</center>",
    unsafe_allow_html=True
)
