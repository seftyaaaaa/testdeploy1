import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model_uts/Seftya Pratista_Laporan 4.pt.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model_uts/Seftya Pratista_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # ==========================
    # MODE DETEKSI OBJEK (YOLO)
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)

        # Konversi ke PIL untuk anotasi tambahan
        annotated_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(annotated_img)

        labels = [r.names[int(cls)] for cls in results[0].boxes.cls]

        # Font untuk teks (gunakan default PIL agar aman)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Tambahkan anotasi berdasarkan label
        if any("forest" in label.lower() for label in labels):
            quote = (
                '"Di dalam hutan yang terdiri dari ribuan pohon, tak ada dua daun pun yang sama. '
                'Dan tak ada dua perjalanan melewati jalur sama pun yang serupa."\n- Paulo Coelho'
            )
            draw.text((20, annotated_img.height - 80), quote, fill="white", font=font)

        if any("desert" in label.lower() for label in labels):
            desc = (
                "Wilayah kering dengan curah hujan sangat rendah (kurang dari 250 mm per tahun), "
                "suhu ekstrem (panas di siang hari dan dingin di malam hari), kelembapan rendah, "
                "dan tanah tandus yang tidak mampu menyimpan air."
            )
            draw.text((20, annotated_img.height - 60), desc, fill="white", font=font)

        st.image(annotated_img, caption="Hasil Deteksi dengan Anotasi", use_container_width=True)

    # ==========================
    # MODE KLASIFIKASI GAMBAR
    # ==========================
    elif menu == "Klasifikasi Gambar":
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Contoh label (sesuaikan dengan label model kamu)
        labels = ["Sepatu Sport", "Sepatu Formal", "Sandal", "Boots"]
        predicted_label = labels[class_index] if class_index < len(labels) else f"Label {class_index}"

        st.write("### Hasil Prediksi:", predicted_label)
        st.write(f"Probabilitas: **{confidence:.2f}**")

        # Tambahan keterangan sepatu
        st.info(
            f"🛍️ Produk ini tersedia di **Matahari Department Store** "
            f"dengan pilihan ukuran **35 hingga 42**."
        )
