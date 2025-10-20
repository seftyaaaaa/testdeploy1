import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import io

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
# Fungsi Tambahan
# ==========================
def add_annotation(image_pil, text, font_size=20, position="bottom"):
    """Menambahkan teks anotasi ke gambar"""
    img = image_pil.convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text_w, text_h = draw.textsize(text, font=font)
    x = (width - text_w) // 2

    if position == "bottom":
        y = height - text_h - 10
    else:
        y = 10

    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)

        # Konversi ke PIL untuk anotasi
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

        # Tambah anotasi sesuai kelas
        detected_labels = results[0].boxes.cls.tolist()
        class_names = yolo_model.names

        if len(detected_labels) > 0:
            for c in detected_labels:
                label_name = class_names[int(c)].lower()

                if label_name == "forest":
                    text = ("“Di dalam hutan yang terdiri dari ribuan pohon, tak ada dua daun pun yang sama. "
                            "Dan tak ada dua perjalanan melewati jalur sama pun yang serupa.” - Paulo Coelho")
                    result_pil = add_annotation(result_pil, text, font_size=16, position="bottom")

                elif label_name == "desert":
                    text = ("Wilayah kering dengan curah hujan sangat rendah (kurang dari 250 mm per tahun), "
                            "suhu ekstrem (panas di siang hari dan dingin di malam hari), kelembapan rendah, "
                            "dan tanah tandus yang tidak mampu menyimpan air.")
                    result_pil = add_annotation(result_pil, text, font_size=16, position="bottom")

        st.image(result_pil, caption="Hasil Deteksi dan Anotasi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)

        # Misalnya label klasifikasi sepatu
        labels = ["Sepatu Sneakers", "Sepatu Sandal", "Sepatu Formal"]
        predicted_label = labels[class_index] if class_index < len(labels) else "Tidak diketahui"

        st.write("### Hasil Prediksi:", predicted_label)
        st.write("Probabilitas:", float(np.max(prediction)))

        # Tambahan informasi label sepatu
        st.info("""
        **Tempat Pembelian:** Sepatu tersedia di *Matahari Department Store*  
        **Ukuran Tersedia:** 35 - 42  
        """)
