import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# ==========================
# 🔧 Load Models
# ==========================
@st.cache_resource
def load_models():
    # Pastikan path sesuai dengan nama file kamu di folder model_uts
    yolo_model = YOLO("model_uts/Seftya Pratista_Laporan 4.pt.pt")   # Model YOLO (.pt)
    classifier = tf.keras.models.load_model("model_uts/Seftya Pratista_Laporan 2.h5")  # Model Klasifikasi (.h5)
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# 🎨 Fungsi Tambahan
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

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (width - text_w) // 2
    y = height - text_h - 10 if position == "bottom" else 10

    # Tambahkan background semi-transparan agar teks lebih jelas
    draw.rectangle([(0, y - 5), (width, y + text_h + 5)], fill=(0, 0, 0, 150))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img

# ==========================
# 🧠 STREAMLIT UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    # ==========================
    # 🔹 MODE YOLO (DETEKSI OBJEK)
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan bounding box)
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

        # Ambil label hasil deteksi
        labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]

        # Tambahkan anotasi otomatis tergantung label deteksi
        if any("forest" in label.lower() for label in labels):
            quote = (
                '"Di dalam hutan yang terdiri dari ribuan pohon, tak ada dua daun pun yang sama. '
                'Dan tak ada dua perjalanan melewati jalur sama pun yang serupa."\n- Paulo Coelho'
            )
            result_pil = add_annotation(result_pil, quote, font_size=16)

        elif any("desert" in label.lower() for label in labels):
            desc = (
                "Wilayah kering dengan curah hujan sangat rendah (kurang dari 250 mm per tahun), "
                "suhu ekstrem (panas di siang hari dan dingin di malam hari), kelembapan rendah, "
                "dan tanah tandus yang tidak mampu menyimpan air."
            )
            result_pil = add_annotation(result_pil, desc, font_size=16)

        st.image(result_pil, caption="🎯 Hasil Deteksi Objek (YOLO)", use_container_width=True)

    # ==========================
    # 🔹 MODE CNN (KLASIFIKASI)
    # ==========================
    elif menu == "Klasifikasi Gambar":
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        # Label contoh — sesuaikan dengan label dataset kamu
        labels = ["Sepatu Sport", "Sepatu Formal", "Sandal", "Boots"]
        predicted_label = labels[class_index] if class_index < len(labels) else f"Label {class_index}"

        st.subheader("🧩 Hasil Klasifikasi")
        st.write("**Prediksi:**", predicted_label)
        st.write(f"**Probabilitas:** {confidence:.2f}")

        st.info(
            f"🛍️ Produk ini tersedia di **Matahari Department Store** "
            f"dengan pilihan ukuran **35 hingga 42**."
        )
