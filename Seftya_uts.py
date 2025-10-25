import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import datetime
import torch
import os

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Dashboard Klasifikasi dan Deteksi Objek",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CSS THEME (BIRU LEMBUT)
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #b3c7ff, #8e9fff);
    color: #f2f4ff;
    font-family: 'Arial', sans-serif;
    padding: 2rem 3rem;
    margin: 0;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

@keyframes fadeSlideIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

.main-title, .section-title, .detect-result, .explain-box {
    transition: all 0.3s ease-in-out;
    transform: scale(1);
    animation: fadeSlideIn 0.8s ease forwards;
}

.main-title:hover, .section-title:hover, .detect-result:hover, .explain-box:hover {
    transform: scale(1.03);
    box-shadow: 6px 6px 15px rgba(0,0,0,0.12);
}

.main-title {
    background: linear-gradient(145deg, #6b7bd6, #4f63c8);
    border: 2px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    color: #f8fbff;
    text-align: center;
    padding: 20px;
    font-size: 28px;
    font-weight: bold;
    margin: 20px auto 25px auto;
    box-shadow: 4px 6px 12px rgba(0,0,0,0.08);
    width: 100%;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    background-color: rgba(255,255,255,0.06);
    padding: 8px 15px;
    border-radius: 12px;
    color: #f2f6ff;
    margin-bottom: 15px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.04);
}

.explain-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 15px;
    color: #f2f6ff;
    margin-top: 10px;
    font-weight: 500;
    text-align: justify;
}

.footer-center { text-align: center; color: #eef2ff; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ==========================
# SESSION STATE INIT
# ==========================
for key, val in {
    'page': 'home',
    'user_name': '',
    'user_campus': '',
    'used_feature': False,
    'last_mode': None,
    'last_pred': None,
    'last_conf': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================
# LOAD MODELS (dengan perbaikan PyTorch 2.6)
# ==========================
@st.cache_resource
def load_models():
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])

        # Cek keberadaan file model
        deteksi_path = "model_uts/deteksi.pt"
        klasifikasi_path = "model_uts/klasifikasi.h5"

        if not os.path.exists(deteksi_path):
            st.error("❌ File model deteksi tidak ditemukan. Pastikan 'model_uts/deteksi.pt' ada.")
            return None, None
        if not os.path.exists(klasifikasi_path):
            st.error("❌ File model klasifikasi tidak ditemukan. Pastikan 'model_uts/klasifikasi.h5' ada.")
            return None, None

        # Patch torch.load untuk kompatibilitas PyTorch 2.6
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = patched_torch_load

        try:
            yolo_model = YOLO(deteksi_path)
        except Exception as e:
            raise RuntimeError(f"Model YOLO gagal dimuat. Kemungkinan file rusak: {e}")

        torch.load = original_torch_load  # Kembalikan fungsi torch.load

        try:
            classifier = tf.keras.models.load_model(klasifikasi_path)
        except Exception as e:
            raise RuntimeError(f"Model klasifikasi gagal dimuat: {e}")

        st.success("✅ Model berhasil dimuat dan siap digunakan!")
        return yolo_model, classifier

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None


yolo_model, classifier = load_models()

# ==========================
# CLASS LABELS
# ==========================
class_names = ["Ballet Flat", "Boat", "Brogue", "Clog", "Sneaker"]

# ==========================
# PAGES
# ==========================
def halaman_awal():
    st.markdown("<div class='main-title'>👟 <b>SELAMAT DATANG DI DASHBOARD SEFTYA PRATISTA</b> 👟</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Aplikasi Deteksi & Klasifikasi Sepatu (YOLO + CNN)</div>', unsafe_allow_html=True)
    st.markdown('<div class="explain-box">Halo! Ini adalah aplikasi demo yang memperlihatkan bagaimana model deteksi (YOLO) dan klasifikasi (CNN) bekerja pada citra sepatu. Klik tombol di bawah untuk melanjutkan ke proses registrasi singkat sebelum menggunakan dashboard.</div>', unsafe_allow_html=True)
    if st.button("Lanjut ke Registrasi →"):
        st.session_state['page'] = 'register'


def halaman_registrasi():
    st.markdown('<div class="main-title">📝 Registrasi Singkat</div>', unsafe_allow_html=True)
    with st.form("form_reg"):
        name = st.text_input("Nama Lengkap", value=st.session_state.get('user_name', ''))
        campus = st.text_input("Nama Kampus", value=st.session_state.get('user_campus', ''))
        submitted = st.form_submit_button("Masuk ke Dashboard Utama")
        if submitted:
            if not name.strip() or not campus.strip():
                st.warning("Mohon isi Nama dan Kampus sebelum melanjutkan.")
            else:
                st.session_state['user_name'] = name.strip()
                st.session_state['user_campus'] = campus.strip()
                st.session_state['page'] = 'main'

    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state['page'] = 'home'


def halaman_main():
    st.sidebar.image("model_uts/LOGO UUSK.jpg", use_container_width=True)
    menu = st.sidebar.radio("🧭 Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("📂 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="🖼️ Gambar Diupload", use_container_width=True)

        if menu == "Deteksi Objek (YOLO)":
            if yolo_model is None:
                st.error("Model YOLO belum tersedia.")
            else:
                with st.spinner("🚀 AI sedang mendeteksi objek..."):
                    results = yolo_model(np.array(img))
                    result_img = results[0].plot()
                    boxes = results[0].boxes

                with col2:
                    if boxes is not None and len(boxes) > 0:
                        st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
                        st.success(f"✅ {len(boxes)} objek berhasil terdeteksi!")
                    else:
                        st.warning("⚠️ Tidak ada objek terdeteksi.")

        elif menu == "Klasifikasi Gambar":
            if classifier is None:
                st.error("Model klasifikasi belum tersedia.")
            else:
                with st.spinner("🧠 Sedang melakukan klasifikasi..."):
                    target_size = classifier.input_shape[1:3]
                    img_resized = img.resize(target_size)
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    prediction = classifier.predict(img_array)
                    class_index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction)) * 100.0

                with col2:
                    st.subheader("📊 Hasil Klasifikasi")
                    st.metric("Kategori Prediksi", class_names[class_index])
                    st.progress(confidence / 100.0)
                    st.write(f"**Tingkat Keyakinan Model:** {confidence:.2f}%")
    else:
        st.info("👆 Silakan unggah gambar terlebih dahulu.")

    if st.button("⬅️ Kembali ke Registrasi"):
        st.session_state['page'] = 'register'


# ==========================
# ROUTING
# ==========================
if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'register':
    halaman_registrasi()
elif st.session_state['page'] == 'main':
    halaman_main()

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<center>created by <b>Seftya Pratista | 2208108010054</b><br>Proyek UAS Praktikum Pemrograman Big Data | Universitas Syiah Kuala</center>",
    unsafe_allow_html=True
)
