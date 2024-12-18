import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Fungsi untuk memuat model berdasarkan pilihan pengguna (h5 dan keras)
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "MobileNet": 'E:/Model Kentang/potato_disease_model_kyknya_bagus.h5',  # Ganti dengan path model .h5
        "ConvNextBase": 'E:/Model Kentang/pp.keras',  # Ganti dengan path model .keras
        "VGG": 'E:/Model Kentang/my_model.h5'  # Ganti dengan path model .h5
    }
    
    model_path = model_paths.get(model_name)
    if model_path:
        # Cek ekstensi file dan muat model sesuai dengan ekstensi
        if model_path.endswith('.h5') or model_path.endswith('.keras'):
            return keras.models.load_model(model_path)
        else:
            st.error("Format model tidak dikenali. Harus dengan ekstensi .h5 atau .keras.")
            return None
    else:
        st.error("Model tidak ditemukan.")
        return None

# Fungsi preprocessing gambar
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img.convert('RGB'))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Fungsi untuk klasifikasi gambar
def classify_image(model, img):
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    classes = ['Early_Blight', 'Sehat', 'Late_Blight']
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Uji Daun Kentang",
    layout="wide",
    page_icon="🥔"
)

# CSS untuk menyelaraskan konten ke tengah
st.markdown("""
    <style>
    /* Selaraskan semua elemen ke tengah */
    .block-container {
        max-width: 800px;
        margin: auto;
        padding-top: 50px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header aplikasi
st.title("🌿 Uji Daun Kentang dengan AI")
st.markdown("""
    Unggah gambar daun kentang dan pilih model CNN untuk menganalisis kondisinya.
    - **Early Blight**: Gejala awal penyakit hawar.
    - **Sehat**: Daun dalam kondisi sehat.
    - **Late Blight**: Penyakit hawar lanjut.
""")

# Pilihan model menggunakan dropdown
st.header("1. Pilih Model CNN")
model_choice = st.selectbox(
    "Pilih model CNN yang ingin digunakan:",
    ["MobileNet", "ConvNextBase", "VGG"],
    help="Dropdown ini menyediakan pilihan model Convolutional Neural Network (CNN)."
)

# Unggah gambar
st.header("2. Unggah Gambar")
uploaded_file = st.file_uploader(
    "Pilih gambar daun kentang (format: JPG, PNG, JPEG):",
    type=["jpg", "png", "jpeg"]
)

# Jika ada file yang diunggah
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

# Proses prediksi
if uploaded_file:
    with st.spinner(f"Memuat model {model_choice}..."):
        model = load_model(model_choice)

    if model:
        with st.spinner("Menganalisis gambar..."):
            predicted_class, confidence = classify_image(model, img)

        # Tampilkan hasil
        st.success("Analisis selesai!")
        st.markdown(f"""
            ### **Hasil Prediksi**
            - **Kondisi Daun:** :green[{predicted_class}]
            - **Tingkat Kepercayaan:** :blue[{confidence:.2f}%]
        """)

        # Visualisasi hasil dengan matplotlib
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f"Prediksi: {predicted_class}\nKepercayaan: {confidence:.2f}%", 
            fontsize=14, color="blue"
        )
        st.pyplot(fig)

# Footer aplikasi
st.markdown("---")
st.markdown("""
    **Aplikasi Uji Daun Kentang** | Dibuat untuk mendukung deteksi dini penyakit tanaman.
    📧 Hubungi: [email@example.com](mailto:email@example.com)
""")
