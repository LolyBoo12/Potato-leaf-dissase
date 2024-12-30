import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Fungsi untuk memuat model berdasarkan pilihan pengguna (h5 dan keras)
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "MobileNet": 'mobilenet.h5',  
        "ConvNextBase": 'convnextbase.h5',  
        "VGG": 'vgg.h5'  
    }
    
    model_path = model_paths.get(model_name)
    if model_path:
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
    return prediction[0], predicted_class, confidence

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Uji Daun Kentang",
    layout="wide",
    page_icon="🥔"
)

# CSS untuk menyelaraskan konten ke tengah
st.markdown("""
    <style>
    .block-container {
        max-width: 800px;
        margin: auto;
        padding-top: 50px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header aplikasi
st.title("🌿 Klasifikasi Penyakit Daun Kentang ")
st.markdown("""
    Unggah gambar daun kentang dan pilih model CNN untuk menganalisis kondisinya.
    \n **Early Blight**: Gejala awal penyakit hawar.
    \n **Sehat**: Daun dalam kondisi sehat.
    \n **Late Blight**: Penyakit hawar lanjut.
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

# Tombol submit untuk memulai analisis setelah gambar diunggah
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width =True)

    # Tombol untuk memulai proses
    if st.button("Submit"):
        with st.spinner(f"Memuat model {model_choice}..."):
            model = load_model(model_choice)

        if model:
            with st.spinner("Menganalisis gambar..."):
                class_predictions, predicted_class, confidence = classify_image(model, img)

            # Tampilkan hasil
            st.success("Analisis selesai!")
            st.markdown(f"""
                ### **Hasil Prediksi**
                 \n**Kondisi Daun:** :green[{predicted_class}]
                 \n**Tingkat Kepercayaan:** :blue[{confidence:.2f}%]
            """)

            # Tampilkan confidence dengan bilah horizontal (progress bar)
            st.markdown("### **Confidence per Kelas**")
            confidence_data = {
                'Early Blight': class_predictions[0],
                'Sehat': class_predictions[1],
                'Late Blight': class_predictions[2]
            }

            # Menampilkan progress bar dan nilai confidence
            for class_name, confidence_value in confidence_data.items():
                st.markdown(f"**{class_name}: {confidence_value * 100:.2f}%**")
                st.progress(int(confidence_value * 100))

            # Visualisasi gambar dengan prediksi
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
# st.markdown("""
#     **Aplikasi Uji Daun Kentang** | Dibuat untuk mendukung deteksi dini penyakit tanaman.
#     📧 Hubungi: [dickyu63@gmail.com](mailto:dickyu63@gmail.com)
# """)
