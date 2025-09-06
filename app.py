import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import re
import emoji
import plotly.express as px
import pandas as pd
import time

st.set_page_config(
    page_title="Prediksi Emosi Komentar Publik soal Kasus Tom Lembong",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .sidebar-info {
        background-color: #f1f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    repo_id = "Wijdanadam/Fine-tuned_indoBERTweet_for_Emotion_Classification"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="saved_model")
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder="saved_model")
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

labels = ['SADNESS', 'ANGER', 'SUPPORT', 'HOPE', 'DISAPPOINTMENT']

def preprocess_indoBertweet(text: str) -> str:
    text = text.lower()
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'http\S+|www\S+|https\S+', 'HTTPURL', text)
    text = emoji.demojize(text, language='id')
    return text

def predict_emotion(text: str):
    processed_text = preprocess_indoBertweet(text)
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        inference_time = time.time() - start

    pred_class = int(probs.argmax())
    pred_prob = float(probs[pred_class])
    return pred_class, pred_prob, probs, inference_time

def display_results(pred_class, pred_prob, probs, inference_time):
    st.markdown(f"""
    <div class="prediction-box">
        <h2>Prediksi Teratas</h2>
        <h1 style="color:#007bff; margin:0;">{labels[pred_class]}</h1>
        <h3 style="color:#6c757d; margin:0;">Confidence: {pred_prob:.1%}</h3>
        <p style="margin-top:1rem;">Inference time: {inference_time*1000:.1f} ms</p>
    </div>
    """, unsafe_allow_html=True)

    fig = px.bar(
        x=probs,
        y=labels,
        orientation="h",
        title="Confidence Scores (%)",
        color=probs,
        color_continuous_scale="Blues"
    )
    fig.update_layout(showlegend=False, height=350, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    df = pd.DataFrame([
        {"Label": labels[i], "Confidence": f"{probs[i]:.1%}"} for i in range(len(labels))
    ])
    with st.expander("Detail Prediksi"):
        st.dataframe(df, use_container_width=True, hide_index=True)

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-info">
        <b>Model:</b> IndoBERTweet<br>
        <b>Jumlah kelas:</b> {len(labels)}<br>
        <b>Device:</b> {"GPU" if torch.cuda.is_available() else "CPU"}<br>
        <b>Max sequence length:</b> 128<br>
    </div>
    """, unsafe_allow_html=True)

    st.info("Pilih mode inferensi di bawah ini.")
    mode = st.radio("Mode inferensi", ["Satu Kalimat", "Dataset CSV"])

st.markdown('<h1 class="main-header">Prediksi Emosi Komentar Publik soal Kasus Tom Lembong</h1>', unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center;'>Model klasifikasi emosi berbasis IndoBERTweet</h5>", unsafe_allow_html=True)

st.image(
        "https://assets.pikiran-rakyat.com/crop/0x0:0x0/703x0/webp/photo/2025/02/15/3013879046.jpg",
        use_container_width=True
    )

if mode == "Satu Kalimat":
    st.markdown("## Input Teks")
    user_input = st.text_area("Tulis teks di sini...", height=150, placeholder="Contoh: Aku merasa sangat senang hari ini 😊")

    if st.button("Predict", type="primary"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            pred_class, pred_prob, probs, inference_time = predict_emotion(user_input)
            display_results(pred_class, pred_prob, probs, inference_time)

elif mode == "Dataset CSV":
    st.markdown("## Upload Dataset CSV")
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'text'", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("File CSV harus memiliki kolom 'text'")
        else:
            if st.button("Proses Dataset", type="primary"):
                results = []
                st.info("Sedang memproses dataset...")
                progress_bar = st.progress(0)

                for i, text in enumerate(df["text"].astype(str)):
                    pred_class, pred_prob, probs, _ = predict_emotion(text)
                    results.append({
                        "Text": text,
                        "Predicted_Label": labels[pred_class],
                        "Confidence": f"{pred_prob:.1%}"
                    })
                    progress_bar.progress((i+1)/len(df))

                result_df = pd.DataFrame(results)
                st.success("Selesai memproses dataset")
                st.dataframe(result_df, use_container_width=True)

                csv_download = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Hasil CSV",
                    data=csv_download,
                    file_name="prediksi_dataset.csv",
                    mime="text/csv"
                )

with st.expander("Tentang Kasus"):
    st.write("""
    Pada 2024, publik digemparkan oleh kasus Tom Lembong, mantan Menteri Perdagangan yang divonis bersalah dalam dugaan korupsi impor gula namun kemudian memperoleh abolisi dari Presiden. Keputusan tersebut memicu perdebatan sengit, sebagian menilai ia hanyalah korban kriminalisasi politik, sementara yang lain kecewa pada sistem hukum dan keputusan pengadilan. Di media sosial, emosi publik tumpah ruah dari kemarahan, kesedihan, dan kekecewaan, hingga harapan akan reformasi hukum serta dukungan penuh sebagai bentuk solidaritas terhadap Tom.
    """)

with st.expander("Tentang Model"):
    st.write("""
    Model ini dilatih menggunakan **IndoBERTweet** untuk melakukan klasifikasi emosi pada teks bahasa Indonesia.
    
    **Kelas yang dikenali:**
    - SADNESS
    - ANGER
    - SUPPORT
    - HOPE
    - DISAPPOINTMENT
    """)
