import os
import json
import re
import pickle
import random
import numpy as np
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import load_model

# Config & UI Styling
st.set_page_config(
    page_title="PMB Nusa Putra - Assistant",
    page_icon="🎓",
    layout="centered"
)

# Custom styling 
st.markdown("""
    <style>
    @import url('https://googleapis.com');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stApp {
        background-color: #FAFAFB;
    }
    .main-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 15px;
    }
    .title-text {
        color: #0F172A;
        font-size: 26px !important;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .desc-text {
        color: #64748B;
        font-size: 14px;
    }
    .status-badge {
        display: inline-block;
        background-color: #DCFCE7;
        color: #15803D;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
    }
    .stChatInputContainer {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.02) !important;
    }
    .stChatMessage {
        border-radius: 12px !important;
        padding: 12px !important;
        margin-bottom: 10px !important;
    }
    [data-testid="stChatMessageUser"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border-bottom-right-radius: 2px !important;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border-bottom-left-radius: 2px !important;
        border: 1px solid #F1F5F9 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <div class="title-text">🎓 PMB Nusa Putra Virtual Assistant</div>
        <div class="desc-text">Tanya jawab seputar pendaftaran mahasiswa baru Universitas Nusa Putra</div>
        <span class="status-badge">● Online 24 Jam</span>
    </div>
""", unsafe_allow_html=True)


# Load Model & Setup NLP
@st.cache_resource
def init_bot():
    # Load semua file binary hasil training
    bot_model = load_model('chatbot_pmb_model.h5')
    vocab = pickle.load(open('words.pkl', 'rb'))
    tags = pickle.load(open('classes.pkl', 'rb'))
    
    with open('intents.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    stemmer = StemmerFactory().create_stemmer()
    return bot_model, vocab, tags, dataset, stemmer

try:
    model, words, classes, intents, stemmer = init_bot()
except Exception:
    st.error("Gagal meload file model. Pastikan file model, words, classes, dan intents sudah lengkap di folder prodi.")
    st.stop()

# Kamus slang
slang_words = {
    "gmn": "bagaimana", "gmana": "bagaimana", "gimana": "bagaimana",
    "brp": "berapa", "brpa": "berapa", "brapa": "berapa",
    "ga": "tidak", "gk": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak", "gabisa": "tidak bisa",
    "klo": "kalau", "kalo": "kalau", "klw": "kalau",
    "bwt": "buat", "utk": "untuk", "yg": "yang", "dgn": "dengan",
    "skbm": "sukabumi", "maba": "mahasiswa baru", "prodi": "program studi",
    "univ": "universitas", "pake": "pakai", "bgt": "banget", "udh": "sudah",
    "dftr": "daftar", "nyari": "cari", "tau": "tahu", "hub": "hubung"
}

stop_words = ['sih', 'ya', 'kah', 'deh', 'dong', 'tuh', 'kok', 'p', 'ping']


# Core Logic Functions 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'(.)\1+', r'\1', text) # bersihin huruf dobel (misal: halooo)
    text = re.sub(r'[^a-z\s]', '', text)  # buang tanda baca
    
    # normalisasi slang & stemming
    tokens = text.split()
    normalized = [slang_words[w] if w in slang_words else w for w in tokens]
    stemmed = [stemmer.stem(w) for w in normalized]
    
    # filter stopword
    return [w for w in stemmed if w not in stop_words and len(w) > 1]

def get_bow(text):
    clean_tokens = clean_text(text)
    bag = * len(words)
    for token in clean_tokens:
        for idx, w in enumerate(words):
            if w == token:
                bag[idx] = 1
    return np.array(bag)

def check_intent(text):
    bag = get_bow(text)
    prediction = model.predict(np.array([bag]), verbose=0)
    
    # ambil index dengan probabilitas tertinggi di atas threshold 60%
    best_idx = np.argmax(prediction)
    confidence = prediction[0][best_idx]
    
    if confidence > 0.60:
        return classes[best_idx]
    return "fallback"

def fetch_reply(tag):
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Maaf, ada kendala internal sistem."


# Chat Interface State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "sender": "bot",
            "text": "👋 Halo! Ada yang bisa dibantu seputar info pendaftaran, beasiswa, atau biaya kuliah di Universitas Nusa Putra?"
        }
    ]

# Tampilkan history obrolan
for msg in st.session_state.chat_history:
    icon = "👤" if msg["sender"] == "user" else "🤖"
    role_type = "user" if msg["sender"] == "user" else "assistant"
    with st.chat_message(role_type, avatar=icon):
        st.markdown(msg["text"])

# Shortcut pertanyaan / Quick Replies
st.markdown("<p style='font-size: 13px; font-weight:600; color:#64748B; margin-bottom: 5px;'>💡 Pertanyaan populer:</p>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

shortcut = None
with c1:
    if st.button("📌 Cara Daftar", use_container_width=True):
        shortcut = "Bagaimana alur dan cara daftar kuliah di nusa putra?"
with c2:
    if st.button("💰 Biaya Kuliah", use_container_width=True):
        shortcut = "Rincian biaya kuliah per semester berapa ya?"
with c3:
    if st.button("🎓 Info Beasiswa", use_container_width=True):
        shortcut = "Apakah ada beasiswa maba di nusa putra?"


# Chat Input Handler
user_query = st.chat_input("Ketik pertanyaan kamu disini...")

if shortcut:
    user_query = shortcut

if user_query:
    # 1. Tampilkan pesan user
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"sender": "user", "text": user_query})
    
    # 2. Proses nyari balasan via NN
    predicted_tag = check_intent(user_query)
    reply = fetch_reply(predicted_tag)
    
    # 3. Tampilkan balasan bot
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(reply)
    st.session_state.chat_history.append({"sender": "bot", "text": reply})