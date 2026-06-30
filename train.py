import json
import numpy as np
import pickle
import random
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# 2. Kamus Slang 
slang_dict = {
    "gmn": "bagaimana", "gmana": "bagaimana", "gimana": "bagaimana",
    "brp": "berapa", "brpa": "berapa", "brapa": "berapa",
    "ga": "tidak", "gk": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak", "gabisa": "tidak bisa",
    "klo": "kalau", "kalo": "kalau", "klw": "kalau",
    "bwt": "buat", "utk": "untuk", "yg": "yang", "dgn": "dengan",
    "skbm": "sukabumi", "maba": "mahasiswa baru", "prodi": "program studi",
    "univ": "universitas", "pake": "pakai", "pakean": "pakaian",
    "bgt": "banget", "udh": "sudah", "udah": "sudah", "dftr": "daftar",
    "nyari": "cari", "tau": "tahu", "nemu": "temu", "hub": "hubung"
}

# 3. Fungsi Preprocessing 
def preprocess_text(text):
    text = text.lower() # Case Folding: Lowercasing
    text = re.sub(r'(.)\1+', r'\1', text) # Elongation Removal: Menghapus huruf berulang (e.g., "haloooo" -> "halo")
    text = re.sub(r'[^a-z\s]', '', text) # Punctuation Stripping: Menghapus tanda baca, angka, dan simbol
    tokens = text.split() # Tokenization
    
    # Normalization slang_dict
    normalized_tokens = [slang_dict[word] if word in slang_dict else word for word in tokens]
    
    # Stemming: Mengubah kata berimbuhan menjadi kata dasar (e.g., "pendaftaran" -> "daftar")
    stemmed_tokens = [stemmer.stem(word) for word in normalized_tokens]
    
    # Stopwords Filter: Menghapus kata tidak penting
    ignore_words = ['sih', 'ya', 'kah', 'deh', 'dong', 'tuh', 'kok', 'p', 'ping']
    final_tokens = [word for word in stemmed_tokens if word not in ignore_words and len(word) > 1]
    
    return final_tokens

# 4. Memuat File Data Intents JSON
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

words = []
classes = []
documents = []

for intent in data['intents']:
    for pattern in intent['input']:
        stemmed_tokens = preprocess_text(pattern)  # Memproses setiap baris pertanyaan dataset
        words.extend(stemmed_tokens)  # Memasukkan semua kata dasar ke list besar 'words'
        documents.append((stemmed_tokens, intent['tag']))  # Menyimpan pasangan (kata, kategori)
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  

# Menghapus duplikasi kata dan mengurutkannya secara alfabetis
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# 5. Ekstraksi Fitur Menjadi Representasi Angka (Bag of Words)
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    
    # Beri nilai 1 jika kata unik ada di dalam kalimat user, beri 0 jika tidak ada
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mengacak urutan data agar pembagian bobot saat training optimal
random.shuffle(training)

# Konversi ke struktur array NumPy
X_train = np.array([item[0] for item in training])
y_train = np.array([item[1] for item in training])

# 6. Membangun Arsitektur Neural Network (ANN)
model = Sequential([
    Dense(128, input_shape=(len(X_train[0]),), activation='relu'),
    Dropout(0.5), # Mencegah overfitting data
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax') # Softmax untuk klasifikasi probabilitas multi-kelas
])

# Kompilasi setelan Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train Model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Simpan hasil model terlatih
model.save('chatbot_pmb_model.h5')
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))